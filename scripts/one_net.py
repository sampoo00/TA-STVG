import argparse
import json
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, ToTensor

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from utils.checkpoint import VSTGCheckpointer
from utils.logger import setup_logger
from utils.misc import NestedTensor
from models import build_model, build_postprocessors
from engine.evaluate import single_forward, linear_interp


def parse_args():
    parser = argparse.ArgumentParser(description="Single Video Grounding Inference (TA-STVG)")
    parser.add_argument(
        "--video", "-v",
        required=True,
        type=str,
        help="Path to input video file (.mp4, .mkv, .avi, etc.)"
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        type=str,
        help="Text prompt / query sentence to ground in the video"
    )
    parser.add_argument(
        "--config-file",
        default="experiments/hcstvg.yaml",
        type=str,
        help="Path to config file (default: experiments/hcstvg.yaml)"
    )
    parser.add_argument(
        "--weight",
        default="ckpts/TASTVG_HCSTVG.pth",
        type=str,
        help="Path to model checkpoint (default: ckpts/TASTVG_HCSTVG.pth)"
    )
    parser.add_argument(
        "--subject",
        default=None,
        type=str,
        help="Optional subject noun (e.g. 'man', 'woman', 'person'). If omitted, auto-detected or left neutral."
    )
    parser.add_argument(
        "--output-json", "-o",
        default=None,
        type=str,
        help="Path to save grounding results as a JSON file"
    )
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="default",
        default=None,
        help="Export video (.mp4) consisting only of grounded frames with bounding box drawn. Optionally specify output path."
    )
    parser.add_argument(
        "--save-frame",
        action="store_true",
        help="If set, saves each grounded frame with bounding box drawn as {video_name}_frame{frame#}.jpg in output directory"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run inference on ('cuda' or 'cpu')"
    )
    args, unknown_opts = parser.parse_known_args()
    args.opts = unknown_opts
    return args


def load_video_frames(video_path):
    """
    Decodes all frames from the video using OpenCV (with fallback).
    Returns:
        frames (list of np.ndarray in RGB): decoded frames
        fps (float): video frames per second
        width (int): video width
        height (int): video height
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR; convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    if fps <= 0 or np.isnan(fps):
        # Default fallback for HC-STVG (typical 20s clips) or 24 fps
        fps = len(frames) / 20.0 if len(frames) > 0 else 24.0

    return frames, fps, width, height


def sample_and_preprocess_frames(frames, fps, cfg):
    """
    Samples frames according to HC-STVG test-time sampling rate,
    resizes according to cfg.INPUT.RESOLUTION, and normalizes them.
    """
    total_frames = len(frames)
    input_fps = cfg.INPUT.SAMPLE_FPS * 2  # Test mode uses double sampling rate
    video_fps = fps if fps > 0 else 24.0
    sampling_rate = input_fps / video_fps

    # Replicate make_hcstvg_input_clip sampling slice logic
    sample_slice = [0]
    for idx in range(total_frames):
        if int(sample_slice[-1] * sampling_rate) < int(idx * sampling_rate):
            sample_slice.append(idx)

    if sample_slice[-1] != total_frames - 1:
        sample_slice.append(total_frames - 1)

    resolution = cfg.INPUT.RESOLUTION
    max_rate = 1.4
    orig_h, orig_w = frames[0].shape[:2]
    rate = orig_w / orig_h
    target_size = (resolution, min(int(resolution * rate), int(resolution * max_rate)))

    to_tensor = ToTensor()
    resize_op = Resize(target_size, antialias=True)

    tensor_frames = []
    for f_idx in sample_slice:
        img_tensor = to_tensor(frames[f_idx])  # [C, H, W], values in [0, 1]
        img_tensor = resize_op(img_tensor)
        tensor_frames.append(img_tensor)

    # Stack: [T, C, H, W]
    stacked_frames = torch.stack(tensor_frames)

    # ImageNet Normalize
    mean = cfg.INPUT.PIXEL_MEAN
    std = cfg.INPUT.PIXEL_STD
    normalized_frames = TF.normalize(stacked_frames, mean=mean, std=std)

    return normalized_frames, sample_slice


def save_grounded_video(raw_frames, grounded_trajectory, output_video_path, fps):
    """
    Collects only the grounded frames, draws predicted bounding boxes, and exports them as an MP4 video.
    """
    if not grounded_trajectory:
        print("[Warning] No grounded frames found to save video.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    height, width = raw_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for item in grounded_trajectory:
        fid = item['frame']
        if fid < 0 or fid >= len(raw_frames):
            continue

        frame_bgr = cv2.cvtColor(raw_frames[fid], cv2.COLOR_RGB2BGR)

        box = item['bbox']
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        # Draw green bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Target (F:{fid})"
        cv2.putText(
            frame_bgr, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Draw timestamp and frame info
        info_str = f"Frame: {fid} | Time: {fid / fps:.2f}s"
        cv2.putText(
            frame_bgr, info_str, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        out.write(frame_bgr)

    out.release()
    print(f"Grounded video ({len(grounded_trajectory)} frames) saved to: {output_video_path}")


def save_grounded_frames(raw_frames, grounded_trajectory, video_path, output_dir, fps):
    """
    Saves grounded frames as JPEG images with predicted bounding boxes drawn.
    Naming format: {video_basename}_frame{frame#}.jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []

    for item in grounded_trajectory:
        fid = item['frame']
        if fid < 0 or fid >= len(raw_frames):
            continue

        # raw_frames are in RGB; convert to BGR for OpenCV saving
        frame_bgr = cv2.cvtColor(raw_frames[fid], cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]

        box = item['bbox']
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        # Draw green bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Target (F:{fid})"
        cv2.putText(
            frame_bgr, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Draw timestamp and frame info
        info_str = f"Frame: {fid} | Time: {fid / fps:.2f}s"
        cv2.putText(
            frame_bgr, info_str, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        filename = f"{video_stem}_frame{fid}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame_bgr)
        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} grounded frame images to: {output_dir}/")
    return saved_paths


@torch.no_grad()
def run_grounding(args):
    # 1. Config & Device setup
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.weight:
        cfg.MODEL.WEIGHT = args.weight
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(args.device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger = setup_logger("Video Grounding One-Net", None, 0)

    # 2. Build model and load weights
    model, _, _ = build_model(cfg)
    model.to(device)
    model.eval()

    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, with_optim=False)

    # Setup subject if specified or fallback
    vid_name = os.path.basename(args.video)
    if hasattr(model, 'verb_label2'):
        if vid_name not in model.verb_label2:
            model.verb_label2[vid_name] = {
                'sub': args.subject if args.subject else '',
                'verb_index_list': [],
                'adj_index_list': []
            }
        elif args.subject:
            model.verb_label2[vid_name]['sub'] = args.subject

    # 3. Load video frames
    raw_frames, fps, orig_w, orig_h = load_video_frames(args.video)
    total_frames = len(raw_frames)

    # 4. Sample and preprocess
    normalized_frames, sample_slice = sample_and_preprocess_frames(raw_frames, fps, cfg)
    videos = NestedTensor.from_tensor_list([normalized_frames]).to(device)

    # 5. Dual-pass subsampling (even/odd) as in evaluate.py
    videos1 = videos.subsample(2, start_idx=0)
    videos2 = videos.subsample(2, start_idx=1)

    frame_ids1 = sample_slice[0::2]
    frame_ids2 = sample_slice[1::2]

    targets1 = [{
        'item_id': 0,
        'ori_size': (orig_h, orig_w),
        'vid': vid_name,
        'qtype': 'none',
        'frame_ids': frame_ids1,
        'actioness': torch.ones(len(frame_ids1), dtype=torch.long, device=device),
        'eval': True,
    }]

    targets2 = [{
        'item_id': 0,
        'ori_size': (orig_h, orig_w),
        'vid': vid_name,
        'qtype': 'none',
        'frame_ids': frame_ids2,
        'actioness': torch.ones(len(frame_ids2), dtype=torch.long, device=device),
        'eval': True,
    }]

    postprocessor = build_postprocessors()

    # 6. Forward passes
    bbox_pred1, att_pred1, temp_pred1, _ = single_forward(
        cfg, model, videos1, [args.prompt], targets1, device, postprocessor
    )
    bbox_pred2, att_pred2, temp_pred2, _ = single_forward(
        cfg, model, videos2, [args.prompt], targets2, device, postprocessor
    )

    # 7. Merge predictions & linear interpolation
    bbox_pred1[0].update(bbox_pred2[0])
    interp_boxes = linear_interp(bbox_pred1[0])  # Dict: frame_id -> [[x1, y1, x2, y2]]

    pred_st_frame = min(temp_pred1[0]['sted'][0], temp_pred2[0]['sted'][0])
    pred_ed_frame = max(temp_pred1[0]['sted'][1], temp_pred2[0]['sted'][1])

    start_frame = max(0, min(pred_st_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(pred_ed_frame, total_frames))

    start_time = float(start_frame / fps)
    end_time = float(end_frame / fps)

    # 8. Assemble output results
    # Bounding box array during the grounded temporal segment [start_frame, end_frame]
    grounded_trajectory = []
    bbox_array = []
    min_interp_fid = min(interp_boxes.keys())
    max_interp_fid = max(interp_boxes.keys())

    for fid in range(start_frame, end_frame):
        # Clip to available frame ids in interp_boxes
        lookup_fid = max(min_interp_fid, min(max_interp_fid, fid))
        box = [round(float(coord), 2) for coord in interp_boxes[lookup_fid][0]]
        grounded_trajectory.append({
            "frame": fid,
            "time": round(float(fid / fps), 3),
            "bbox": box  # [x1, y1, x2, y2]
        })
        bbox_array.append(box)

    result = {
        "video": args.video,
        "prompt": args.prompt,
        "fps": round(float(fps), 2),
        "total_frames": total_frames,
        "video_duration": round(float(total_frames / fps), 2),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "bbox": bbox_array,
        "trajectory": grounded_trajectory,
    }

    # Determine output directory: if --output-json is specified, save frames to the same directory
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
    elif cfg.OUTPUT_DIR:
        output_dir = cfg.OUTPUT_DIR
    else:
        output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]

    # Optional save grounded frame images (.jpg) with bounding boxes drawn
    if args.save_frame:
        saved_images = save_grounded_frames(
            raw_frames, grounded_trajectory, args.video, output_dir, fps
        )
        result["saved_images_count"] = len(saved_images)
        result["output_dir"] = output_dir

    # Optional export video (.mp4) consisting only of grounded frames
    if args.save_video:
        if args.save_video == "default":
            video_save_path = os.path.join(output_dir, f"{video_stem}_grounded.mp4")
        else:
            video_save_path = args.save_video
        save_grounded_video(
            raw_frames, grounded_trajectory, video_save_path, fps
        )
        result["saved_video_path"] = video_save_path

    # Optional JSON save
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result JSON saved to: {args.output_json}")

    return result


def main():
    args = parse_args()
    result = run_grounding(args)

    # Print summary output to stdout
    print("\n" + "=" * 60)
    print("           VIDEO GROUNDING RESULT")
    print("=" * 60)
    print(f"Video           : {result['video']}")
    print(f"Prompt          : {result['prompt']}")
    print(f"FPS             : {result['fps']}")
    print(f"Total Frames    : {result['total_frames']} ({result['video_duration']}s)")
    print(f"Start Time (sec): {result['start_time']}s (frame: {result['start_frame']})")
    print(f"End Time (sec)  : {result['end_time']}s (frame: {result['end_frame']})")
    print(f"Grounding Duration: {result['end_time'] - result['start_time']:.2f}s")
    print(f"Grounded Frames : {len(result['bbox'])} frames")
    if result.get("saved_images_count"):
        print(f"Saved Images    : {result['saved_images_count']} files -> {result['output_dir']}/")
    if result.get("saved_video_path"):
        print(f"Saved Video     : {result['saved_video_path']}")
    if len(result['bbox']) > 0:
        print(f"First BBox [x1, y1, x2, y2]: {result['bbox'][0]}")
        print(f"Last  BBox [x1, y1, x2, y2]: {result['bbox'][-1]}")
    print("=" * 60)

    # Print exact JSON output structure
    output_summary = {
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "bbox": result["bbox"]
    }
    print("\n[Output JSON Summary]")
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
