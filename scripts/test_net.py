import argparse
import os

import torch
import torch.backends.cudnn as cudnn

import sys
# Ensure project root (which contains `config/`, `utils/`, ...) is on sys.path
# when running as `python scripts/test_net.py` (sys.path[0] would be scripts/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from utils.comm import synchronize, get_rank, is_main_process
from utils.logger import setup_logger
from utils.misc import mkdir, set_seed
from utils.checkpoint import VSTGCheckpointer
from datasets import make_data_loader, build_evaluator, build_dataset
from models import build_model, build_postprocessors
from engine import do_eval


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    parser.add_argument(
        "--config-file",
        default="experiments/vidstg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="If use the random seed",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # torchrun sets LOCAL_RANK/WORLD_SIZE via env, old launch passes --local-rank arg.
    # Prefer env when present so both launchers work.
    if "LOCAL_RANK" in os.environ:
        try:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        except ValueError:
            pass
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # Clamp to physically visible GPUs: nvidia-smi shows only 1 GPU here,
    # so --nproc_per_node=8 makes ranks 1..7 fail with
    # "CUDA error: invalid device ordinal" at torch.cuda.set_device().
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if num_gpus > visible:
            print(
                f"[warn] WORLD_SIZE={num_gpus} > visible GPUs={visible}. "
                f"Falling back to single-process (or set --nproc_per_node={visible})."
            )
            num_gpus = 1
            args.distributed = False
            args.local_rank = 0
        else:
            args.distributed = num_gpus > 1
    else:
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("Video Grounding", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model, _, _ = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, with_optim=False)
    
    # Prepare the dataset cache
    if args.local_rank == 0:
        _ = build_dataset(cfg, split='test', transforms=None)
        
    synchronize()
    
    test_data_loader = make_data_loader(
        cfg,
        mode='test',
        is_distributed=args.distributed,
    )
    
    logger.info("Start Testing")
    evaluator = build_evaluator(cfg, logger, mode='test')  # mode = ['val','test']
    postprocessor = build_postprocessors()
    do_eval(
        cfg,
        mode='test',
        logger=logger,
        model=model,
        postprocessor=postprocessor,
        data_loader=test_data_loader,
        evaluator=evaluator,
        device=device
    )
    synchronize()


if __name__ == "__main__":
    main()
