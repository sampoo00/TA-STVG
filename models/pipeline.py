import torch
from torch import nn
from .net_utils import MLP
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder, build_TemporalSampling, build_SpatialActivation
from utils.misc import NestedTensor
from .vidswin.video_swin_transformer import vidswin_model
import json
import os

import torch

def precision_recall(predicted_labels, true_labels):
    try:
        predicted_set = set(predicted_labels)
        true_set = set(true_labels)

        intersection_size = len(predicted_set.intersection(true_set))
    
        if len(predicted_set) == 0:
            precision = 0
        else:
            precision = intersection_size / len(predicted_set)
        if len(true_set) == 0:
            recall = 0
        else:
            recall = intersection_size / len(true_set)
    except Exception as e:
        print(e)
        precision, recall = 0, 0
    return precision, recall

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

class TASTVGNet(nn.Module):
    """
    [TA-STVG: Target-Aware Spatio-Temporal Video Grounding]
    ICLR 2025 Oral 논문:
    "Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding"
    
    주요 특징:
    - 기존 Transformer 기반 STVG 모델들이 영벡터(zeros)나 무작위 학습 파라미터로 디코더 쿼리를 초기화했던 것과 달리,
      비디오 및 텍스트로부터 타깃 객체의 시공간 단서(Target-Aware Cues)를 사전에 추출하여 초기 쿼리를 생성합니다.
    - 1) Text-guided Temporal Sampling: 텍스트와 비디오의 상호작용으로 타깃 액션이 일어나는 프레임을 동적으로 샘플링.
    - 2) Attribute-aware Spatial Activation: 선택된 프레임에서 외형/동작 속성에 주목하여 공간 활성화 맵 생성.
    - 3) Target-Aware Object Query Generation: 어텐션 가중합을 통해 타깃 중심의 초기 공간(Spatial) 및 시간(Temporal) 쿼리 생성.
    - 4) Iterative Refinement: 1차 디코딩된 Actionness 점수를 바탕으로 프레임을 재필터링하고 쿼리를 정제하여 최종 디코딩 수행.
    """

    def __init__(self, cfg):
        super(TASTVGNet, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN     # 비디오 최대 프레임 수
        self.use_attn = cfg.SOLVER.USE_ATTN             # 어텐션 맵 가이드 손실 사용 여부
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS     # Transformer 디코더 각 레이어의 중간 출력에 보조 손실(Auxiliary Loss) 적용 여부
        self.use_actioness = cfg.MODEL.TASTVG.USE_ACTION # 프레임별 Actionness 예측 헤드 및 손실 사용 여부
        self.query_dim = cfg.MODEL.TASTVG.QUERY_DIM     # 쿼리 좌표 차원 (4: cx, cy, w, h)

        # 1. 2D Visual Backbone (Appearance / 정적 외형 특징 추출: ResNet 등)
        self.vis_encoder = build_vis_encoder(cfg)
        vis_fea_dim = self.vis_encoder.num_channels     # 2D 백본 출력 채널 수 (예: 2048)

        hidden_dim = cfg.MODEL.TASTVG.HIDDEN            # 모델 은닉 차원 (예: 256)

        # 2. Text Encoder (텍스트 특징 추출: RoBERTa 등)
        self.text_encoder = build_text_encoder(cfg)

        # 3. Text-guided Temporal Sampling 모듈
        # 텍스트 질의와 비디오의 상호작용을 통해 타깃 행동이 발생하는 시간 프레임을 선별
        self.s_temporal_clas = build_TemporalSampling(hidden_dim)  # Appearance(2D 정적 시각 특징) 기반 시간 샘플링
        self.t_temporal_clas = build_TemporalSampling(hidden_dim)  # Motion(3D 비디오 스윈 특징) 기반 시간 샘플링

        # 4. Attribute-aware Spatial Activation 모듈
        # 선택된 타깃 프레임에서 외형(Appearance) 및 동작(Motion) 속성에 주목하여 공간 활성화 맵(Attention Map) 생성
        self.s_spatial_clas = build_SpatialActivation(hidden_dim, cfg.DATASET.APP_NUM)  # 외형/형용사 속성 분류 및 공간 활성화 맵
        self.t_spatial_clas = build_SpatialActivation(hidden_dim, cfg.DATASET.MOT_NUM)  # 동작/동사 속성 분류 및 공간 활성화 맵

        # 5. Multimodal Cross-Modal Encoder & Decoder
        self.ground_encoder = build_encoder(cfg)  # 시각(2D/3D)과 텍스트 간의 시공간 상호작용을 모델링하는 Cross-Modal 인코더
        self.ground_decoder = build_decoder(cfg)  # 타깃 인식 쿼리를 입력받아 시공간 위치를 예측하는 디코더 (PosDecoder + TimeDecoder)
        
        # 6. 특징 차원 정렬을 위한 1x1 합성곱 투영 계층
        self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)  # 2D 시각 특징 채널(vis_fea_dim) -> hidden_dim(256)

        # 7. 예측 헤드 (MLP Heads)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)      # 비디오 내 타깃 시작/종료 시점(start/end timestamps) 예측 헤드
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)                   # 각 프레임별 바운딩 박스(cx, cy, w, h) 회귀 헤드

        # 8. 3D Video Backbone (Motion / 동적 비디오 특징 추출: Video Swin Transformer)
        # Kinetics-400으로 사전 학습된 3D 비디오 특징 추출기 (파라미터 고정/freeze)
        self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)         # 3D Video Swin 특징 채널(768) -> hidden_dim(256)
        for param in self.vid.parameters():
            param.requires_grad = False

        # 프레임별 타깃 행동 존재 여부(Actionness) 예측 MLP 헤드
        self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        self.ground_decoder.time_embed2 = self.action_embed

        # 디코더의 점진적 바운딩 박스 갱신(Iterative Anchor Update)을 위해 bbox_embed 연결
        self.ground_decoder.decoder.bbox_embed = self.bbox_embed

        # 9. 데이터셋 주석 라벨 로드 및 프레임 선택 임계값(theta) 설정
        self.verb_label = load_json(cfg.DATA_DIR + "/annos/train.json")
        self.verb_label2 = load_json(cfg.DATA_DIR + "/annos/test.json")
        self.theta = 0.45 if cfg.DATASET.NAME == 'VidSTG' else 0.7            # 시간 샘플링 임계값 theta (VidSTG: 0.45, HC-STVG: 0.7)

    def forward(self, videos, texts, targets, iteration_rate=-1):
        """
        순전파 (Forward Pass)
        
        Args:
            videos (NestedTensor): 비디오 텐서 [B*T, C, H, W], 패딩 마스크, 비디오 길이 정보 포함
            texts (list[str]): 비디오 내 타깃 객체/행동을 기술하는 자연어 질의 텍스트 리스트
            targets (list[dict]): 정답(GT) 어노테이션 리스트 (정답 바운딩 박스, actioness, vid, item_id 등)
            iteration_rate (float): 반복 쿼리 정제(Iterative Refinement) 수행 여부 제어 (< 0 일 때 추론 반복 정제 수행)

        Returns:
            dict: 모델 최종 예측 결과
                - 'pred_boxes': 최종 예측 바운딩 박스 좌표 [B*T, 4] (cx, cy, w, h normalized)
                - 'pred_sted': 최종 예측 시작/종료 시간 확률/구간 [B, 2]
                - 'pred_actioness': 각 프레임별 타깃 액션 발생 확률 로짓 [B*T, 1]
                - 'logits_f_m', 'logits_f_a': 모션 및 외형 기반 시간적 프레임 관련도 로짓
                - 'logits_r_m', 'logits_r_a': 동작(동사) 및 외형(형용사) 속성 분류 점수
                - 'aux_outputs': 디코더 각 레이어의 중간 출력 리스트 (보조 손실 계산용)
                - 'att_sequences': 프레임별 타깃 관련도 점수 시퀀스
                - 'pr': 샘플링된 프레임과 GT 액션 프레임 간의 정밀도/재현율 (Precision, Recall)
        """
        # =========================================================================
        # [Step 1] Visual Feature Extraction (시각 특징 추출)
        # 1-1. 2D Appearance 특징 추출 (ResNet 백본)
        # =========================================================================
        vis_outputs, vis_pos_embed = self.vis_encoder(videos)
        vis_res_features, vis_mask, vis_durations = vis_outputs.decompose()  
        vis_features = self.input_proj(vis_res_features)  # 채널 투영: [B*T, vis_fea_dim, H, W] -> [B*T, hidden_dim, H, W]
        vis_outputs = NestedTensor(vis_features, vis_mask, vis_durations)

        # 1-2. 3D Motion 특징 추출 (Video Swin Transformer, 가중치 고정)
        with torch.no_grad():
            vid_features_all = self.vid(videos.tensors, len(videos.tensors))
        vid_features = self.input_proj2(vid_features_all['3'])  # 채널 투영: [B*T, 768, H, W] -> [B*T, hidden_dim, H, W]

        # =========================================================================
        # [Step 2] Textual Feature Extraction & Subject Augmentation (텍스트 특징 추출 및 주어 보강)
        # =========================================================================
        vid_key = targets[0]['vid']
        # HC-STVG 주석 파일(test.json / train.json)은 확장자를 포함한 전체 파일명을 키로 사용하므로 확장자를 통일
        if '.' in vid_key:
            vid_key = os.path.splitext(vid_key)[0] + '.mp4'
        info_key = str(targets[0]['item_id']) if self.cfg.DATASET.NAME == 'VidSTG' else vid_key
        
        # 주석에서 타깃의 Subject(주어 명사구)를 조회하여 텍스트 쿼리 앞에 결합 ("subject + original_text")
        if info_key not in self.verb_label2:
            # 주석 파일에 해당 비디오 ID가 없는 경우 예외 처리 (알 수 없는 타깃 대체)
            logger = self.cfg.get_logger("Video Grounding") if hasattr(self.cfg, "get_logger") else None
            if logger is not None:
                logger.warning(
                    f"HC-STVG subject annotation not found for vid={info_key!r}; "
                    f"using placeholder subject '[unknown]'. "
                    f"Available keys: {list(self.verb_label2.keys())[:10]}..."
                )
            subject = "[unknown]"
        else:
            subject = self.verb_label2[info_key]['sub']
        texts = [subject + " " + texts[0]]  # 주어가 보강된 텍스트 질의 생성
        text_outputs, text_cls = self.text_encoder(texts, vis_features.device)

        # =========================================================================
        # [Step 3] Multimodal Feature Fusion (Cross-Modal Encoder 상호작용)
        # 2D 외형(Appearance), 3D 모션(Motion), 텍스트(Text) 특징을 결합하여 교차 어텐션 수행
        # =========================================================================
        encoded_info = self.ground_encoder(videos=vis_outputs, vis_pos=vis_pos_embed, texts=text_outputs, vid=vid_features)

        # 인코더 출력 시퀀스에서 각 모달리티 특징을 분리 (spatial tokens 수 l = H * W)
        l = vid_features.size(-1) * vid_features.size(-2)
        # 3D 비디오 모션 특징 추출 (시퀀스 끝부분 [-l:])
        f_vid_features = encoded_info['encoded_feature'][-l:].permute(1, 2, 0).reshape(vid_features.size()).detach()
        # 2D 시각 외형 특징 추출 (시퀀스 앞부분 [:l])
        f_vis_features = encoded_info['encoded_feature'][:l].permute(1, 2, 0).reshape(vid_features.size()).detach()
        # 텍스트 토큰 임베딩의 평균을 통한 문장 표현 추출 (중간 부분 [l:-l])
        f_text_cls = encoded_info['encoded_feature'][l:-l].mean(1).unsqueeze(0).detach()
        
        # =========================================================================
        # [Step 4] Text-guided Temporal Sampling (텍스트 유도 시간적 프레임 샘플링)
        # 타깃 행동과 관련된 프레임의 존재 확률을 평가하고 후보 프레임을 선별
        # =========================================================================
        logits_f_m = self.t_temporal_clas(f_vid_features, f_text_cls)  # 모션 기반 프레임별 타깃 관련도 점수
        logits_f_a = self.s_temporal_clas(f_vis_features, f_text_cls)  # 외형 기반 프레임별 타깃 관련도 점수
        
        # 모션과 외형의 시간적 점수를 융합하여 프레임별 어텐션 시퀀스 계산
        att_sequences = (logits_f_m.sigmoid() + logits_f_a.sigmoid()) / 2
        # 임계값(theta)을 초과하는 프레임의 인덱스 선택
        choose_index = torch.nonzero(att_sequences > self.theta).squeeze().tolist()
        choose_index = [choose_index] if isinstance(choose_index, int) else choose_index
        # 임계값을 넘는 프레임이 없을 경우, 0보다 큰 프레임을 fallback으로 선택
        choose_index = choose_index or torch.nonzero(att_sequences > 0).squeeze().tolist()   

        # =========================================================================
        # [Step 5] Attribute-aware Spatial Activation (속성 인식 공간적 활성화)
        # 선택된 타깃 후보 프레임에 대해서만 공간 활성화 맵과 속성 분류 점수 계산 (노이즈 억제 및 연산 효율화)
        # =========================================================================
        logits_r_m, att_map_t = self.t_spatial_clas(f_vid_features[choose_index], f_text_cls[:,:1])  # 동작(동사) 속성 점수 및 시간 어텐션 맵
        logits_r_a, att_map_s = self.s_spatial_clas(f_vis_features[choose_index], f_text_cls[:,:1])  # 외형(형용사) 속성 점수 및 공간 어텐션 맵

        # =========================================================================
        # [Step 6] Generating Target-Aware Object Queries (타깃 인식 객체 쿼리 생성)
        # 논문의 핵심: zero query 대신 타깃 특화 시각 특징과 어텐션 맵을 가중합하여 초기 쿼리 생성
        # =========================================================================
        # itq (Initial Temporal Query): 모션 특징과 시간 어텐션 맵의 가중 평균
        init_tempral_query = (encoded_info['encoded_feature'][-l:].permute(1, 0, 2)[choose_index] * att_map_t.unsqueeze(2)).mean((0, 1))
        # isq (Initial Spatial Query): 외형 특징과 공간 어텐션 맵의 가중 평균
        init_spatial_query = (encoded_info['encoded_feature'][:l].permute(1, 0, 2)[choose_index] * att_map_s.unsqueeze(2)).mean((0, 1))

        # =========================================================================
        # [Step 7] Query-based Decoding (1차 시공간 디코딩)
        # 생성된 타깃 인식 쿼리(isq, itq)를 PosDecoder 및 TimeDecoder에 주입하여 위치 및 구간 디코딩
        # =========================================================================
        outputs_pos, outputs_time = self.ground_decoder(
            encoded_info=encoded_info, 
            vis_pos=vis_pos_embed, 
            isq=init_spatial_query, 
            itq=init_tempral_query
        )

        # =========================================================================
        # [Step 8] Iterative Query Refinement (반복적 쿼리 정제 - 추론 모드: iteration_rate < 0)
        # 1차 디코딩된 actionness 예측값을 바탕으로 프레임을 재선별하고 쿼리를 갱신하여 2차 디코딩 수행
        # =========================================================================
        if iteration_rate < 0:
            # 1차 디코딩 결과의 actionness 확률이 0.5를 초과하는 정제된 프레임 인덱스 추출
            choose_index = torch.nonzero((self.action_embed(outputs_time)[-1].squeeze().sigmoid() > 0.5).int()).squeeze().tolist()
            choose_index = [choose_index] if isinstance(choose_index, int) else choose_index
            choose_index = choose_index or torch.nonzero(att_sequences > 0).squeeze().tolist()  

            # 재선별된 프레임으로 공간 활성화 맵 다시 계산
            logits_r_a, att_map_s = self.s_spatial_clas(f_vis_features[choose_index], f_text_cls[:,:1])
            logits_r_m, att_map_t = self.t_spatial_clas(f_vid_features[choose_index], f_text_cls[:,:1])
           
            # 정제된 공간 및 시간 타깃 쿼리 재생성
            init_tempral_query = (encoded_info['encoded_feature'][-l:].permute(1, 0, 2)[choose_index] * att_map_t.unsqueeze(2)).mean((0, 1))
            init_spatial_query = (encoded_info['encoded_feature'][:l].permute(1, 0, 2)[choose_index] * att_map_s.unsqueeze(2)).mean((0, 1))
            
            # 갱신된 타깃 인식 쿼리로 2차 디코딩 실행 (최종 위치 예측 향상)
            outputs_pos, outputs_time = self.ground_decoder(
                encoded_info=encoded_info, 
                vis_pos=vis_pos_embed, 
                isq=init_spatial_query, 
                itq=init_tempral_query
            )

        # =========================================================================
        # [Step 9] Output Packaging (최종 출력값 구성)
        # =========================================================================
        out = {}
        
        # 9-1. 예측된 공간 바운딩 박스 (cx, cy, w, h)
        outputs_coord = outputs_pos.flatten(1, 2) 
        out.update({"pred_boxes": outputs_coord[-1]})  # 디코더 마지막 레이어의 예측값

        # 9-2. 시간적 관련도 점수 (Temporal Relevance Scores)
        out.update({"logits_f_m": logits_f_m})
        out.update({"logits_f_a": logits_f_a})

        # 9-3. 속성 분류 점수 (Attribute Classification Scores)
        out.update({"logits_r_a": logits_r_a})
        out.update({"logits_r_m": logits_r_m})

        # 9-4. 시작/종료 시점 및 프레임별 Actionness 예측
        time_hiden_state = outputs_time  
        outputs_time = self.temp_embed(time_hiden_state)      # 시간 시작/종료 예측 [B, 2]
        outputs_actioness = self.action_embed(time_hiden_state) # 프레임별 타깃 행동 존재 확률 [B*T, 1]
        out.update({"pred_sted": outputs_time[-1]})
        out.update({"pred_actioness": outputs_actioness[-1]})
        
        # 9-5. 보조 손실(Auxiliary Loss)용 중간 레이어 예측값 수집
        if self.use_aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_sted": a,
                    "pred_boxes": b,
                    "pred_actioness": c
                }
                for a, b, c in zip(outputs_time[:-1], outputs_coord[:-1], outputs_actioness[:-1])
            ]

        # 9-6. 동사 및 형용사 정답(GT) 라벨 인덱스 조회 (손실 계산용)
        if not self.training and info_key not in self.verb_label2:
            logger = self.cfg.get_logger("Video Grounding") if hasattr(self.cfg, "get_logger") else None
            if logger is not None:
                logger.warning(
                    f"HC-STVG verb/attr annotation missing for vid={info_key!r}; using empty placeholders."
                )
            verb_labels = []
            attr_labels = []
        else:
            verb_labels = self.verb_label[info_key]['verb_index_list'] if self.training else self.verb_label2[info_key]['verb_index_list']
            attr_labels = self.verb_label[info_key]['adj_index_list'] if self.training else self.verb_label2[info_key]['adj_index_list']
        out['verb_labels'] = verb_labels
        out['attr_labels'] = attr_labels
        
        # 9-7. 프레임별 어텐션 시퀀스 및 프레임 샘플링 평가 지표 (Precision / Recall)
        out['att_sequences'] = att_sequences.unsqueeze(0)
        gt_index = torch.nonzero(targets[0]['actioness']).squeeze().tolist()
        out["pr"] = precision_recall(choose_index, gt_index)
        
        return out