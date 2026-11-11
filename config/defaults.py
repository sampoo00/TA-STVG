from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition (기본 설정 및 전역 경로)
# -----------------------------------------------------------------------------
_C = CN()
_C.FROM_SCRATCH = True           # 사전학습 가중치 없이 모델을 처음부터(Scratch) 학습할지 여부
_C.DATA_TRUNK = None             # 데이터 트렁크(서브셋) 경로 설정 시 사용

_C.OUTPUT_DIR = ''               # 모델 가중치 체크포인트, 평가 결과 파일이 저장될 출력 디렉터리 경로
_C.DATA_DIR = ''                 # 데이터셋 루트 디렉터리 경로 (예: data/hc-stvg, data/vidstg)
_C.GLOVE_DIR = ''                # GloVe 사전학습 임베딩 경로 (LSTM 텍스트 모델 사용 시)
_C.TENSORBOARD_DIR = ''          # Tensorboard 시각화 로그 파일 저장 디렉터리 경로


# -----------------------------------------------------------------------------
# INPUT (입력 비디오 및 텍스트 데이터 전처리/증강 설정)
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.MAX_QUERY_LEN = 26      # 자연어 텍스트 질의(Query)의 최대 토큰 길이
_C.INPUT.MAX_VIDEO_LEN = 200     # 비디오 최대 프레임 수 (패딩 및 위치 인코딩 상한)

# VidSTG 데이터셋용: 비디오당 고정 샘플링할 프레임 수
_C.INPUT.TRAIN_SAMPLE_NUM = 64
# HC-STVG 데이터셋용: 비디오 프레임 샘플링 레이트 FPS (20초 기준 약 64 프레임 추출)
_C.INPUT.SAMPLE_FPS = 3.2

# 입력 비디오 프레임 해상도 (224 또는 420 등)
_C.INPUT.RESOLUTION = 224
# 이미지 정규화(Normalization)용 RGB 채널별 평균값 (ImageNet 표준)
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# 이미지 정규화용 RGB 채널별 표준편차 (ImageNet 표준)
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# 다중 스케일(Multiscale) 리사이즈 데이터 증강 적용 여부
_C.INPUT.AUG_SCALE = True
# 평행 이동(Translation) 데이터 증강 적용 여부
_C.INPUT.AUG_TRANSLATE = False

# 학습 시 좌우 반전(Horizontal Flip) 증강 적용 확률
_C.INPUT.FLIP_PROB_TRAIN = 0.5
# 시간축 랜덤 크롭(Temporal Crop) 증강 적용 확률
_C.INPUT.TEMP_CROP_PROB = 0.5


# -----------------------------------------------------------------------------
# Model Config (전역 모델 구조 및 학습 방식 설정)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"         # 연산 장치 ("cuda" 또는 "cpu")
_C.MODEL.WEIGHT = ""             # 사전 학습된 백본 또는 전체 모델 체크포인트 경로 (.pth)
_C.MODEL.EMA = True              # 모델 파라미터 지수 이동평균(EMA) 적용 여부 (평가 시 성능 안정화)
_C.MODEL.EMA_DECAY = 0.9998      # EMA 파라미터 감쇠율(decay rate)
_C.MODEL.QUERY_NUM = 1           # 프레임당 타깃 객체 쿼리 수 (단일 객체 추적: 1)
_C.MODEL.DOWN_RATIO = 4          # 시각 특징 맵의 다운샘플링 비율


# -----------------------------------------------------------------------------
# Vision Encoder options (2D 정적 시각 백본 설정: Appearance)
# -----------------------------------------------------------------------------
_C.MODEL.VISION_BACKBONE = CN()
_C.MODEL.VISION_BACKBONE.NAME = 'resnet101'  # 2D 백본 모델 이름 ('resnet50', 'resnet101', 'resnet50-gn', 'resnet101-gn')
_C.MODEL.VISION_BACKBONE.POS_ENC = 'sine'   # 2D 공간 위치 인코딩 방식 ('sine', 'sineHW', 'learned')
_C.MODEL.VISION_BACKBONE.DILATION = False   # True 시 마지막 Conv 블록(DC5)에서 stride 대신 dilation 적용 (특징 맵 해상도 2배 유지)
_C.MODEL.VISION_BACKBONE.FREEZE = False     # True 시 2D 백본 가중치를 고정(freeze)하여 학습에서 제외


# -----------------------------------------------------------------------------
# Language Encoder Config (텍스트 인코더 설정)
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_MODEL = CN()
_C.MODEL.TEXT_MODEL.NAME = 'roberta-base'   # 텍스트 백본 모델 ('roberta-base', 'bert-base', 'roberta-large')
_C.MODEL.TEXT_MODEL.FREEZE = False         # True 시 텍스트 인코더 파라미터 고정

# LSTM 텍스트 인코더 사용 옵션 (USE_LSTM=True 시 RoBERTa 대신 LSTM 사용)
_C.MODEL.USE_LSTM = False
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.NAME = 'lstm'                # LSTM 인코더 모델 이름
_C.MODEL.LSTM.HIDDEN_SIZE = 512            # LSTM 은닉 상태(Hidden state) 차원
_C.MODEL.LSTM.BIDIRECTIONAL = True         # 양방향(Bi-directional) LSTM 사용 여부
_C.MODEL.LSTM.DROPOUT = 0                  # LSTM 드롭아웃 확률
_C.MODEL.LSTM_NUM_LAYERS = 2               # LSTM 레이어 수


# -----------------------------------------------------------------------------
# TASTVG Pipeline Config (TA-STVG 트랜스포머 인코더/디코더 설정)
# -----------------------------------------------------------------------------
_C.MODEL.TASTVG = CN()
_C.MODEL.TASTVG.HIDDEN = 256               # Transformer 인코더/디코더의 은닉 차원 (d_model)
_C.MODEL.TASTVG.QUERY_DIM = 4              # 바운딩 박스 쿼리/앵커 좌표 차원 (cx, cy, w, h: 4)
_C.MODEL.TASTVG.ENC_LAYERS = 6             # CrossModalEncoder의 인코더 레이어 수
_C.MODEL.TASTVG.DEC_LAYERS = 6             # QueryDecoder의 디코더 레이어 수 (PosDecoder / TimeDecoder)
_C.MODEL.TASTVG.FFN_DIM = 2048             # Transformer 피드포워드 신경망(FFN) 중간 은닉 차원
_C.MODEL.TASTVG.DROPOUT = 0.1              # Transformer 내 드롭아웃 확률
_C.MODEL.TASTVG.HEADS = 8                  # Multi-Head Attention의 어텐션 헤드 수
_C.MODEL.TASTVG.USE_LEARN_TIME_EMBED = False # 시간 위치 임베딩 방식 (False: Sinusoidal, True: 학습 가능한 Embedding)
_C.MODEL.TASTVG.USE_ACTION = True          # 프레임별 타깃 행동 존재 확률(Actionness) 예측 헤드 활성화 여부
_C.MODEL.TASTVG.FROM_SCRATCH = True        # Transformer 가중치를 처음부터 초기화할지 여부

# 2D-Map temporal prediction 관련 설정 (시간 시작/종료 시점 맵 예측)
_C.MODEL.TASTVG.TEMP_PRED_LAYERS = 6       # 시간 구간 예측을 위한 레이어 수
_C.MODEL.TASTVG.CONV_LAYERS = 4            # 시간 맵 합성곱(Conv) 레이어 수
_C.MODEL.TASTVG.TEMP_HEAD = 'attn'         # 시간 예측 헤드 유형 ('attn' 또는 'conv')
_C.MODEL.TASTVG.KERNAL_SIZE = 9            # 시간 Conv 연산 커널 크기
_C.MODEL.TASTVG.MAX_MAP_SIZE = 128         # 2D 시간 상관관계 맵 최대 크기
_C.MODEL.TASTVG.POOLING_COUNTS = [15,8,8,8] # 시간 피라미드 풀링 스트라이드/카운트 리스트


# -----------------------------------------------------------------------------
# DATASET related params (데이터셋별 어휘 수 및 클립 설정)
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'VidSTG'                 # 사용할 데이터셋 이름 ('VidSTG' 또는 'HC-STVG')
_C.DATASET.NUM_CLIP_FRAMES = 32            # 클립 단위 샘플링 시 클립당 프레임 수
_C.DATASET.MIN_GT_FRAME = 4                # 샘플링된 클립 내 포함되어야 하는 최소 정답(GT) 프레임 수
_C.DATASET.APP_NUM = 20                    # 외형(Appearance) 속성/형용사 카테고리 수 (VidSTG: 20, HC-STVG: 15)
_C.DATASET.MOT_NUM = 34                    # 동작(Motion) 속성/동사 카테고리 수 (VidSTG: 34, HC-STVG: 60)


# -----------------------------------------------------------------------------
# DataLoader (데이터 로더 및 배치 구성 설정)
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4              # 데이터 로딩 멀티프로세스 CPU 워커 수
_C.DATALOADER.SIZE_DIVISIBILITY = 0        # 이미지 크기가 나누어떨어져야 하는 정수 단위 (0이면 비활성화)
_C.DATALOADER.ASPECT_RATIO_GROUPING = False # 유사한 종횡비(가로세로 비율)를 가진 영상을 그룹화하여 패딩 최소화 여부


# ---------------------------------------------------------------------------- #
# Solver (학습 최적화, 손실 가중치, 학습률 스케줄러 설정)
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 30                   # 총 학습 에폭(Epoch) 수
_C.SOLVER.BATCH_SIZE = 1                   # GPU당 비디오 배치 크기 (비디오 모델 메모리 특성상 1 권장)
_C.SOLVER.SHUFFLE = True                   # 에폭마다 데이터 셔플 여부
_C.SOLVER.OPTIMIZER = 'adamw'              # 최적화 알고리즘 ('adamw')
_C.SOLVER.MAX_GRAD_NORM = 0.1              # 그래디언트 클리핑(Gradient Clipping) 최대 Norm (기울기 폭발 방지)

# 모듈별 세부 학습률 (Learning Rates)
_C.SOLVER.BASE_LR = 2e-5                   # 트랜스포머 인코더/디코더 기본 학습률
_C.SOLVER.VIS_BACKBONE_LR = 1e-5           # 2D 시각 백본(ResNet) 학습률 (보통 백본 파인튜닝을 위해 낮게 설정)
_C.SOLVER.TEXT_LR = 2e-5                   # 텍스트 인코더(RoBERTa) 학습률
_C.SOLVER.TEMP_LR = 1e-4                   # 시간 구간 예측 모듈(Temporal Module) 학습률
_C.SOLVER.VERB_LR = 3e-3                   # 속성 분류기(동사/형용사 Spatial Activation) 학습률

# 손실 함수별 가중치 계수 (Loss Weight Hyperparameters)
_C.SOLVER.BBOX_COEF = 5                    # 바운딩 박스 L1 회귀 손실 가중치
_C.SOLVER.GIOU_COEF = 2                    # 바운딩 박스 GIoU(Generalized IoU) 손실 가중치
_C.SOLVER.TEMP_COEF = 2                    # 시작/종료 시점(start/end time) 예측 손실 가중치
_C.SOLVER.ATTN_COEF = 1                    # 어텐션 가이드 손실 가중치
_C.SOLVER.ACTIONESS_COEF = 2               # 프레임별 Actionness 예측 BCE 손실 가중치
_C.SOLVER.CONF_COEF = 1                    # logits_f_m (모션 기반 시간 샘플링 BCE 손실) 가중치
_C.SOLVER.CONF2_COEF = 1                   # logits_f_a (외형 기반 시간 샘플링 BCE 손실) 가중치
_C.SOLVER.CONF3_COEF = 1                   # logits_r_a (외형 속성 분류 Multi-label BCE 손실) 가중치
_C.SOLVER.CONF4_COEF = 1                   # logits_r_m (동작 속성 분류 Multi-label BCE 손실) 가중치

# 학습률 감쇠 및 모멘텀 설정
_C.SOLVER.MOMENTUM = 0.9                   # 옵티마이저 모멘텀 계수
_C.SOLVER.WEIGHT_DECAY = 0.0001            # 가중치 감쇠(L2 Regularization) 계수
_C.SOLVER.GAMMA = 0.1                      # 학습률 감소 비율 (새로운 lr = lr * gamma)
_C.SOLVER.POWER = 0.9                      # Poly LR 스케줄러 지수
_C.SOLVER.STEPS = (30000,)                 # MultiStepLR 스케줄러의 이터레이션 기준 감소 스텝

# 학습 초기 웜업(Warmup) 설정
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3          # 웜업 시작 시 초기 학습률 비율 (base_lr * warmup_factor)
_C.SOLVER.WARMUP_ITERS = 500               # 웜업 수행 이터레이션 수
_C.SOLVER.WARMUP_PROP = 0.01               # 전체 학습 대비 웜업 비율
_C.SOLVER.WARMUP_METHOD = "linear"         # 웜업 방식 ("linear" 또는 "constant")

# 학습률 스케줄러 상세 설정
_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "linear_with_warmup" # 스케줄러 유형 ("linear_with_warmup", "multistep_with_warmup")
_C.SOLVER.SCHEDULE.DROP_STEP = [8,12]      # 학습률 감소 에폭 단계 (예: 8에폭, 12에폭에서 감쇠)

# WarmupReduceLROnPlateau 스케줄러 전용 설정 (성능 정체 시 학습률 감소)
_C.SOLVER.SCHEDULE.PATIENCE = 2            # 검증 성능 향상이 없을 때 대기할 에폭 수
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4        # 개선으로 인정할 최소 변화량
_C.SOLVER.SCHEDULE.COOLDOWN = 1            # 감쇠 후 일시 정지 에폭 수
_C.SOLVER.SCHEDULE.FACTOR = 0.5            # 학습률 감쇠 비율
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7      # 최대 감쇠 횟수

# 검증 및 모델 체크포인트 저장 주기
_C.SOLVER.PRE_VAL = False                  # 학습 시작 전 0번째 이터레이션에서 사전 검증 수행 여부
_C.SOLVER.TO_VAL = True                    # 학습 도중 주기적으로 검증(Validation) 수행 여부
_C.SOLVER.VAL_PERIOD = 3000                # 검증 수행 주기 (이터레이션 단위)
_C.SOLVER.CHECKPOINT_PERIOD = 5000         # 모델 체크포인트 저장 주기 (이터레이션 단위)

# 보조 손실 및 정규화 계수
_C.SOLVER.USE_ATTN = False                 # TubeDETR과 비교를 위한 가이드 어텐션 손실 사용 여부
_C.SOLVER.SIGMA = 2.0                      # KL 발산 손실에 사용되는 양자화 가우시안 분포의 표준편차
_C.SOLVER.USE_AUX_LOSS = True              # 디코더의 모든 중간 레이어 출력에 보조 손실(Auxiliary Loss) 적용 여부
_C.SOLVER.EOS_COEF = 0.1                   # 배경/음성 샘플(Negative sample)에 대한 분류 손실 가중치 계수
