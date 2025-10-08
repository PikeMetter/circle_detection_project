import os

class Config:
    # 数据路径
    TRAIN_DATA_DIR = "datasets_processed/train"
    VAL_DATA_DIR = "datasets_processed/val"
    TEST_DATA_DIR = "datasets_processed/test"
    
    # 模型参数
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    IMAGE_SIZE = (512, 512)
    
    # 训练参数
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001 #学习率,默认值为0.001
    NUM_EPOCHS = 100
    DEVICE = "cuda"
    
    # 损失函数权重
    HEATMAP_WEIGHT = 2.0 #1.0,现在增加了热力图损失权重
    COORD_WEIGHT = 5.0 #10.0,现在增加了坐标损失权重
    
    # 热力图参数
    HEATMAP_SIGMA = 10
    
    # 保存路径
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"

    # 训练优化参数
    USE_MIXED_PRECISION = False  # 或者 True，根据你的需求
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # 创建必要目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
