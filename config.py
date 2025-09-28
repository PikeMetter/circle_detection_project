import os

class Config:
    # 数据路径
    TRAIN_DATA_DIR = "datasets/train"
    VAL_DATA_DIR = "datasets/val"
    TEST_DATA_DIR = "datasets/test"
    
    # 模型参数
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    IMAGE_SIZE = (512, 512)
    
    # 训练参数
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = "cuda"
    
    # 损失函数权重
    HEATMAP_WEIGHT = 1.0
    COORD_WEIGHT = 10.0
    
    # 热力图参数
    HEATMAP_SIGMA = 10
    
    # 保存路径
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # 创建必要目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
