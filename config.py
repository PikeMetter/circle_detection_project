import os

class Config:
    # 数据路径
    TRAIN_DATA_DIR = "datasets_processed/train"
    VAL_DATA_DIR = "datasets_processed/val"
    TEST_DATA_DIR = "datasets_processed/test"
    
    # 模型选择
    MODEL_TYPE = "yolo_v1"  # 可选值: unet_circle, unet_segmentation, yolo_v1
    
    # 模型通用参数
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    IMAGE_SIZE = (448, 448)  # YOLOv1标准输入尺寸为448x448
    
    # 训练参数
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001  # 学习率,默认值为0.001
    NUM_EPOCHS = 100
    DEVICE = "cuda"
    
    # UNet相关参数
    HEATMAP_WEIGHT = 2.0  # 热力图损失权重
    COORD_WEIGHT = 5.0    # 坐标损失权重
    HEATMAP_SIGMA = 10    # 热力图参数
    
    # YOLOv1相关参数
    YOLO_S = 7        # 网格大小
    YOLO_B = 2        # 每个网格预测的边界框数量
    YOLO_C = 1        # 类别数量
    YOLO_CONF_THRESHOLD = 0.5  # 置信度阈值
    YOLO_NMS_THRESHOLD = 0.4   # 非极大值抑制阈值
    YOLO_LAMBDA_COORD = 5.0    # 坐标损失权重
    YOLO_LAMBDA_NOOBJ = 0.5    # 无物体损失权重
    YOLO_CONF_THRESHOLD = 0.5  # 置信度阈值
    YOLO_NMS_THRESHOLD = 0.5   # 非极大值抑制阈值
    
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
