from .unet import UNetForCircleCenter
from .unet.unet import UNetForCircleCenter, UNetSegmentation
from .yolo.yolov1 import YOLOv1
from .yolo.yolov3 import YOLOv3
from .yolo.yolov5 import YOLOv5
from .yolo.yolov8 import YOLOv8
from .unet.losses import CombinedLoss, FocalLoss
from .yolo.losses import YOLOv1Loss
from .model_interface import BaseModel

class ModelFactory:
    """模型工厂类，用于创建不同类型的模型和损失函数"""
    
    # 支持的模型类型
    _models = {
        'unet_circle': UNetForCircleCenter,
        'unet_segmentation': UNetSegmentation,
        'yolo_v1': YOLOv1,
        'yolo_v3': YOLOv3,
        'yolo_v5': YOLOv5,
        'yolo_v8': YOLOv8
    }
    
    # 支持的损失函数类型
    _losses = {
        'combined': CombinedLoss,
        'focal': FocalLoss,
        'yolo_v1': YOLOv1Loss,
        'yolo_v3': YOLOv1Loss,  # YOLOv3可以使用类似的损失函数
        'yolo_v5': YOLOv1Loss,  # YOLOv5可以使用类似的损失函数
        'yolo_v8': YOLOv1Loss   # YOLOv8可以使用类似的损失函数
    }
    
    # 模型类型分组
    _model_groups = {
        'yolo': ['yolo_v1', 'yolo_v3', 'yolo_v5', 'yolo_v8'],
        'unet': ['unet_circle', 'unet_segmentation']
    }
    
    @classmethod
    def create_model(cls, model_type, **kwargs):
        """
        创建指定类型的模型
        Args:
            model_type: 模型类型字符串
            **kwargs: 传递给模型构造函数的参数
        Returns:
            model: 创建的模型实例
        Raises:
            ValueError: 如果指定的模型类型不支持
        """
        if model_type not in cls._models:
            supported_models = ', '.join(cls._models.keys())
            raise ValueError(f"不支持的模型类型: {model_type}。支持的模型类型有: {supported_models}")
        
        model_class = cls._models[model_type]
        
        # 处理配置参数
        if 'config' in kwargs:
            config = kwargs.pop('config')
            # 根据模型类型将配置参数映射到模型构造函数参数
            if model_type == 'yolo_v1':
                kwargs.setdefault('S', config.get('S', 7))
                kwargs.setdefault('B', config.get('B', 2))
                kwargs.setdefault('C', config.get('C', 1))
            elif model_type in ['yolo_v3', 'yolo_v5', 'yolo_v8']:
                kwargs.setdefault('num_classes', config.get('num_classes', 1))
            elif model_type == 'unet_circle':
                # UNet模型可能需要特殊处理
                pass
        
        return model_class(**kwargs)
    
    @classmethod
    def create_loss(cls, loss_type, **kwargs):
        """
        创建指定类型的损失函数
        Args:
            loss_type: 损失函数类型字符串
            **kwargs: 传递给损失函数构造函数的参数
        Returns:
            loss: 创建的损失函数实例
        Raises:
            ValueError: 如果指定的损失函数类型不支持
        """
        if loss_type not in cls._losses:
            supported_losses = ', '.join(cls._losses.keys())
            raise ValueError(f"不支持的损失函数类型: {loss_type}。支持的损失函数类型有: {supported_losses}")
        
        loss_class = cls._losses[loss_type]
        return loss_class(**kwargs)
    
    @classmethod
    def get_supported_models(cls):
        """
        获取所有支持的模型类型
        Returns:
            list: 支持的模型类型列表
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_supported_losses(cls):
        """
        获取所有支持的损失函数类型
        Returns:
            list: 支持的损失函数类型列表
        """
        return list(cls._losses.keys())
    
    @classmethod
    def register_model(cls, model_type, model_class):
        """
        注册新的模型类型
        Args:
            model_type: 模型类型字符串
            model_class: 模型类
        """
        cls._models[model_type] = model_class
    
    @classmethod
    def register_loss(cls, loss_type, loss_class):
        """
        注册新的损失函数类型
        Args:
            loss_type: 损失函数类型字符串
            loss_class: 损失函数类
        """
        cls._losses[loss_type] = loss_class
    
    @classmethod
    def is_yolo_model(cls, model_type):
        """
        判断是否为YOLO系列模型
        Args:
            model_type: 模型类型字符串
        Returns:
            bool: 是否为YOLO系列模型
        """
        return model_type in cls._model_groups['yolo']
    
    @classmethod
    def is_unet_model(cls, model_type):
        """
        判断是否为UNet系列模型
        Args:
            model_type: 模型类型字符串
        Returns:
            bool: 是否为UNet系列模型
        """
        return model_type in cls._model_groups['unet']
    
    @classmethod
    def get_model_group(cls, model_type):
        """
        获取模型所属的组
        Args:
            model_type: 模型类型字符串
        Returns:
            str: 模型组名称，如果未找到则返回None
        """
        for group, models in cls._model_groups.items():
            if model_type in models:
                return group
        return None
    
    @classmethod
    def get_config(cls, model_type):
        """
        获取指定模型类型的配置
        Args:
            model_type: 模型类型字符串
        Returns:
            dict: 模型配置字典
        """
        return ModelConfig.get_config(model_type)

# 扩展配置类来支持不同的模型参数
class ModelConfig:
    """模型配置类，用于存储不同模型的配置参数"""
    
    # YOLOv1 模型配置
    yolo_v1 = {
        'S': 7,  # 网格大小
        'B': 2,  # 每个网格的边界框数量
        'C': 1,  # 类别数量
        'lambda_coord': 5.0,  # 坐标损失权重
        'lambda_noobj': 0.5,  # 无物体损失权重
        'conf_threshold': 0.5,  # 置信度阈值
        'nms_threshold': 0.5,  # 非极大值抑制阈值
        # 可选的锚框配置
        # 'ANCHOR_BOXES': [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)]
    }
    
    # YOLOv3 模型配置
    yolo_v3 = {
        'num_classes': 1,
        'num_boxes': 3,
        'conf_threshold': 0.5,
        'nms_threshold': 0.5
    }
    
    # YOLOv5 模型配置
    yolo_v5 = {
        'num_classes': 1,
        'conf_threshold': 0.5,
        'nms_threshold': 0.5
    }
    
    # YOLOv8 模型配置
    yolo_v8 = {
        'num_classes': 1,
        'conf_threshold': 0.5,
        'nms_threshold': 0.5
    }
    
    # UNet 模型配置
    unet_circle = {
        'heatmap_weight': 1.0,
        'coord_weight': 10.0
    }
    
    @classmethod
    def get_config(cls, model_type):
        """
        获取指定模型类型的配置
        Args:
            model_type: 模型类型字符串
        Returns:
            dict: 模型配置字典
        Raises:
            ValueError: 如果指定的模型类型没有配置
        """
        if hasattr(cls, model_type):
            return getattr(cls, model_type).copy()
        raise ValueError(f"没有找到 {model_type} 的配置")