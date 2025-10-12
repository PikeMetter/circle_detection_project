import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """所有模型的基类，定义统一接口"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def detect(self, outputs, conf_threshold=0.5, nms_threshold=0.5):
        """
        从模型输出中提取检测结果
        Args:
            outputs: 模型输出
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        Returns:
            detections: 检测结果
        """
        raise NotImplementedError("子类必须实现detect方法")
    
    def get_loss_function(self):
        """
        获取模型对应的损失函数
        Returns:
            loss_function: 损失函数
        """
        raise NotImplementedError("子类必须实现get_loss_function方法")
    
    def get_config(self):
        """
        获取模型配置
        Returns:
            config: 模型配置字典
        """
        raise NotImplementedError("子类必须实现get_config方法")