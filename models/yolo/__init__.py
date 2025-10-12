# YOLO 模块初始化文件
from .yolov1 import YOLOv1
from .yolov3 import YOLOv3
from .yolov5 import YOLOv5
from .yolov8 import YOLOv8
from .losses import YOLOv1Loss

__all__ = ['YOLOv1', 'YOLOv3', 'YOLOv5', 'YOLOv8', 'YOLOv1Loss']