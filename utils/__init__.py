from .metrics import calculate_metrics
from .visualization import visualize_predictions
from .logger import setup_logger

# 添加别名以兼容旧代码
compute_metrics = calculate_metrics

__all__ = ['calculate_metrics', 'visualize_predictions', 'setup_logger', 'compute_metrics']
