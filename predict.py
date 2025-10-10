import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from models import ModelFactory
from utils import visualize_predictions, setup_logger
from torch.cuda.amp import autocast

class CirclePredictor:
    def __init__(self, model_path, device='cuda', model_type=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(name="predictor")
        
        # 使用指定的模型类型或从配置中获取
        self.model_type = model_type if model_type is not None else Config.MODEL_TYPE
        self.logger.info(f"Using {self.model_type} model")
        
        # 使用ModelFactory创建模型
        self.model = self._create_model()
        self.load_model(model_path)
        
        # 数据变换 - 专门为预测创建
        self.transform = self._get_predict_transforms(Config.IMAGE_SIZE)
        
        self.logger.info(f"Predictor initialized on {self.device}")
    
    def _create_model(self):
        """使用ModelFactory创建模型"""
        if self.model_type == 'yolo_v1':
            model = ModelFactory.create_model(
                'yolo_v1',
                in_channels=Config.INPUT_CHANNELS,
                num_classes=Config.YOLO_C,
                S=Config.YOLO_S,
                B=Config.YOLO_B
            )
        elif self.model_type == 'unet_segmentation':
            model = ModelFactory.create_model(
                'unet_segmentation',
                in_channels=Config.INPUT_CHANNELS,
                out_channels=Config.OUTPUT_CHANNELS
            )
        else:
            model = ModelFactory.create_model(
                'unet_circle',
                in_channels=Config.INPUT_CHANNELS,
                out_channels=Config.OUTPUT_CHANNELS
            )
        return model
    
    def _get_predict_transforms(self, image_size):
        """获取用于预测的数据增强变换"""
        transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return transforms
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                self.logger.info(f"Model validation metrics: {metrics}")
            elif 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
                self.logger.info(f"Model validation metrics: {metrics}")
        else:
            self.model.load_state_dict(checkpoint)
            self.logger.info("Loaded model state dict")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image_path):
        """预处理单张图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # numpy array
        
        original_size = image.shape[:2]  # (H, W)
        
        # 应用变换
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # 添加batch维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_size
    
    def predict_single_image(self, image_path, return_details=False):
        """预测单张图像中的圆形"""
        # 预处理
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # 混合精度推理
            with autocast(enabled=Config.USE_MIXED_PRECISION):
                if self.model_type == 'yolo_v1':
                    outputs = self.model(image_tensor)
                    # 使用模型的detect方法提取检测结果
                    detections = self.model.detect(
                        outputs,
                        conf_threshold=Config.YOLO_CONF_THRESHOLD,
                        nms_threshold=Config.YOLO_NMS_THRESHOLD
                    )
                    # 转换检测结果到原始图像尺寸
                    original_h, original_w = original_size
                    results = []
                    for detection in detections[0]:  # 取第一个样本的检测结果
                        if len(detection) > 0:
                            x1, y1, x2, y2, conf, class_idx = detection
                            # 转换归一化坐标到原始图像尺寸
                            x1 = int(x1 * original_w)
                            y1 = int(y1 * original_h)
                            x2 = int(x2 * original_w)
                            y2 = int(y2 * original_h)
                            # 计算中心点
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            # 计算半径
                            radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                            results.append({
                                'center_x': center_x,
                                'center_y': center_y,
                                'radius': radius,
                                'confidence': conf.item(),
                                'bbox': (x1, y1, x2, y2),
                                'original_size': original_size
                            })
                    # 如果没有检测结果，返回默认值
                    if not results:
                        results = [{
                            'center_x': original_w // 2,
                            'center_y': original_h // 2,
                            'radius': 10,
                            'confidence': 0.0,
                            'original_size': original_size
                        }]
                    
                    result = results[0]  # 默认返回第一个检测结果
                    if return_details:
                        result['detections'] = results
                        result['outputs'] = outputs
                        result['image_tensor'] = image_tensor
                else:
                    # UNet模型处理
                    pred_heatmap, pred_coords = self.model(image_tensor)
                    
                    # 后处理坐标
                    pred_coords = pred_coords.cpu().numpy()[0]  # 取第一个样本
                    
                    # 转换到原始图像尺寸
                    original_h, original_w = original_size
                    center_x = int(pred_coords[0] * original_w)
                    center_y = int(pred_coords[1] * original_h)
                    
                    # 从热力图获取峰值位置作为验证
                    heatmap = pred_heatmap[0, 0].cpu().numpy()
                    heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
                    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), 
                                                     heatmap_resized.shape)
                    
                    # 计算置信度分数
                    confidence = float(np.max(heatmap))
                    
                    # 结合两种方法的结果
                    final_x = int(0.7 * center_x + 0.3 * peak_x)  # 权重组合
                    final_y = int(0.7 * center_y + 0.3 * peak_y)
                    
                    result = {
                        'center_x': final_x,
                        'center_y': final_y,
                        'regression_coords': (center_x, center_y),
                        'heatmap_coords': (peak_x, peak_y),
                        'confidence': confidence,
                        'original_size': original_size
                    }
                    
                    if return_details:
                        result['heatmap'] = heatmap_resized
                        result['image_tensor'] = image_tensor
        
        return result
    
    def predict_batch(self, image_paths, batch_size=4):
        """批量预测多张图像"""
        results = []
        
        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            batch_original_sizes = []
            
            # 预处理批次
            for path in batch_paths:
                tensor, orig_size = self.preprocess_image(path)
                batch_tensors.append(tensor)
                batch_original_sizes.append(orig_size)
            
            # 合并为批次tensor
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                with autocast(enabled=Config.USE_MIXED_PRECISION):
                    if self.model_type == 'yolo_v1':
                        # YOLOv1批量预测
                        outputs = self.model(batch_tensor)
                        detections = self.model.detect(
                            outputs,
                            conf_threshold=Config.YOLO_CONF_THRESHOLD,
                            nms_threshold=Config.YOLO_NMS_THRESHOLD
                        )
                        
                        # 处理每个样本的检测结果
                        for j, path in enumerate(batch_paths):
                            orig_h, orig_w = batch_original_sizes[j]
                            detection = detections[j] if len(detections) > j else []
                            
                            if len(detection) > 0:
                                # 取置信度最高的检测结果
                                best_idx = np.argmax(detection[:, 4])
                                best_det = detection[best_idx]
                                x1, y1, x2, y2, conf, class_idx = best_det
                                # 计算预测中心点
                                center_x = int((x1 + x2) // 2 * orig_w)
                                center_y = int((y1 + y2) // 2 * orig_h)
                                # 计算半径
                                radius = max((x2 - x1) // 2, (y2 - y1) // 2) * max(orig_w, orig_h)
                                confidence = conf
                            else:
                                # 如果没有预测结果，使用默认值
                                center_x, center_y = orig_w // 2, orig_h // 2
                                radius = 10
                                confidence = 0.0
                            
                            results.append({
                                'image_path': path,
                                'center_x': center_x,
                                'center_y': center_y,
                                'radius': radius,
                                'confidence': confidence,
                                'original_size': batch_original_sizes[j]
                            })
                    else:
                        # UNet批量预测
                        pred_heatmaps, pred_coords = self.model(batch_tensor)
                        
                        # 处理批次结果
                        for j, path in enumerate(batch_paths):
                            coords = pred_coords[j].cpu().numpy()
                            heatmap = pred_heatmaps[j, 0].cpu().numpy()
                            orig_h, orig_w = batch_original_sizes[j]
                            
                            # 转换坐标
                            center_x = int(coords[0] * orig_w)
                            center_y = int(coords[1] * orig_h)
                            
                            # 热力图峰值
                            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
                            peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), 
                                                             heatmap_resized.shape)
                            
                            confidence = float(np.max(heatmap))
                            
                            # 组合结果
                            final_x = int(0.7 * center_x + 0.3 * peak_x)
                            final_y = int(0.7 * center_y + 0.3 * peak_y)
                            
                            results.append({
                                'image_path': path,
                                'center_x': final_x,
                                'center_y': final_y,
                                'confidence': confidence,
                                'original_size': batch_original_sizes[j]
                            })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None, show=True):
        """可视化预测结果"""
        # 获取预测结果
        result = self.predict_single_image(image_path, return_details=True)
        
        # 读取原始图像
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 可视化
        self.plot_result(
            original_image, 
            result,
            save_path=save_path,
            show=show
        )
        
        return result
    
    def plot_result(self, image, result, save_path=None, show=True):
        """绘制预测结果"""
        center_x, center_y = result['center_x'], result['center_y']
        
        # 创建图形
        if self.model_type == 'yolo_v1':
            # YOLOv1可视化
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image)
            ax.plot(center_x, center_y, 'ro', markersize=12, 
                   label=f'Center ({center_x}, {center_y})')
            
            # 绘制边界框
            if 'bbox' in result:
                x1, y1, x2, y2 = result['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    edgecolor='blue', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            
            # 绘制半径
            if 'radius' in result:
                circle = plt.Circle((center_x, center_y), result['radius'], 
                                  edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
                ax.add_patch(circle)
                
            ax.set_title(f'Circle Detection (YOLOv1)\nConfidence: {result.get("confidence", 0.0):.3f}')
            ax.legend()
            ax.axis('off')
        else:
            # UNet可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原图 + 预测点
            axes[0].imshow(image)
            axes[0].plot(center_x, center_y, 'ro', markersize=12, 
                        label=f'Final ({center_x}, {center_y})')
            if 'regression_coords' in result:
                reg_x, reg_y = result['regression_coords']
                axes[0].plot(reg_x, reg_y, 'bo', markersize=8, 
                            label=f'Regression ({reg_x}, {reg_y})')
            if 'heatmap_coords' in result:
                hm_x, hm_y = result['heatmap_coords']
                axes[0].plot(hm_x, hm_y, 'go', markersize=8, 
                            label=f'Heatmap ({hm_x}, {hm_y})')
            axes[0].set_title(f'Circle Center Detection\nConfidence: {result.get("confidence", 0.0):.3f}')
            axes[0].legend()
            axes[0].axis('off')
            
            # 热力图叠加
            if 'heatmap' in result:
                axes[1].imshow(image, alpha=0.7)
                axes[1].imshow(result['heatmap'], alpha=0.6, cmap='hot')
                axes[1].plot(center_x, center_y, 'ro', markersize=10)
                axes[1].set_title('Heatmap Overlay')
                axes[1].axis('off')
            
            # 纯热力图
            if 'heatmap' in result:
                im = axes[2].imshow(result['heatmap'], cmap='hot')
                axes[2].plot(center_x, center_y, 'ro', markersize=8)
                axes[2].set_title('Prediction Heatmap')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def predict_and_save_results(self, input_dir, output_dir):
        """批量预测并保存结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, file))
        
        if not image_files:
            self.logger.warning(f"No image files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # 批量预测
        results = self.predict_batch(image_files)
        
        # 保存结果
        import json
        import pandas as pd
        
        # 保存JSON格式
        json_results = []
        for result in results:
            json_result = {
                'filename': os.path.basename(result['image_path']),
                'center_x': result['center_x'],
                'center_y': result['center_y'],
                'confidence': result['confidence']
            }
            if 'radius' in result:
                json_result['radius'] = result['radius']
            if 'bbox' in result:
                json_result['bbox'] = result['bbox']
            json_results.append(json_result)
        
        json_path = os.path.join(output_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 保存CSV格式
        df = pd.DataFrame(json_results)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        
        # 生成可视化结果
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for result in results[:10]:  # 只可视化前10个结果
            image_path = result['image_path']
            filename = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = os.path.join(vis_dir, f'{filename}_prediction.png')
            
            try:
                self.visualize_prediction(image_path, save_path=vis_path, show=False)
            except Exception as e:
                self.logger.warning(f"Failed to visualize {image_path}: {e}")
        
        self.logger.info(f"Results saved to {output_dir}")
        self.logger.info(f"JSON: {json_path}")
        self.logger.info(f"CSV: {csv_path}")
        self.logger.info(f"Visualizations: {vis_dir}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Circle Center Detection Prediction')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--single', action='store_true',
                       help='Predict single image (show visualization)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for batch prediction')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type to use (yolo_v1/unet_circle/unet_segmentation)')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = CirclePredictor(args.model, args.device, args.model_type)
    
    if args.single or os.path.isfile(args.input):
        # 单张图像预测
        if not os.path.isfile(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        print(f"Predicting single image: {args.input}")
        result = predictor.visualize_prediction(args.input, show=True)
        
        print(f"Prediction Results:")
        print(f"  Center: ({result['center_x']}, {result['center_y']}) pixels")
        if 'regression_coords' in result:
            print(f"  Regression: {result['regression_coords']}")
        if 'heatmap_coords' in result:
            print(f"  Heatmap Peak: {result['heatmap_coords']}")
        if 'radius' in result:
            print(f"  Radius: {result['radius']} pixels")
        print(f"  Confidence: {result['confidence']:.3f}")
        
    else:
        # 批量预测
        if not os.path.isdir(args.input):
            print(f"Error: Input directory not found: {args.input}")
            return
        
        print(f"Batch prediction from directory: {args.input}")
        results = predictor.predict_and_save_results(args.input, args.output)
        
        if results:
            # 统计信息
            confidences = [r['confidence'] for r in results]
            print(f"\nPrediction Summary:")
            print(f"  Total images: {len(results)}")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  Min confidence: {np.min(confidences):.3f}")
            print(f"  Max confidence: {np.max(confidences):.3f}")

if __name__ == '__main__':
    main()
