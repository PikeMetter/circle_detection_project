import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.unet_segmentation import UNetSegmentation
from config import Config

def load_model(model_path, device):
    """加载训练好的分割模型"""
    model = UNetSegmentation(in_channels=3, out_channels=1).to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """预处理输入图像"""
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 保存原始尺寸
    original_shape = image.shape[:2]
    
    # 调整图像大小
    image_resized = cv2.resize(image, target_size)
    
    # 转换为tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)  # 添加batch维度
    
    return image_tensor, original_shape, image_resized

def postprocess_mask(mask, original_shape):
    """后处理分割掩码"""
    # 转换为numpy数组
    mask_np = mask.squeeze().cpu().numpy()
    
    # 调整回原始尺寸
    mask_resized = cv2.resize(mask_np, (original_shape[1], original_shape[0]))
    
    # 二值化
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    
    return mask_binary, mask_resized

def find_circle_center(mask):
    """从分割掩码中找到圆形中心"""
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # 计算轮廓的质心
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), (int(x), int(y)), int(radius)
    
    return None, None, None

def predict_single_image(model, image_path, device, output_dir="results/segmentation"):
    """预测单张图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 预处理图像
    image_tensor, original_shape, image_resized = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 模型预测
    with torch.no_grad():
        pred_mask = model(image_tensor)
    
    # 后处理掩码
    mask_binary, mask_prob = postprocess_mask(pred_mask, original_shape)
    
    # 从掩码中找到圆形中心
    centroid, circle_center, radius = find_circle_center(mask_binary)
    
    # 可视化结果
    image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    result_image = image_bgr.copy()
    
    # 在结果图像上绘制分割掩码
    mask_overlay = np.zeros_like(result_image)
    mask_overlay[mask_binary > 0] = [0, 255, 0]  # 绿色掩码
    result_image = cv2.addWeighted(result_image, 0.7, mask_overlay, 0.3, 0)
    
    # 绘制检测到的中心点
    if centroid is not None:
        # 质心（红色）
        cv2.circle(result_image, centroid, 5, (0, 0, 255), -1)
        # 最小外接圆中心（蓝色）
        cv2.circle(result_image, circle_center, 5, (255, 0, 0), -1)
        # 最小外接圆
        cv2.circle(result_image, circle_center, radius, (255, 0, 0), 2)
        
        print(f"检测到的圆形中心点:")
        print(f"  质心: ({centroid[0]}, {centroid[1]})")
        print(f"  最小外接圆中心: ({circle_center[0]}, {circle_center[1]})")
        print(f"  半径: {radius}")
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # 保存原始图像
    cv2.imwrite(os.path.join(output_dir, f"{name}_original{ext}"), image_bgr)
    
    # 保存分割掩码
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask_binary.png"), mask_binary)
    mask_prob_vis = (mask_prob * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask_prob.png"), mask_prob_vis)
    
    # 保存结果图像
    cv2.imwrite(os.path.join(output_dir, f"{name}_result{ext}"), result_image)
    
    print(f"结果已保存到 {output_dir} 目录")
    
    return centroid, circle_center, radius

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='圆形分割预测')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='results/segmentation', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device(args.device)
    model = load_model(args.model, device)
    print(f"模型已加载到设备: {device}")
    
    # 预测
    if os.path.isfile(args.input):
        # 单张图像预测
        centroid, circle_center, radius = predict_single_image(
            model, args.input, device, args.output
        )
    elif os.path.isdir(args.input):
        # 批量预测
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(args.input, image_file)
            print(f"\n处理图像: {image_path}")
            predict_single_image(model, image_path, device, args.output)
    else:
        print("输入路径既不是文件也不是目录")
        return
    
    print("预测完成!")

if __name__ == "__main__":
    main()