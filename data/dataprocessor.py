import os
import json
import shutil
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

class DataProcessor:
    def __init__(self, source_dir="../datasets", target_dir="../datasets_processed"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_images = os.path.join(source_dir, "oriData")
        self.source_annotations = os.path.join(source_dir, "annotations")
        
    def create_directories(self):
        """创建目标目录结构"""
        dirs = [
            os.path.join(self.target_dir, "train", "images"),
            os.path.join(self.target_dir, "train", "annotations"),
            os.path.join(self.target_dir, "val", "images"),
            os.path.join(self.target_dir, "val", "annotations"),
            os.path.join(self.target_dir, "test", "images"),
            os.path.join(self.target_dir, "test", "annotations")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def get_data_files(self):
        """获取所有有效的数据文件对"""
        image_files = [f for f in os.listdir(self.source_images) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        valid_pairs = []
        for img_file in image_files:
            json_file = img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json')
            json_path = os.path.join(self.source_annotations, json_file)
            
            if os.path.exists(json_path):
                valid_pairs.append((img_file, json_file))
            else:
                print(f"Warning: No annotation found for {img_file}")
        
        return valid_pairs
    
    def split_data(self, valid_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """分割数据集"""
        random.shuffle(valid_pairs)
        
        total = len(valid_pairs)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_pairs = valid_pairs[:train_size]
        val_pairs = valid_pairs[train_size:train_size + val_size]
        test_pairs = valid_pairs[train_size + val_size:]
        
        print(f"数据分割:")
        print(f"Total: {total}")
        print(f"Train: {len(train_pairs)}")
        print(f"Val: {len(val_pairs)}")
        print(f"Test: {len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs
    
    def get_augmentation_transforms(self):
        """定义数据增强变换"""
        transforms = [
            # 几何变换
            A.Compose([
                A.Rotate(limit=15, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy')),
            
            A.Compose([
                A.RandomScale(scale_limit=0.2, p=0.8),
                A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.8),
            ], keypoint_params=A.KeypointParams(format='xy')),
            
            # 亮度对比度变换
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], keypoint_params=A.KeypointParams(format='xy')),
            
            # 噪声和模糊
            A.Compose([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy')),
            
            # 颜色变换
            A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy')),
        ]
        
        return transforms
    
    def apply_augmentation(self, image, keypoint, transform):
        """应用数据增强"""
        try:
            transformed = transform(image=image, keypoints=[keypoint])
            aug_image = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            if len(aug_keypoints) > 0:
                return aug_image, aug_keypoints[0]
            else:
                # 如果关键点变换失败，返回原始数据
                return image, keypoint
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, keypoint
    
    def update_json_annotation(self, json_path, new_keypoint, new_image_size, new_filename, subset_name):
        """更新JSON标注文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新关键点坐标
        for shape in data['shapes']:
            if shape['label'] == 'center' and shape['shape_type'] == 'point':
                shape['points'] = [list(new_keypoint)]
                break
        
        # 更新图片信息 - 修正imagePath
        data['imagePath'] = f"../images/{new_filename}"  # 从annotations文件夹到images文件夹的相对路径
        data['imageWidth'] = new_image_size[1]
        data['imageHeight'] = new_image_size[0]
        
        return data

    
    def copy_original_data(self, pairs, subset_name):
        """复制原始数据"""
        count = 0
        for img_file, json_file in pairs:
            # 复制图片
            src_img = os.path.join(self.source_images, img_file)
            dst_img = os.path.join(self.target_dir, subset_name, "images", img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标注并更新imagePath
            src_json = os.path.join(self.source_annotations, json_file)
            
            # 读取原始标注
            with open(src_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 更新imagePath
            json_data['imagePath'] = f"../images/{img_file}"
            
            # 保存更新后的标注
            dst_json = os.path.join(self.target_dir, subset_name, "annotations", json_file)
            with open(dst_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            count += 1
        
        return count

    def augment_training_data(self, train_pairs, target_count=200):
        """对训练数据进行增强"""
        transforms = self.get_augmentation_transforms()
        current_count = len(train_pairs)
        augmentation_count = 0
        
        print(f"开始数据增强，目标数量: {target_count}, 当前数量: {current_count}")
        
        while current_count < target_count:
            for img_file, json_file in train_pairs:
                if current_count >= target_count:
                    break
                
                # 加载图片和标注
                img_path = os.path.join(self.source_images, img_file)
                json_path = os.path.join(self.source_annotations, json_file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 加载标注
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 获取关键点
                keypoint = None
                for shape in json_data['shapes']:
                    if shape['label'] == 'center' and shape['shape_type'] == 'point':
                        keypoint = tuple(shape['points'][0])
                        break
                
                if keypoint is None:
                    continue
                
                # 随机选择一个增强变换
                transform = random.choice(transforms)
                aug_image, aug_keypoint = self.apply_augmentation(image, keypoint, transform)
                
                # 生成新的文件名
                base_name = os.path.splitext(img_file)[0]
                aug_img_name = f"{base_name}_aug_{augmentation_count:03d}.jpg"
                aug_json_name = f"{base_name}_aug_{augmentation_count:03d}.json"
                
                # 保存增强后的图片
                aug_img_path = os.path.join(self.target_dir, "train", "images", aug_img_name)
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_img_path, aug_image_bgr)
                
                # 保存增强后的标注 - 传入subset_name参数
                aug_json_data = self.update_json_annotation(
                    json_path, aug_keypoint, aug_image.shape, aug_img_name, "train"
                )
                aug_json_path = os.path.join(self.target_dir, "train", "annotations", aug_json_name)
                with open(aug_json_path, 'w', encoding='utf-8') as f:
                    json.dump(aug_json_data, f, ensure_ascii=False, indent=2)
                
                augmentation_count += 1
                current_count += 1
                
                if current_count % 50 == 0:
                    print(f"已生成 {current_count} 个训练样本")
        
        print(f"数据增强完成，共生成 {augmentation_count} 个增强样本")
        return augmentation_count

    def process_all_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, target_train_count=200):
        """处理所有数据"""
        print("开始处理数据...")
        
        # 创建目录结构
        self.create_directories()
        
        # 获取所有有效数据
        valid_pairs = self.get_data_files()
        if len(valid_pairs) == 0:
            print("没有找到有效的数据文件!")
            return
        
        # 分割数据
        train_pairs, val_pairs, test_pairs = self.split_data(
            valid_pairs, train_ratio, val_ratio, test_ratio
        )
        
        # 复制原始数据
        print("\n复制原始数据...")
        train_original = self.copy_original_data(train_pairs, "train")
        val_count = self.copy_original_data(val_pairs, "val")
        test_count = self.copy_original_data(test_pairs, "test")
        
        print(f"原始数据复制完成:")
        print(f"Train原始: {train_original}")
        print(f"Val: {val_count}")
        print(f"Test: {test_count}")
        
        # 对训练数据进行增强
        if target_train_count > train_original:
            print(f"\n开始训练数据增强...")
            aug_count = self.augment_training_data(train_pairs, target_train_count)
            total_train = train_original + aug_count
        else:
            total_train = train_original
        
        print(f"\n数据处理完成!")
        print(f"最终数据分布:")
        print(f"Train: {total_train}")
        print(f"Val: {val_count}")
        print(f"Test: {test_count}")
        print(f"Total: {total_train + val_count + test_count}")

def main():
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 创建处理器
    processor = DataProcessor(
        source_dir="../datasets",           # 原始数据目录
        target_dir="../datasets_processed"  # 处理后的数据目录
    )
    
    # 处理所有数据
    processor.process_all_data(
        train_ratio=0.6,      # 训练集比例
        val_ratio=0.2,        # 验证集比例
        test_ratio=0.2,       # 测试集比例
        target_train_count=200 # 训练集目标数量（包括增强）
    )

if __name__ == "__main__":
    main()
