import os
import json
import shutil

def rename_files_to_numbers(ori_data_dir, annotation_dir):
    """
    将图片和标注文件重命名为数字格式
    """
    # 获取所有jpg文件
    jpg_files = [f for f in os.listdir(ori_data_dir) if f.lower().endswith('.jpg')]
    jpg_files.sort()  # 排序确保一致性
    
    # 创建重命名映射
    rename_mapping = {}
    
    for i, old_jpg in enumerate(jpg_files):
        new_name = f"{i+1:03d}"  # 001, 002, 003...
        old_jpg_path = os.path.join(ori_data_dir, old_jpg)
        new_jpg_path = os.path.join(ori_data_dir, f"{new_name}.jpg")
        
        # 重命名jpg文件
        shutil.move(old_jpg_path, new_jpg_path)
        
        # 查找对应的json文件
        old_json = old_jpg.replace('.jpg', '.json')
        old_json_path = os.path.join(annotation_dir, old_json)
        
        if os.path.exists(old_json_path):
            new_json_path = os.path.join(annotation_dir, f"{new_name}.json")
            
            # 读取并更新json内容
            with open(old_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 更新imagePath
            json_data['imagePath'] = f"..\\oriData\\{new_name}.jpg"
            
            # 保存更新的json文件
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # 删除旧json文件
            os.remove(old_json_path)
            
            rename_mapping[old_jpg] = f"..\\oriData\\{new_name}.jpg"
            print(f"重命名: {old_jpg} -> ..\\oriData\\{new_name}.jpg")
        else:
            print(f"警告: 找不到对应的json文件: {old_json}")
    
    return rename_mapping

if __name__ == "__main__":
    ori_data_dir = "oriData"  # 相对路径
    annotation_dir = "annotations"  # 相对路径
    rename_mapping = rename_files_to_numbers(ori_data_dir, annotation_dir)
