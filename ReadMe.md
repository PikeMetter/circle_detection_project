# 1. 预测单张图像
python predict.py --model checkpoints/best_model.pth --input test_image.jpg --single

# 2. 批量预测整个文件夹
python predict.py --model checkpoints/best_model.pth --input datasets/test/images --output results/predictions

# 3. 在测试集上评估模型
python test.py --model checkpoints/best_model.pth --test_data datasets/test

# 4. 使用CPU进行预测
python predict.py --model checkpoints/best_model.pth --input test_image.jpg --device cpu --single
