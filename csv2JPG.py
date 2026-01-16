import os

import numpy as np
import pandas as pd
from PIL import Image

# 读取CSV文件
train_csv_file = "model_train.csv"  # 训练数据 CSV 文件
test_csv_file = "model_test.csv"  # 测试数据 CSV 文件

# 读取数据
train_data = pd.read_csv(train_csv_file, header=0).values
test_data = pd.read_csv(test_csv_file, header=0).values

# 获取所有样本的数据（不包括标签）
train_samples = train_data[:, :-1]  # 去掉标签列
test_samples = test_data[:, :-1]  # 去掉标签列

# 计算训练数据截断区域
train_mean = np.mean(train_samples)
train_std = np.std(train_samples)
train_lower_bound = train_mean - 2 * train_std
train_upper_bound = train_mean + 2 * train_std
print(f"训练数据的下截断点: {train_lower_bound}, 上截断点: {train_upper_bound}")
test_mean = np.mean(test_samples)
test_std = np.std(test_samples)
test_lower_bound = test_mean - 2 * test_std
test_upper_bound = test_mean + 2 * test_std
print(f"测试数据的下截断点: {test_lower_bound}, 上截断点: {test_upper_bound}")
# 创建 train 和 test 文件夹及分类子文件夹
os.makedirs("Data_jpg/train/class_0", exist_ok=True)
os.makedirs("Data_jpg/train/class_1", exist_ok=True)
os.makedirs("Data_jpg/test/class_0", exist_ok=True)
os.makedirs("Data_jpg/test/class_1", exist_ok=True)


# 函数：处理并保存数据
def process_and_save_data(data, folder_prefix, data_min, data_max):
    for i, row in enumerate(data):
        sample = row[:-1]
        label = int(row[-1])  # 读取最后一列作为标签

        # 数据截断：限制在 [data_min, data_max] 范围内
        sample_clipped = np.clip(
            sample, data_min, data_max
        )  # 限制在 [data_min, data_max] 范围内

        # Min-Max 归一化：使用训练数据的最小值和最大值
        sample_normalized = (sample_clipped - data_min) / (data_max - data_min)

        # 确定保存路径
        folder = (
            f"{folder_prefix}/class_0" if label == 0 else f"{folder_prefix}/class_1"
        )
        file_path = os.path.join(folder, f"sample_{i + 1}.jpg")

        # 转换数据为 8-bit 灰度图
        sample_normalized = (sample_normalized * 255).astype(
            np.uint8
        )  # 归一化数据已在 0-1 之间，转换为 0-255
        # print(f"Sample {i+1} - Normalized Data:\n", sample_normalized)
        img = Image.fromarray(sample_normalized, mode="L")  # "L" 表示灰度图

        # 保存图片
        img.save(file_path)


# 处理并保存训练数据
process_and_save_data(
    train_data, "Data_jpg/train", train_lower_bound, train_upper_bound
)

# 处理并保存测试数据
process_and_save_data(test_data, "Data_jpg/test", test_lower_bound, test_upper_bound)

print("所有样本已分类并保存到 Data_jpg/train 和 Data_jpg/test 文件夹中。")
