import os
import pandas as pd

# 读取CSV文件
train_csv_file = "model_train.csv"  # 训练数据 CSV 文件
test_csv_file = "model_test.csv"    # 测试数据 CSV 文件

# 读取数据
train_data = pd.read_csv(train_csv_file, header=0).values
test_data = pd.read_csv(test_csv_file, header=0).values

# 获取所有样本的数据（不包括标签）
train_samples = train_data[:, :-1]  # 去掉标签列
test_samples = test_data[:, :-1]    # 去掉标签列


# 创建 train 和 test 文件夹及分类子文件夹
os.makedirs("Data_caffe/train/class_0", exist_ok=True)
os.makedirs("Data_caffe/train/class_1", exist_ok=True)
os.makedirs("Data_caffe/test/class_0", exist_ok=True)
os.makedirs("Data_caffe/test/class_1", exist_ok=True)

# 函数：处理并保存数据
def process_and_save_data(data, folder_prefix):
    for i, row in enumerate(data):
        label = int(row[-1])             # 读取最后一列作为标签

        # 确定保存路径
        folder = f"{folder_prefix}/class_0" if label == 0 else f"{folder_prefix}/class_1"
        file_path = os.path.join(folder, f"sample_{i+1}.jpg")

# 处理并保存训练数据
process_and_save_data(train_data, "Data_caffe/train")

# 处理并保存测试数据
process_and_save_data(test_data, "Data_caffe/test")

print("所有样本已分类并保存到 Data_caffe/train 和 Data_caffe/test 文件夹中。")

