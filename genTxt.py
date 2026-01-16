import os

# 训练集和测试集路径
train_path = r'C:\Users\djy\OneDrive\Desktop\work\QX\Arc-fault-detection\new\Data_jpg\train'
test_path = r'C:\Users\djy\OneDrive\Desktop\work\QX\Arc-fault-detection\new\Data_jpg\test'

# 子目录名称（类别）
subPath = ['class_0', 'class_1']
# 存放 train.txt 和 test.txt 的位置
restoreFile = r'C:\Users\djy\OneDrive\Desktop\work\QX\Arc-fault-detection\new\Data_jpg'

# 清空原来的 train.txt 和 test.txt，避免多次运行时数据重复
restoreFile_train = os.path.join(restoreFile, 'train.txt')
restoreFile_test = os.path.join(restoreFile, 'test.txt')
open(restoreFile_train, 'w').close()  # 清空 train.txt
open(restoreFile_test, 'w').close()  # 清空 test.txt

# 生成 train.txt
with open(restoreFile_train, 'w') as f_train:
    for i, class_name in enumerate(subPath):
        train_class_path = os.path.join(train_path, class_name)
        if not os.path.exists(train_class_path):
            raise Exception(f'Error: {train_class_path} does not exist')
        for file_name in os.listdir(train_class_path):
            file_path = os.path.join(class_name, file_name).replace("\\", "/")  # 替换 `\` 为 `/`
            f_train.write(f"{file_path} {i}\n")

# 生成 test.txt
with open(restoreFile_test, 'w') as f_test:
    for i, class_name in enumerate(subPath):
        test_class_path = os.path.join(test_path, class_name)
        if not os.path.exists(test_class_path):
            raise Exception(f'Error: {test_class_path} does not exist')
        for file_name in os.listdir(test_class_path):
            file_path = os.path.join(class_name, file_name).replace("\\", "/")  # 替换 `\` 为 `/`
            f_test.write(f"{file_path} {i}\n")

print("train.txt 和 test.txt 生成完成！")
