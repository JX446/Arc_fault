import os
import pandas as pd
import numpy as np
# 关闭 oneDNN 优化提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 屏蔽 TensorFlow 大多数 INFO/WARNING 日志（0=all, 1=WARNING+, 2=ERROR+, 3=FATAL+）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from getData import GetData
from processData import DataProcessor
from CNNModel import ModelTrainer_DL

if __name__ == "__main__":
    
    # 从CSV文件加载数据
    test_conditions = {
        # "607V_8p5A_300nF": ["arc_3", "normal_3"],
        "607V_8p5A_300nF": ["arc_0", "normal_0", "arc_1", "normal_1", "arc_2", "normal_2"]
    }
    data_loader = GetData(path="./Data", windowSize=1024, test_conditions=test_conditions)
    X_train, X_test, y_train, y_test = data_loader.get_data()

    # 窗口数据保存为CSV文件
    raw_train_df = pd.DataFrame(np.hstack((X_train, np.array(y_train).reshape(-1, 1))))
    raw_test_df = pd.DataFrame(np.hstack((X_test, np.array(y_test).reshape(-1, 1))))
    raw_test_df.to_csv("raw_test.csv", index=False)
    raw_train_df.to_csv("raw_train.csv", index=False)

#############################################################################################

    # 处理数据, 提取特征值
    # 1. 初始化
    data_processor = DataProcessor()

    # 全局打乱一次，保证 X 和 y 完全随机
    X_train, y_train = data_processor.shuffleData(X_train, y_train)

    # 在训练集上 fit+transform，测试集上只 transform
    X_train = data_processor.processData(X_train, y_train, fit_scaler=True)
    X_test  = data_processor.processData(X_test, y_test, fit_scaler=False)

    # 再次打乱训练集顺序再喂模型
    X_train, y_train = data_processor.shuffleData(X_train, y_train)

##############################################################################################

    # 训练模型
    trainer = ModelTrainer_DL()
    result = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    print(result)
 