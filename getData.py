import os
import re
import glob
import numpy as np
import pandas as pd

class GetData:
    def __init__(self, path, windowSize, test_conditions=None):
        self.path = path
        self.windowSize = int(windowSize)
        # test_conditions 是 dict: {组合键: [子文件关键字列表]}
        self.test_conditions = test_conditions or {}

    def extract_condition_key(self, filename):
        """
        提取电压_电流_电容 组合键，如 318V_8A_300nF
        """
        match = re.search(r'(\d+V)_(\d+(?:p?\d*)A)_(\d+nF)', filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
        else:
            raise ValueError(f"无法从文件名中提取条件组合: {filename}")

    def get_file_tag(self, filename):
        """
        从文件名中提取如 arc_2 或 normal_1
        """
        match = re.search(r'(arc|normal)_(\d+)\.csv$', filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
        else:
            return None

    def load_signals_from_files(self, files, label):
        X, y = [], []
        for file in files:
            try:
                df = pd.read_csv(file, header=None)
                signal = df.iloc[:, 0].dropna().values
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            total_windows = len(signal) // self.windowSize
            for i in range(total_windows):
                start = i * self.windowSize
                end = start + self.windowSize
                segment = signal[start:end]
                X.append(segment)
                y.append(label)
        return X, y

    def get_data(self):
        X_train, y_train, X_test, y_test = [], [], [], []

        for class_label in [0, 1]:
            folder = f"class_{class_label}_{'normal' if class_label == 0 else 'arc'}"
            full_path = os.path.join(self.path, folder)
            files = glob.glob(os.path.join(full_path, "*.csv"))

            for file in files:
                filename = os.path.basename(file)
                condition = self.extract_condition_key(filename)
                tag = self.get_file_tag(filename)
                label = class_label

                if (condition in self.test_conditions) and (tag in self.test_conditions[condition]):
                    # 加入测试集
                    X_part, y_part = self.load_signals_from_files([file], label)
                    X_test.extend(X_part)
                    y_test.extend(y_part)
                else:
                    # 加入训练集
                    X_part, y_part = self.load_signals_from_files([file], label)
                    X_train.extend(X_part)
                    y_train.extend(y_part)

        print("\n 训练样本数:", len(X_train), "测试样本数:", len(X_test))
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


