import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def processData(self, data_batch, labels, fit_scaler = True, windowSize = 1024, poolSize = 256):
        """
        输入：
            signal_batch: numpy array of shape (N, windowSize)
        输出：
            features: numpy array of shape (N, poolSize, 1)
        """
        N, L = data_batch.shape
        assert L == windowSize, "输入数据形状与窗口大小不匹配"

        # 1. FFT + magnitude，取前 windowsize / 2 点
        fft_vals = np.fft.rfft(data_batch, axis=1)  # shape: (N, windowSize / 2 + 1)
        fft_mag = np.abs(fft_vals[:, 1:])     # shape: (N, windowSize / 2)

        # 2. 能量池化为 poolSize 段
        segment_size = (windowSize // 2) // poolSize   # 512 / 256 = 2
        pooled = []
        for i in range((windowSize // 2) // segment_size):
            seg = fft_mag[:, i * segment_size:(i + 1) * segment_size]
            energy = np.sum(seg ** 2, axis=1)      # 每个样本对应一个能量值
            pooled.append(energy)                  # poolSize 个 list，每个 shape (N,)
        pooled = np.stack(pooled, axis=1)          # shape: (N, poolSize)

        # 3. 能量取log 
        log_pooled = 10 * np.log(pooled + 1e-10) - 10 * np.log(windowSize) # shape: (N, poolSize)

        # 4. Z-score 标准化（对每一行标准化）
        if fit_scaler:
            normalized = self.scaler.fit_transform(log_pooled)
            raw_train_df = pd.DataFrame(np.hstack((normalized, np.array(labels).reshape(-1, 1))))
            raw_train_df.to_csv("model_train.csv", index=False)                      
        else:
            normalized = self.scaler.transform(log_pooled)
            raw_train_df = pd.DataFrame(np.hstack((normalized, np.array(labels).reshape(-1, 1))))
            raw_train_df.to_csv("model_test.csv", index=False)
            print("均值:", self.scaler.mean_)
            print("方差:", self.scaler.scale_)
            print("标准差:", self.scaler.var_) 
        
        # 5. reshape 模型输入需为高维数据
        return normalized.reshape(N, poolSize, 1)
    
        # 6. 量化模型输入数据需为INT8类型, 具体实现在csv2JPG.py

    def shuffleData(self, data_batch, labels):
        assert len(data_batch) == len(labels)
        indices = np.arange(len(data_batch))
        np.random.shuffle(indices)
        return data_batch[indices], labels[indices]