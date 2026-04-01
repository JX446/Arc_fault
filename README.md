# Arc-fault-detection

电弧故障检测模型 - 基于1D CNN的信号分类

## 项目结构

```
./
├── Data/                    # 数据目录
│   ├── class_0_normal/      # 正常样本
│   └── class_1_arc/         # 电弧故障样本
├── model_convert/           # 模型转换工具
│   ├── gen_model_tf.py      # TensorFlow模型导出
│   ├── onnx_to_tvm.py       # ONNX转TVM
│   └── test_onnx.py         # ONNX测试
├── quantization/            # 量化工具
│   └── int8_caffe/          # INT8量化
├── CNNModel.py              # 神经网络模型定义
├── getData.py               # 数据加载
├── main.py                  # 主程序入口
├── processData.py           # 数据预处理
└── README.md
```

## 网络结构

### 模型配置（param3，最终使用）

| 层类型 | 参数 |
|--------|------|
| Input | shape=(256, 1) |
| Conv1D | filters=8, kernel_size=8, activation='relu', padding='same' |
| MaxPooling1D | pool_size=2 |
| Flatten | - |
| Dense | units=1, activation='sigmoid' |

### 编译配置

- 优化器：Adam
- 损失函数：binary_crossentropy
- 评估指标：accuracy

### 备选模型

- param1：filters=2
- param2：filters=4
- param3：filters=8（默认使用）

## 输入数据格式

### 原始数据

- 来源：CSV文件，存储在 `./Data/class_0_normal/` 和 `./Data/class_1_arc/`
- 文件命名：`{电压}_{电流}_{电容}/arc_N.csv` 或 `normal_N.csv`
- 示例：`607V_8p5A_300nF/arc_0.csv`

### 预处理流程

```
原始信号 (CSV)
    ↓
窗口划分 (windowSize=1024)
    ↓
FFT变换 → (N, 513)
    ↓
取幅度（去直流分量）→ (N, 512)
    ↓
能量池化 (poolSize=256段) → (N, 256)
    ↓
对数变换: 10*log(pooled + 1e-10) - 10*log(1024)
    ↓
Z-score标准化 (StandardScaler)
    ↓
Reshape → (N, 256, 1)
```

最终输入形状：`(N, 256, 1)`

## 输出数据格式

| 项目 | 说明 |
|------|------|
| 输出形状 | (N, 1) |
| 激活函数 | sigmoid |
| 输出范围 | [0, 1] |
| 判定阈值 | 0.5 |
| 分类标签 | 0=normal（正常）, 1=arc（电弧故障） |

## 数据输入方式

### 完整流程

```python
# 1. 数据加载
data_loader = GetData(path="./Data", windowSize=1024, test_conditions={...})
X_train, X_test, y_train, y_test = data_loader.get_data()

# 2. 数据预处理
data_processor = DataProcessor()
X_train = data_processor.processData(X_train, y_train, fit_scaler=True)   # 训练集：fit+transform
X_test  = data_processor.processData(X_test, y_test, fit_scaler=False)     # 测试集：仅transform

# 3. 打乱数据
X_train, y_train = data_processor.shuffleData(X_train, y_train)

# 4. 训练模型
trainer = ModelTrainer_DL()
result = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

# 5. 预测
y_pred_proba = cnn_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
```

### 训练参数

- epochs: 10
- batch_size: 32
- verbose: 1

## 评估指标

- FPR (False Positive Rate)
- FNR (False Negative Rate)
- Precision
- Recall
- Accuracy
- F1, F2, F0.5 Score

## 运行

```bash
python main.py
```

## 数据文件格式

CSV文件命名规范：`{电压}V_{电流}A_{电容}nF/arc_{编号}.csv` 或 `{电压}V_{电流}A_{电容}nF/normal_{编号}.csv`

示例：`607V_8p5A_300nF/arc_0.csv`
