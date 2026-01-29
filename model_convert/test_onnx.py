import numpy as np
import onnxruntime as ort

# 加载模型
sess = ort.InferenceSession("model.onnx")

# 查看输入/输出信息
print("Inputs:")
for inp in sess.get_inputs():
    print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("Outputs:")
for out in sess.get_outputs():
    print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

# 准备输入
input_name = sess.get_inputs()[0].name
input_data = np.random.randn(1, 1, 256).astype(np.float32)

outputs = sess.run(None, {input_name: input_data})

# outputs 是一个列表，按输出顺序排列
output = outputs[0]
print("Output shape:", output.shape)
print("Output sample:", output[0][:5])  # 打印前5个值