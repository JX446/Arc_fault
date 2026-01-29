import torch
import torch.nn as nn
c = 1

class Model(nn.Module):
    def __init__(self, input_channel=c, kernel_nums=16):
        super().__init__()
        self.input_channel = input_channel
        self.kernel_nums = kernel_nums
        self.layer1 = nn.Conv1d(input_channel, kernel_nums, kernel_size=3)

    def forward(self, x):
        x = self.layer1(x)
        return x

model = Model()
model.eval()
batch_size = 1
seq_length = 256
dummy_input = torch.randn(batch_size, c, seq_length)

# 导出模型
onnx_path = "model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

