import torch
import torch.nn as nn
c = 8
k = 8

class Model(nn.Module):
    def __init__(self, input_channel=c, kernel_nums=k):
        super().__init__()
        self.input_channel = input_channel
        self.kernel_nums = kernel_nums
        self.layer1 = nn.Conv2d(input_channel, kernel_nums, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.layer1(x)
        return x

model = Model().to(memory_format=torch.channels_last)
model.eval()
batch_size = 1
seq_length = 256
dummy_input = torch.randn(batch_size, c, 16, 64)


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

