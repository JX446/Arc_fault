import onnx
import tvm
from tvm import relax
from tvm.contrib import utils

# ----------------------------
# 1. 加载 ONNX 模型
# ----------------------------
onnx_model = onnx.load("your_model.onnx")

# ----------------------------
# 2. 定义输入 shape 和 dtype
# ----------------------------
input_name = "input"  # 替换为你的模型实际输入名（可用 Netron 查看）
input_shape = (1, 3, 224, 224)  # 示例：ImageNet 输入
input_dtype = "float32"

shape_dict = {input_name: input_shape}
dtype_dict = {input_name: input_dtype}

# ----------------------------
# 3. 转换为 Relay IR
# ----------------------------
mod, params = relax.frontend.from_onnx(
    onnx_model,
    shape=shape_dict,
    dtype=dtype_dict,
)

# ----------------------------
# 4. 设置目标平台（根据你的部署设备修改）
# ----------------------------
# 例如：
target = "llvm"                     # x86 CPU（Linux/macOS/Windows）
# target = "llvm -mtriple=aarch64-linux-gnu"  # ARM64 Linux（如树莓派）
# target = "c"                        # 纯 C（用于微控制器）
# target = "cuda"                     # NVIDIA GPU

# ----------------------------
# 5. AOT 编译（关键！）
# ----------------------------
with tvm.transform.PassContext(opt_level=3):
    lib = relax.build(
        mod,
        target=target,
        params=params,
        # 👇 指定使用 AOT 执行器（生成 C 接口）
        executor=relax.backend.Executor("aot", {
            "interface_api": "c",
            "unpacked_api": False,  # 推荐 False，使用全局符号
        }),
        runtime=relax.backend.Runtime("cpp"),
    )

# ----------------------------
# 6. 导出为 tar 包（包含 .a 和 .h）
# ----------------------------
lib.export_library("model.tar")
print("✅ 编译完成！生成 model.tar")