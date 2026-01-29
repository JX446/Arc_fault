import tensorflow as tf
from pathlib import Path
import tf2onnx

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, 1, padding='same', input_shape=(16, 16, 1)),  # NHWC: [H, W, C]
    tf.keras.layers.ReLU()
])

model.save(Path.cwd() / ("model" + ".h5"))

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([1, 16, 16, 1], tf.float32, name="input")],
    opset=13
)

# 保存
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())