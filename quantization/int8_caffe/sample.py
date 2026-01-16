#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt
import os
import numpy as np

# For our custom calibrator
from calibrator import load_data, load_labels, CustomEntropyCalibrator

# For ../common.py
import sys
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

TRT_LOGGER = trt.Logger()


class ModelData(object):
    DEPLOY_PATH = "./deploy.prototxt"
    MODEL_PATH = "./model.caffemodel"
    OUTPUT_NAME = "prob" # top name.
    # The original model is a float32 one.
    DTYPE = trt.float32


# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser, trt.Runtime(TRT_LOGGER) as runtime:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        # Parse Caffe model
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        plan = builder.build_serialized_network(network, config)
        return runtime.deserialize_cuda_engine(plan)


def check_accuracy(context, batch_size, test_set, test_labels):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    num_correct = 0
    num_total = 0

    batch_num = 0
    for start_idx in range(0, test_set.shape[0], batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print("Validating batch {:}".format(batch_num))
        # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
        # This logic is used for handling that case.
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx

        # Do inference for every batch.
        inputs[0].host = test_set[start_idx:start_idx + effective_batch_size]
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=effective_batch_size)

        # Use argmax to get predictions and then check accuracy
        # print(output)
        preds = np.argmax(output.reshape(batch_size, 2)[0:effective_batch_size], axis=1)
        labels = test_labels[start_idx:start_idx + effective_batch_size]
        num_total += effective_batch_size
        num_correct += np.count_nonzero(np.equal(preds, labels))

        # Output the predictions and the true labels for each batch
        for i in range(effective_batch_size):
            print(f"Sample {start_idx + i}: Predicted = {preds[i]}, Actual = {labels[i]}")

    percent_correct = 100 * num_correct / float(num_total)
    print("Total Accuracy: {:}%".format(percent_correct))


def main():
    # 使用原始字符串，避免转义字符问题
    test_images_dir = "./data/test/"
    test_labels_path = "./data/test.txt"

    # print(f"图像文件夹路径: {test_images_dir}")
    # print(f"标签文件路径: {test_labels_path}")


    # 加载数据
    test_set = load_data(test_images_dir)       # 加载图像数据
    test_labels = load_labels(test_labels_path) # 加载图像标签
    # print(test_labels)

    # 指定模型结构文件和模型权重文件
    deploy_file = ModelData.DEPLOY_PATH
    model_file = ModelData.MODEL_PATH
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = "calibration.cache"
    calib = CustomEntropyCalibrator(test_set, cache_file=calibration_cache)

    # Inference batch size can be different from calibration batch size.
    batch_size = 32
    with build_int8_engine(deploy_file, model_file, calib, batch_size) as engine, engine.create_execution_context() as context:
        # Batch size for inference can be different than batch size used for calibration.
        check_accuracy(context, batch_size, test_set=test_set, test_labels=test_labels)

if __name__ == '__main__':
    main()
 
 