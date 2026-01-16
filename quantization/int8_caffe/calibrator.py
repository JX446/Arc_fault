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
import cv2 as cv
import pycuda.driver as cuda
import numpy as np

def load_data(filepath):
    test_imgs = []
    global fileList
    fileList = os.listdir(filepath)
    for img_name in fileList:
        # print(img_name)
        img_path = os.path.join(filepath, img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        if img is not None:
            test_imgs.append(img)
        else:
            print(f"Failed to load image: {img_path}")
    return np.ascontiguousarray(test_imgs).astype(np.float32)

def load_labels(filepath):
    global fileList
    test_labels = []
    labels_mapping = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for each in lines:
            imageName, labels = each.strip('\n').split(' ')
            labels_mapping[imageName] = int(labels)
    for each in fileList:
        test_labels.append(labels_mapping[each])
    return np.ascontiguousarray(test_labels).astype(np.int32)



class CustomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=32):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = training_data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)