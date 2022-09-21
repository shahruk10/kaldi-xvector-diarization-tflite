#!/usr/bin/env python3

# Copyright (2021-) Shahruk Hossain <shahruk10@gmail.com>
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
# ==============================================================================


from typing import Iterable

import os
from pprint import pformat

import tensorflow as tf


def SavedModel2TFLite(
    savedModelPath: str,
    outPath: str,
    optimize: bool = False,
    target_dtypes: Iterable[tf.dtypes.DType] = [tf.float32],
    enable_select_tf_ops: bool = False,
    enable_resource_variables: bool = False,
):
    """
    Converts a SavedModel into a TFLite model saved as a FlatBuffer
    file. Not all models may be convertible. Please check Tensorflow
    docs for all supported model layers and operations:
        https://www.tensorflow.org/lite/convert 
    Parameters
    ----------
    savedModelPath : str
        Path to SavedModel directory.
    outPath : str
        Path to where converted TFLite model will be written
        (.tflite extension included if not provided)
    optimize : bool, optional
        Applies latency and model size optimizations, such as
        dynamic quantization of model weights, by default False.
    target_dtypes : Iterable[tf.dtypes.DType], optional
        Applicable if optimize = True. Will optimize assuming that
        the target devices will run on these data types. Optimization
        might be driven by the smallest type in this set. By default,
        set to [tf.float32].
    enable_select_tf_ops: bool, optional
        If true, will allow ops from the core tensorflow library which will
        require linking to the flex ops library when building applications
        with the TF Lite API.
    enable_resource_variables: bool, optional
        If true, enables resource variables to be converted by the converter.
    """

    print(f"TFLiteConverter: using tensorflow v{tf.__version__}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
    if enable_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        ]

    # Setting optimizations for model size and latency
    if optimize:
        print(f"Optimizing for model size and inference latency")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        print(f"*Not* doing any optimizations for model size and inference latency")
        converter.optimizations = []

    converter.target_spec.supported_types = target_dtypes
    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = enable_resource_variables

    tfliteModel = converter.convert()

    # Save the model
    outPath = os.path.abspath(outPath)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    if not outPath.endswith(".tflite"):
        outPath += ".tflite"
        print(f"Adding missing .tflite extension to output path: {outPath}")

    with open(outPath, 'wb') as f:
        f.write(tfliteModel)

    # Loading converted model and printing model details for inspection.
    ip = tf.lite.Interpreter(outPath)
    print(f"- input details for {os.path.basename(outPath)}: {pformat(ip.get_input_details())}")
    print(f"- output details for {os.path.basename(outPath)}: {pformat(ip.get_output_details())}")
    print("Conversion success")
