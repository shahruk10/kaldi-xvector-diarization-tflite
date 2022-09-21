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


import unittest
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

from kaldi_tflite.lib.models import SavedModel2TFLite
from kaldi_tflite.lib.layers import Framing, MFCC, Delta
from kaldi_tflite.lib.kaldi_numpy import PadWaveform
from kaldi_tflite.lib.testdata import RefDelta

tolerance = 5e-7


class TestDeltaLayer(unittest.TestCase):

    def compareFeats(self, want, got, expectEmpty=False, toleranceAdj=0):

        if expectEmpty:
            self.assertTrue(
                want.size == 0 and got.size == 0,
                f"expected empty arrays, got.shape={got.shape}, want.shape={want.shape}",
            )
            return

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )

        self.assertTrue(
            want.shape == got.shape,
            f"reference feature shape ({want.shape}) does not match shape got ({got.shape})",
        )

        rmse = np.sqrt(np.mean(np.power(want - got, 2)))
        self.assertTrue(
            rmse < tolerance + toleranceAdj,
            f"features does not match reference, rmse={rmse}",
        )

    def defaultCfg(self):
        return {
            "delta": {
                "order": 2,
                "window": 2,
            }
        }

    def checkTFLiteInference(
        self, interpreter: tf.lite.Interpreter, x: np.ndarray,
            wantFrames: int, wantDim: int,
    ):
        inputLayerIdx = interpreter.get_input_details()[0]['index']
        outputLayerIdx = interpreter.get_output_details()[0]['index']

        # Setting input size.
        interpreter.resize_tensor_input(inputLayerIdx, x.shape)

        interpreter.allocate_tensors()
        interpreter.set_tensor(inputLayerIdx, x)
        interpreter.invoke()
        y = interpreter.get_tensor(outputLayerIdx)

        gotFrames = y.shape[1]
        self.assertTrue(
            gotFrames == wantFrames,
            f"output number of frames ({gotFrames}) does not match expected ({wantFrames})",
        )

        gotDim = y.shape[-1]
        self.assertTrue(
            gotDim == wantDim,
            f"output feature dimension ({gotDim}) does not match expected ({wantDim})",
        )

    def test_ConvertTFLite(self):

        tests = {
            "default": {
                "input_shape": [None, 13],
            },
            "order=5": {
                "input_shape": [298, 13],
                "delta": {"order": 5},
            },
            "window=1": {
                "input_shape": [217, 13],
                "delta": {"window": 1},
            },
            "window=6": {
                "input_shape": [128, 13],
                "delta": {"window": 6},
            },
        }

        for name, overrides in tests.items():
            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["delta"].update(overrides.get("delta", {}))

                # Creating model with Delta layer.
                mdl = Sequential([
                    Input(overrides["input_shape"]),
                    Delta(**cfg["delta"]),
                ])

                # Saving model and converting to TF Lite.
                with TemporaryDirectory() as mdlPath, \
                        NamedTemporaryFile(suffix='.tflite') as tflitePath:
                    mdl.save(mdlPath)
                    SavedModel2TFLite(mdlPath, tflitePath.name)

                    # Testing if inference works.
                    interpreter = tf.lite.Interpreter(model_path=tflitePath.name)

                    inputShape = [1] + overrides["input_shape"]
                    if inputShape[1] is None:
                        inputShape[1] = 1

                    x = np.random.random(inputShape).astype(np.float32)
                    wantFrames = x.shape[1]
                    wantDim = x.shape[-1] * (cfg["delta"]["order"] + 1)

                    self.checkTFLiteInference(interpreter, x, wantFrames, wantDim)

    def test_Delta(self):

        # Default config overrides
        testNames = [f"16000_001_{i:03d}" for i in range(1, 10)]

        for name in testNames:
            overrides = RefDelta.getConfig(name)
            feat = RefDelta.getInputs(name)
            want = RefDelta.getOutputs(name)

            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["delta"].update(overrides["delta"])

                # Creating Delta layer and evaluating output.
                delta = Delta(**cfg["delta"])
                got = delta(feat).numpy()

                self.compareFeats(want, got)


if __name__ == "__main__":
    unittest.main()
