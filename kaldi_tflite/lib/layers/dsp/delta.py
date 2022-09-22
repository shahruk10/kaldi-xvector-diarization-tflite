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


from typing import Union, Iterable, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Delta(Layer):

    """
    This layer implements delta feature extraction. The output of this layer is
    compliant with Kaldi, producing the same output has Kaldi's `add-deltas`
    binary.

    The layer expects a 3D tensor of shape (batch, frames, samples). It pads the
    input by repeating values at the edges.
    """

    def __init__(self,
                 order: int = 2,
                 window: int = 2,
                 name: str = None,
                 **kwargs):
        """
        Initializes Delta layer with given configuration.

        Parameters
        ----------
        order : int, optional
            Order of delta computation, by default, 2
        window : int, optional
            Number of frames advanced and delayed frames used in delta computation. By default, 2.
            Total window size is (2 * window + 1).
        name : str, optional
            Name of the given layer. Is auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If window or order <= 0.
        """
        super(Delta, self).__init__(trainable=False, name=name)

        self.order = order
        self.window = window

        if self.order <= 0 or self.window <= 0:
            raise ValueError("`order` and `window` must be > 0")

        # Inputs to this layers are expected to be in the shape
        # (batch, frames, feats)
        self.batchAxis = 0
        self.frameAxis = -2
        self.featAxis = -1

    def build(self, input_shape: tuple):
        """
        Creates the kernels for this layer, given the shape of the input.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, frames, samples).
        """
        super(Delta, self).build(input_shape)

        # Creating convolution kernels for computing deltas. This is ported from
        # the kaldi repo from kaldi/src/feat/feature-functions.cc
        scales = []
        scales.append(np.float32([1.0]))

        for i in range(1, self.order + 1):
            prevScale = scales[i - 1]
            prevOffset = (len(prevScale) - 1) // 2
            curOffset = prevOffset + self.window
            curScale = np.zeros(len(prevScale) + 2 * self.window, dtype=np.float32)

            normalizer = 0.0
            for j in range(-self.window, self.window + 1):
                normalizer += j**2

                for k in range(-prevOffset, prevOffset + 1):
                    curScale[j + k + curOffset] += (j * prevScale[k + prevOffset])

            curScale = np.divide(curScale, normalizer)
            scales.append(curScale)

        # Reshaping into 4-D kernels for 2D conv with height = window size,
        # width = 1, in and out channels = 1
        scales = [s.reshape(-1, 1, 1, 1) for s in scales]

        self.kernels = [tf.constant(scale, dtype=tf.float32) for scale in scales]
        self.paddings = [(scale.shape[0] - 1) // 2 for scale in scales]

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the output of this layer, given the shape of the input.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, frames, samples).

        Returns
        -------
        Tuple[Union[int, None]]
            Shape of the output of this layer.
        """
        outputShape = input_shape
        outputShape[self.featAxis] *= (self.order + 1)

        return outputShape

    def get_config(self) -> dict:
        config = super(Delta, self).get_config()

        config.update({
            "order": self.order,
            "window": self.window,
        })

        return config

    def call(self, inputs):

        inputShape = tf.shape(inputs)
        batchSize = inputShape[self.batchAxis]
        numFrames = inputShape[self.frameAxis]
        featDim = inputShape[self.featAxis]

        deltas = [tf.expand_dims(inputs, axis=2)]

        for order in range(1, self.order + 1):
            kernel = self.kernels[order]
            padding = self.paddings[order]

            # Padding conv input by padding with values at edges of original
            # input. This is a peculiarity specific to the way Kaldi computes
            # the deltas. TODO (shahruk): figure out how to do this more
            # efficiently.
            x = tf.concat([
                tf.tile(inputs[:, :1, :], [1, padding, 1]),
                inputs,
                tf.tile(inputs[:, -1:, :], [1, padding, 1]),
            ], self.frameAxis)

            # Extra "channel" dimension at the end to facilitate the 2D conv.
            # This is removed after the conv.
            x = tf.expand_dims(x, axis=-1)

            deltas.append(tf.expand_dims(tf.cast(tf.squeeze(tf.nn.conv2d(
                tf.cast(x, kernel.dtype), kernel, strides=(1, 1),
                padding="VALID", data_format="NHWC",
            ), axis=-1), x.dtype), axis=2))

        inputsWithDeltas = tf.concat(deltas, 2)
        outputShape = [batchSize, numFrames, featDim * (self.order + 1)]
        inputsWithDeltas = tf.reshape(inputsWithDeltas, outputShape)

        return inputsWithDeltas
