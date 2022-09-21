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



from .plda.plda import PLDA

from .tdnn.tdnn import TDNN
from .tdnn.utils import reshapeKaldiTdnnWeights

from .normalization.batchnorm import BatchNorm
from .normalization.cmvn import CMVN

from .stats.stats_pooling import StatsPooling

from .dsp.framing import Framing
from .dsp.windowing import Windowing
from .dsp.filterbank import FilterBank
from .dsp.dct import DCT
from .dsp.mfcc import MFCC
from .dsp.vad import VAD
from .dsp.delta import Delta
