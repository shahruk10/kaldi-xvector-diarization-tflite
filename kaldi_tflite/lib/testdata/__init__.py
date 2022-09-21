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


from .testdata import KaldiTestDataReader

from .plda.plda_model import RefPldaModel
from .plda.plda_scores import RefPldaScores

from .xvectors.xvectors import RefXVectors

from .tdnn.tdnn_narrow import RefTdnnNarrow
from .tdnn.tdnn_single_layer import RefTdnnSingleLayer
from .tdnn.tdnn_models import RefKaldiXVectorModels

from .ivector_extractor.ivector_extractor import RefIvectorExtractor

from .stats.stats_pooling import RefStatsPooling

from .feats.feats import RefMFCC, RefFbank, RefCMVN, RefVAD, RefDelta

from .models.xvectors_e2e import RefXVectorsE2E
