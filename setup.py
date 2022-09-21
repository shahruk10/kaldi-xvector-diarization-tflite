##!/usr/bin/env python3

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

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

version = {}
with open("kaldi_tflite/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='kaldi_tflite',
    version=version["__version__"],
    author='Shahruk Hossain',
    author_email='shahruk10@gmail.com',
    url='https://github.com/shahruk10/kaldi-tflite',
    description='kaldi-tflite: Kaldi Features and Models implemented in Tensorflow compatible with Tensorflow Lite.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(include=["kaldi_tflite", "kaldi_tflite.*"]),
    python_requires='>=3.6.0',
    setup_requires=[
        'pytest-runner',
        'pylint',
    ],
    tests_require=[
        'pytest>=6.2.5',
        'pytest-cov>=3.0.0',
        'librosa>=0.8.1',
    ],
    install_requires=[
        'tensorflow>=2.8.0',
        'pyyaml>=6.0',
        'tqdm>=4.62.3',
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
