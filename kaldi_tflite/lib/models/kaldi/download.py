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



import os
import tarfile
import gzip
import urllib.request
import hashlib
from tqdm import tqdm


def downloadModel(link: str, outPath: str, sha256: str = None):
    """
    Downloads kaldi model files from the given link. Only links to
    the kaldi website (https://kaldi-asr.org/models) are allowed.

    Parameters
    ----------
    link : str
        Download link for the model. 
    outPath : str
        Path to output directory where the contents of the downloaded
        tarball will be extracted.
    sha256 : str, optional
        SHA-256 hash of the tarball. If provided, will check the hash
        of downloaded file against it. By default None.

    Raises
    ------
    ValueError
        If download link does not point to kaldi website.
        If hash of downloaded file does not match expected.
    IOError
        If download file size exceeds 50 MB. 
    """
    kaldiURL = "https://kaldi-asr.org/models"
    if not link.startswith(kaldiURL):
        raise ValueError(f"invalid dowload link, only models from {kaldiURL} allowed")

    if os.path.exists(outPath):
        print(f"download output path '{outPath}' is not empty, not overwriting")
        return

    maxBytes = 50 * 1024 * 1024  # 50 MB
    tarPath = f"{outPath}.tar"
    bytesRead = 0

    # Downloading and decompressing.
    pbar = tqdm(total=maxBytes, unit="bytes")
    pbar.set_description(f"downloading {link} ")

    with open(tarPath, 'wb') as tarFile:
        with urllib.request.urlopen(link) as handle:
            with gzip.GzipFile(fileobj=handle) as uncompressed:
                while bytesRead <= maxBytes:
                    data = uncompressed.read(4096)

                    if len(data) == 0:
                        pbar.update(maxBytes)
                        pbar.close()
                        break

                    bytesRead += len(data)
                    if bytesRead > maxBytes:
                        raise IOError(f"max download size ({maxBytes} bytes) exceeded")

                    pbar.update(len(data))
                    tarFile.write(data)

    # Checking hash of downloaded tarball.
    if sha256 is not None:
        with open(tarPath, 'rb') as tarFile:
            gotHash = hashlib.sha256(tarFile.read()).hexdigest()

        if sha256 != gotHash:
            os.remove(tarPath)
            raise ValueError(
                "hash of downloaded tarball does not match expected: {gotHash} vs {sha256}"
            )

    # Unzipping tarball.
    with tarfile.open(tarPath) as tarFile:
        contents = [tarinfo for tarinfo in tarFile.getmembers()]
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tarFile, os.path.dirname(outPath), members=contents)
