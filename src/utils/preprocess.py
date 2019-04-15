# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility class for reshaping tensors."""
import numpy as np


def reshape_patch(img_tensor, patch_size):
  """Reshape a 5D image tensor to a 5D patch tensor."""
  assert 5 == img_tensor.ndim
  batch_size = np.shape(img_tensor)[0]
  seq_length = np.shape(img_tensor)[1]
  img_height = np.shape(img_tensor)[2]
  img_width = np.shape(img_tensor)[3]
  num_channels = np.shape(img_tensor)[4]
  a = np.reshape(img_tensor, [
      batch_size, seq_length, img_height // patch_size, patch_size,
      img_width // patch_size, patch_size, num_channels
  ])
  b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
  patch_tensor = np.reshape(b, [
      batch_size, seq_length, img_height // patch_size, img_width // patch_size,
      patch_size * patch_size * num_channels
  ])
  return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
  """Reshape a 5D patch tensor back to a 5D image tensor."""
  assert 5 == patch_tensor.ndim
  batch_size = np.shape(patch_tensor)[0]
  seq_length = np.shape(patch_tensor)[1]
  patch_height = np.shape(patch_tensor)[2]
  patch_width = np.shape(patch_tensor)[3]
  channels = np.shape(patch_tensor)[4]
  img_channels = channels // (patch_size * patch_size)
  a = np.reshape(patch_tensor, [
      batch_size, seq_length, patch_height, patch_width, patch_size, patch_size,
      img_channels
  ])
  b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
  img_tensor = np.reshape(b, [
      batch_size, seq_length, patch_height * patch_size,
      patch_width * patch_size, img_channels
  ])
  return img_tensor
