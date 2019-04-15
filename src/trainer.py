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

"""Functions to train and evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os.path
import cv2
import numpy as np
from skimage.measure import compare_ssim
from src.utils import preprocess


def batch_psnr(gen_frames, gt_frames):
  """Computes PSNR for a batch of data."""
  if gen_frames.ndim == 3:
    axis = (1, 2)
  elif gen_frames.ndim == 4:
    axis = (1, 2, 3)
  x = np.int32(gen_frames)
  y = np.int32(gt_frames)
  num_pixels = float(np.size(gen_frames[0]))
  mse = np.sum((x - y)**2, axis=axis, dtype=np.float32) / num_pixels
  psnr = 20 * np.log10(255) - 10 * np.log10(mse)
  return np.mean(psnr)


def train(model, ims, real_input_flag, configs, itr):
  """Trains a model."""
  ims_list = np.split(ims, configs.n_gpu)
  cost = model.train(ims_list, configs.lr, real_input_flag, itr)

  if configs.reverse_input:
    ims_rev = np.split(ims[:, ::-1], configs.n_gpu)
    cost += model.train(ims_rev, configs.lr, real_input_flag, itr)
    cost = cost / 2

  if itr % configs.display_interval == 0:
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'itr: ' + str(itr))
    print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, save_name):
  """Evaluates a model."""
  print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
  test_input_handle.begin(do_shuffle=False)
  res_path = os.path.join(configs.gen_frm_dir, str(save_name))
  os.mkdir(res_path)
  avg_mse = 0
  batch_id = 0
  img_mse, ssim, psnr = [], [], []
  output_length = configs.total_length - configs.input_length

  for i in range(output_length):
    img_mse.append(0)
    ssim.append(0)
    psnr.append(0)

  real_input_flag_zero = np.zeros((configs.batch_size, output_length - 1,
                                   configs.img_width // configs.patch_size,
                                   configs.img_width // configs.patch_size,
                                   configs.patch_size**2 * configs.img_channel))

  while not test_input_handle.no_batch_left():
    batch_id = batch_id + 1
    test_ims = test_input_handle.get_batch()
    test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
    test_dat = np.split(test_dat, configs.n_gpu)
    img_gen = model.test(test_dat, real_input_flag_zero)

    # Concat outputs of different gpus along batch
    img_gen = np.concatenate(img_gen)
    img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
    img_out = img_gen[:, -output_length:]
    target_out = test_ims[:, -output_length:]
    # MSE per frame
    for i in range(output_length):
      x = target_out[:, i]
      gx = img_out[:, i]
      gx = np.maximum(gx, 0)
      gx = np.minimum(gx, 1)
      mse = np.square(x - gx).sum()
      img_mse[i] += mse
      avg_mse += mse
      # for b in range(configs.batch_size):
      #     ssim[i] += compare_ssim(x[b], gx[b], multichannel=True)
      x = np.uint8(x * 255)
      gx = np.uint8(gx * 255)
      psnr[i] += batch_psnr(gx, x)

    # save prediction examples
    if batch_id <= configs.num_save_samples:
      path = os.path.join(res_path, str(batch_id))
      os.mkdir(path)
      for i in range(configs.total_length):
        if (i + 1) < 10:
          name = 'gt0' + str(i + 1) + '.png'
        else:
          name = 'gt' + str(i + 1) + '.png'
        file_name = os.path.join(path, name)
        img_gt = np.uint8(test_ims[0, i] * 255)
        cv2.imwrite(file_name, img_gt)
      for i in range(output_length):
        if (i + configs.input_length + 1) < 10:
          name = 'pd0' + str(i + configs.input_length + 1) + '.png'
        else:
          name = 'pd' + str(i + configs.input_length + 1) + '.png'
        file_name = os.path.join(path, name)
        img_pd = img_gen[0, i]
        img_pd = np.maximum(img_pd, 0)
        img_pd = np.minimum(img_pd, 1)
        img_pd = np.uint8(img_pd * 255)
        cv2.imwrite(file_name, img_pd)
    test_input_handle.next()

  avg_mse = avg_mse / (batch_id * configs.batch_size * configs.n_gpu)
  print('mse per seq: ' + str(avg_mse))
  for i in range(output_length):
    print(img_mse[i] / (batch_id * configs.batch_size * configs.n_gpu))

  psnr = np.asarray(psnr, dtype=np.float32) / batch_id
  print('psnr per frame: ' + str(np.mean(psnr)))
  for i in range(output_length):
    print(psnr[i])

  # ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
  # print('ssim per frame: ' + str(np.mean(ssim)))
  # for i in range(output_length):
  #     print(ssim[i])
