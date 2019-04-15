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

"""Builds an E3D RNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.layers.rnn_cell import Eidetic3DLSTMCell as eidetic_lstm
import tensorflow as tf


def rnn(images, real_input_flag, num_layers, num_hidden, configs):
  """Builds a RNN according to the config."""
  gen_images, lstm_layer, cell, hidden, c_history = [], [], [], [], []
  shape = images.get_shape().as_list()
  batch_size = shape[0]
  # seq_length = shape[1]
  ims_width = shape[2]
  ims_height = shape[3]
  output_channels = shape[-1]
  # filter_size = configs.filter_size
  total_length = configs.total_length
  input_length = configs.input_length

  window_length = 2
  window_stride = 1

  for i in range(num_layers):
    if i == 0:
      num_hidden_in = output_channels
    else:
      num_hidden_in = num_hidden[i - 1]
    new_lstm = eidetic_lstm(
        name='e3d' + str(i),
        input_shape=[ims_width, window_length, ims_height, num_hidden_in],
        output_channels=num_hidden[i],
        kernel_shape=[2, 5, 5])
    lstm_layer.append(new_lstm)
    zero_state = tf.zeros(
        [batch_size, window_length, ims_width, ims_height, num_hidden[i]])
    cell.append(zero_state)
    hidden.append(zero_state)
    c_history.append(None)

  memory = zero_state

  with tf.variable_scope('generator'):
    input_list = []
    reuse = False
    for time_step in range(window_length - 1):
      input_list.append(
          tf.zeros([batch_size, ims_width, ims_height, output_channels]))

    for time_step in range(total_length - 1):
      with tf.variable_scope('e3d-lstm', reuse=reuse):
        if time_step < input_length:
          input_frm = images[:, time_step]
        else:
          time_diff = time_step - input_length
          input_frm = real_input_flag[:, time_diff] * images[:, time_step] \
                      + (1 - real_input_flag[:, time_diff]) * x_gen  # pylint: disable=used-before-assignment
        input_list.append(input_frm)

        if time_step % (window_length - window_stride) == 0:
          input_frm = tf.stack(input_list[time_step:])
          input_frm = tf.transpose(input_frm, [1, 0, 2, 3, 4])

          for i in range(num_layers):
            if time_step == 0:
              c_history[i] = cell[i]
            else:
              c_history[i] = tf.concat([c_history[i], cell[i]], 1)
            if i == 0:
              inputs = input_frm
            else:
              inputs = hidden[i - 1]
            hidden[i], cell[i], memory = lstm_layer[i](
                inputs, hidden[i], cell[i], memory, c_history[i])

        x_gen = tf.layers.conv3d(hidden[num_layers - 1], output_channels,
                                 [window_length, 1, 1], [window_length, 1, 1],
                                 'same')
        x_gen = tf.squeeze(x_gen)
        gen_images.append(x_gen)
        reuse = True

  gen_images = tf.stack(gen_images)
  gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
  loss = tf.nn.l2_loss(gen_images - images[:, 1:])
  loss += tf.reduce_sum(tf.abs(gen_images - images[:, 1:]))

  out_len = total_length - input_length
  out_ims = gen_images[:, -out_len:]

  return [out_ims, loss]
