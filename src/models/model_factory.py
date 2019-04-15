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

"""Factory to get E3D-LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from src.models import eidetic_3d_lstm_net
import tensorflow as tf


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
  """Builds an adam optimizer."""
  updates = []
  if not isinstance(cost_or_grads, list):
    grads = tf.gradients(cost_or_grads, params)
  else:
    grads = cost_or_grads
  t = tf.Variable(1., 'adam_t')
  for p, g in zip(params, grads):
    mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
    if mom1 > 0:
      v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
      v_t = mom1 * v + (1. - mom1) * g
      v_hat = v_t / (1. - tf.pow(mom1, t))
      updates.append(v.assign(v_t))
    else:
      v_hat = g
    mg_t = mom2 * mg + (1. - mom2) * tf.square(g)
    mg_hat = mg_t / (1. - tf.pow(mom2, t))
    g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
    p_t = p - lr * g_t
    updates.append(mg.assign(mg_t))
    updates.append(p.assign(p_t))  # TODO(yunbo): check if this line is correct
  updates.append(t.assign_add(1))
  return tf.group(*updates)


class Model(object):
  """Model class for E3D-LSTM model."""

  def __init__(self, configs):
    self.configs = configs
    self.x = [
        tf.placeholder(tf.float32, [
            self.configs.batch_size, self.configs.total_length,
            self.configs.img_width // self.configs.patch_size,
            self.configs.img_width // self.configs.patch_size,
            self.configs.patch_size * self.configs.patch_size *
            self.configs.img_channel
        ]) for i in range(self.configs.n_gpu)
    ]

    self.real_input_flag = tf.placeholder(tf.float32, [
        self.configs.batch_size,
        self.configs.total_length - self.configs.input_length - 1,
        self.configs.img_width // self.configs.patch_size,
        self.configs.img_width // self.configs.patch_size,
        self.configs.patch_size * self.configs.patch_size *
        self.configs.img_channel
    ])

    grads = []
    loss_train = []
    self.pred_seq = []
    self.tf_lr = tf.placeholder(tf.float32, shape=[])
    self.itr = tf.placeholder(tf.float32, shape=[])
    self.params = dict()
    num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
    num_layers = len(num_hidden)
    for i in range(self.configs.n_gpu):
      with tf.device('/gpu:%d' % i):
        with tf.variable_scope(
            tf.get_variable_scope(), reuse=True if i > 0 else None):
          # define a model
          output_list = self.construct_model(self.x[i], self.real_input_flag,
                                             num_layers, num_hidden)

          gen_ims = output_list[0]
          loss = output_list[1]
          loss_train.append(loss / self.configs.batch_size)
          # gradients
          all_params = tf.trainable_variables()
          grads.append(tf.gradients(loss, all_params))
          self.pred_seq.append(gen_ims)

    # if self.configs.n_gpu == 1:
    #     self.train_op = tf.train.AdamOptimizer(self.configs.lr).minimize(loss)
    # else:
    # add losses and gradients together and get training updates
    with tf.device('/gpu:0'):
      for i in range(1, self.configs.n_gpu):
        loss_train[0] += loss_train[i]
        for j in range(len(grads[0])):
          grads[0][j] += grads[i][j]
    # keep track of moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.9995)
    maintain_averages_op = tf.group(ema.apply(all_params))
    self.train_op = tf.group(
        adam_updates(
            all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
        maintain_averages_op)

    self.loss_train = loss_train[0] / self.configs.n_gpu

    # session
    variables = tf.global_variables()
    self.saver = tf.train.Saver(variables)
    init = tf.global_variables_initializer()
    config_prot = tf.ConfigProto()
    config_prot.gpu_options.allow_growth = configs.allow_gpu_growth
    config_prot.allow_soft_placement = True
    self.sess = tf.Session(config=config_prot)
    self.sess.run(init)
    if self.configs.pretrained_model:
      self.saver.restore(self.sess, self.configs.pretrained_model)

  def train(self, inputs, lr, real_input_flag, itr):
    feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
    feed_dict.update({self.tf_lr: lr})
    feed_dict.update({self.itr: float(itr)})
    feed_dict.update({self.real_input_flag: real_input_flag})
    loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
    return loss

  def test(self, inputs, real_input_flag):
    feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
    feed_dict.update({self.real_input_flag: real_input_flag})
    gen_ims = self.sess.run(self.pred_seq, feed_dict)
    return gen_ims

  def save(self, itr):
    checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
    self.saver.save(self.sess, checkpoint_path, global_step=itr)
    print('saved to ' + self.configs.save_dir)

  def load(self, checkpoint_path):
    print('load model:', checkpoint_path)
    self.saver.restore(self.sess, checkpoint_path)

  def construct_model(self, images, real_input_flag, num_layers, num_hidden):
    """Contructs a model."""
    networks_map = {
        'e3d_lstm': eidetic_3d_lstm_net.rnn,
    }

    if self.configs.model_name in networks_map:
      func = networks_map[self.configs.model_name]
      return func(images, real_input_flag, num_layers, num_hidden, self.configs)
    else:
      raise ValueError('Name of network unknown %s' % self.configs.model_name)
