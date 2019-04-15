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

"""Data Provider."""

from src.data_provider import kth_action
from src.data_provider import mnist

datasets_map = {
    'mnist': mnist,
    'action': kth_action,
}


def data_provider(dataset_name,
                  train_data_paths,
                  valid_data_paths,
                  batch_size,
                  img_width,
                  seq_length,
                  is_training=True):
  """Returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    train_data_paths: List, [train_data_path1, train_data_path2...]
    valid_data_paths: List, [val_data_path1, val_data_path2...]
    batch_size: Int, the batch size.
    img_width: Int, the width of input images.
    seq_length: Int, the length of the input sequence.
    is_training: Bool, training or testing.

  Returns:
      if is_training is True, it returns two dataset instances for both
      training and evaluation. Otherwise only one dataset instance for
      evaluation.
  Raises:
      ValueError: When `dataset_name` is unknown.
  """
  if dataset_name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % dataset_name)
  train_data_list = train_data_paths.split(',')
  valid_data_list = valid_data_paths.split(',')
  if dataset_name == 'mnist':
    test_input_param = {
        'paths': valid_data_list,
        'minibatch_size': batch_size,
        'input_data_type': 'float32',
        'is_output_sequence': True,
        'name': dataset_name + 'test iterator'
    }
    test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
    test_input_handle.begin(do_shuffle=False)
    if is_training:
      train_input_param = {
          'paths': train_data_list,
          'minibatch_size': batch_size,
          'input_data_type': 'float32',
          'is_output_sequence': True,
          'name': dataset_name + ' train iterator'
      }
      train_input_handle = datasets_map[dataset_name].InputHandle(
          train_input_param)
      train_input_handle.begin(do_shuffle=True)
      return train_input_handle, test_input_handle
    else:
      return test_input_handle

  if dataset_name == 'action':
    input_param = {
        'paths': valid_data_list,
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': seq_length,
        'input_data_type': 'float32',
        'name': dataset_name + ' iterator'
    }
    input_handle = datasets_map[dataset_name].DataProcess(input_param)
    if is_training:
      train_input_handle = input_handle.get_train_input_handle()
      train_input_handle.begin(do_shuffle=True)
      test_input_handle = input_handle.get_test_input_handle()
      test_input_handle.begin(do_shuffle=False)
      return train_input_handle, test_input_handle
    else:
      test_input_handle = input_handle.get_test_input_handle()
      test_input_handle.begin(do_shuffle=False)
      return test_input_handle
