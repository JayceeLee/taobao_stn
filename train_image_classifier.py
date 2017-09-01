# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import model_deploy
import inception_v2

from tensorflow.contrib import slim
from spatial_transformer_alpha import transformer

NUM_TRANSFORMER = 1
NUM_THETA_PARAMS = 4
NUM_CLASSES = 120

LOC_VAR_SCOPE = 'stn/localization'
CLS_VAR_SCOPE = 'stn/classification'

loc_train_vars_scope = ['stn/localization/logits',
                        'stn/localization/inception_net/InceptionV2/Mixed_5c',
                        'stn/localization/inception_net/InceptionV2/Mixed_5b',
                        'stn/localization/inception_net/InceptionV2/Mixed_5a']

cls_train_vars_scope = ['stn/classification/logits',
                        'stn/classification/path_0/InceptionV2/Mixed_5c',
                        'stn/classification/path_0/InceptionV2/Mixed_5b',
                        'stn/classification/path_0/InceptionV2/Mixed_5a']

checkpoint_exclude_scopes = ['stn/localization/logits', 'stn/classification/logits']


dropout_keep_prob = 0.5
transformer_output_size = [224, 224]
default_image_size = 224

num_epochs = 50
learning_rate_cls = 0.01
learning_rate_loc = 0.00001

filename_dir = '/home/deepinsight/tongzhen/data-set/taobao_stn'
file = 'taobao_train_tf.csv'
save_dir = '/home/deepinsight/tongzhen/vars/taobao_stn'
ckpt_dir = '/home/deepinsight/tongzhen/ckpt/init'


tf.app.flags.DEFINE_boolean('train_partial_layers', True,
                            'Fine tune the last few layers.')


# the flags defined by the original code
######################################################################
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', save_dir,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 4,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 3600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 3600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00002, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# changed
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

# tf.app.flags.DEFINE_string(
#     'dataset_name', 'imagenet', 'The name of the dataset to load.')
#
# tf.app.flags.DEFINE_string(
#     'dataset_split_name', 'train', 'The name of the train/test split.')
#
# tf.app.flags.DEFINE_string(
#     'dataset_dir', None, 'The directory where the dataset files are stored.')
#
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', ckpt_dir,
    'The path to a checkpoint from which to fine-tune.')

# tf.app.flags.DEFINE_string(
#     'checkpoint_exclude_scopes', None,
#     'Comma-separated list of scopes of variables to exclude when restoring '
#     'from a checkpoint.')

# tf.app.flags.DEFINE_string(
#     'trainable_scopes', None,
#     'Comma-separated list of scopes to filter the set of variables to train.'
#     'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate_cls(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(learning_rate_cls,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(learning_rate_cls, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(learning_rate_cls,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)



def _configure_learning_rate_loc(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(learning_rate_loc,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(learning_rate_loc, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(learning_rate_loc,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = [scope.strip()for scope in checkpoint_exclude_scopes]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


# def _get_variables_to_train():
#   """Returns a list of variables to train.
#
#   Returns:
#     A list of variables to train by the optimizer.
#   """
#   if FLAGS.trainable_scopes is None:
#     return tf.trainable_variables()
#   else:
#     scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
#
#   variables_to_train = []
#   for scope in scopes:
#     variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#     variables_to_train.extend(variables)
#   return variables_to_train


def _get_localization_vars_to_train(train_scope):
    if FLAGS.train_partial_layers:
        vars_to_train = []
        for scope in train_scope:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            vars_to_train.extend(var_list)
        return vars_to_train
    else:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, LOC_VAR_SCOPE)


def _get_classification_vars_to_train(train_scope):
    if FLAGS.train_partial_layers:
        vars_to_train = []
        for scope in train_scope:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            vars_to_train.extend(var_list)
        return vars_to_train
    else:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, CLS_VAR_SCOPE)


def main(_):
  # if not FLAGS.dataset_dir:
  #   raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    # dataset = dataset_factory.get_dataset(
    #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    # network_fn = nets_factory.get_network_fn(
    #     FLAGS.model_name,
    #     num_classes=(dataset.num_classes - FLAGS.labels_offset),
    #     weight_decay=FLAGS.weight_decay,
    #     is_training=True)

    def localization_net_alpha(inputs, num_transformer, num_theta_params):
        """
        Utilize inception_v2 as the localization net of spatial transformer
        """
        # outputs 7*7*1024: default final_endpoint='Mixed_5c' before full connection layer
        with tf.variable_scope('inception_net'):
            net, _ = inception_v2.inception_v2_base(inputs)

        # fc layer using [1, 1] convolution kernel: 1*1*1024
        with tf.variable_scope('logits'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv2d_a_1x1')
            kernel_size = inception_v2._reduced_kernel_size_for_small_input(net, [7, 7])
            net = slim.conv2d(net, 128, kernel_size, padding='VALID', scope='conv2d_b_{}x{}'.format(*kernel_size))
            init_biase = tf.constant_initializer([1.1, .0, 1.1, .0] * num_transformer)
            logits = slim.conv2d(net, num_transformer * num_theta_params, [1, 1],
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                 biases_initializer=init_biase,
                                 normalizer_fn=None, activation_fn=tf.nn.tanh, scope='conv2d_c_1x1')

            return tf.squeeze(logits, [1, 2])

    def _inception_logits(inputs, num_outputs, dropout_keep_prob, activ_fn=None):
        with tf.variable_scope('logits'):
            kernel_size = inception_v2._reduced_kernel_size_for_small_input(inputs, [7, 7])
            # shape ?*1*1*?
            net = slim.avg_pool2d(inputs, kernel_size, padding='VALID')
            # drop out neuron before fc conv
            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout')
            # [1, 1] fc conv
            logits = slim.conv2d(net, num_outputs, [1, 1], normalizer_fn=None, activation_fn=activ_fn,
                                 scope='conv2_a_1x1')

        return tf.squeeze(logits, [1, 2])

    def network_fn(inputs):
        """Fine grained classification with multiplex spatial transformation channels utilizing inception nets

                """
        end_points = {}
        arg_scope = inception_v2.inception_v2_arg_scope(weight_decay=FLAGS.weight_decay)
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('stn'):
                with tf.variable_scope('localization'):
                    transformer_theta = localization_net_alpha(inputs, NUM_TRANSFORMER, NUM_THETA_PARAMS)
                    transformer_theta_split = tf.split(transformer_theta, NUM_TRANSFORMER, axis=1)
                    end_points['stn/localization/transformer_theta'] = transformer_theta

                transformer_outputs = []
                for theta in transformer_theta_split:
                    transformer_outputs.append(
                        transformer(inputs, theta, transformer_output_size, sampling_kernel='bilinear'))

                inception_outputs = []
                transformer_outputs_shape = [FLAGS.batch_size, transformer_output_size[0],
                                             transformer_output_size[1], 3]
                with tf.variable_scope('classification'):
                    for path_idx, inception_inputs in enumerate(transformer_outputs):
                        with tf.variable_scope('path_{}'.format(path_idx)):
                            inception_inputs.set_shape(transformer_outputs_shape)
                            net, _ = inception_v2.inception_v2_base(inception_inputs)
                            inception_outputs.append(net)
                    # concatenate the endpoints: num_batch*7*7*(num_transformer*1024)
                    multipath_outputs = tf.concat(inception_outputs, axis=-1)

                    # final fc layer logits
                    classification_logits = _inception_logits(multipath_outputs, NUM_CLASSES, dropout_keep_prob)
                    end_points['stn/classification/logits'] = classification_logits

        return classification_logits, end_points


    #####################################
    # Select the preprocessing function #
    #####################################
    # preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    # image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    #     preprocessing_name,
    #     is_training=True)

    def image_preprocessing_fn(image, out_height, out_width):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.central_crop(image, central_fraction=0.975)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [out_height, out_width], align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.image.random_flip_left_right(image)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image.set_shape((out_height, out_width, 3))
        return image

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    def _get_filename_list(file_dir, file):
        filename_path = os.path.join(file_dir, file)
        filename_list = []
        cls_label_list = []
        with open(filename_path, 'r') as f:
            for line in f:
                filename, label, nid, attr = line.strip().split(',')
                filename_list.append(filename)
                cls_label_list.append(int(label))

        return filename_list, cls_label_list

    with tf.device(deploy_config.inputs_device()):
      # create the filename and label example
      filename_list, label_list = _get_filename_list(filename_dir, file)
      num_samples = len(filename_list)
      filename, label = tf.train.slice_input_producer([filename_list, label_list], num_epochs)

      # decode and preprocess the image
      file_content = tf.read_file(filename)
      image = tf.image.decode_jpeg(file_content, channels=3)

      train_image_size = FLAGS.train_image_size or default_image_size
      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, NUM_CLASSES - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      with tf.device(deploy_config.inputs_device()):
        images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        tf.losses.softmax_cross_entropy(
            logits=end_points['AuxLogits'], onehot_labels=labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
      tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate_loc = _configure_learning_rate_loc(num_samples, global_step)
      learning_rate_cls = _configure_learning_rate_cls(num_samples, global_step)
      optimizer_loc = _configure_optimizer(learning_rate_loc)
      optimizer_cls = _configure_optimizer(learning_rate_cls)
      summaries.add(tf.summary.scalar('learning_rate_loc', learning_rate_loc))
      summaries.add(tf.summary.scalar('learning_rate_cls', learning_rate_cls))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer_loc = tf.train.SyncReplicasOptimizer(
          opt=optimizer_loc,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
      optimizer_cls = tf.train.SyncReplicasOptimizer(
          opt=optimizer_cls,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    # variables_to_train = _get_variables_to_train()

    loc_vars_to_train = _get_localization_vars_to_train(loc_train_vars_scope)
    cls_vars_to_train = _get_classification_vars_to_train(cls_train_vars_scope)

    #  and returns a train_tensor and summary_op
    _, clones_gradients_loc = model_deploy.optimize_clones(
        clones,
        optimizer_loc,
        var_list=loc_vars_to_train)
    total_loss, clones_gradients_cls = model_deploy.optimize_clones(
        clones,
        optimizer_cls,
        var_list=cls_vars_to_train)

    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates_loc = optimizer_loc.apply_gradients(clones_gradients_loc)
    grad_updates_cls = optimizer_cls.apply_gradients(clones_gradients_cls, global_step=global_step)
    update_ops.append(grad_updates_loc)
    update_ops.append(grad_updates_cls)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=None)


if __name__ == '__main__':
  tf.app.run()
