import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v2

from tensorflow.contrib.layers.python.layers import layers as layers_lib
from spatial_transformer_alpha import transformer

NUM_THETA_PARAMS = 4
INPUT_IMAGE_SIZE = 224


def get_network_fn(num_classes, num_transformer, transformer_output_height=224, transformer_output_width=224,
                   batch_size=32, dropout_keep_prob=0.7, weight_decay=0.0, is_training=True):
    def network_fn(inputs):
        """Fine grained classification with multiplex spatial transformation channels utilizing inception nets

                """
        end_points = {}
        arg_scope = inception_v2.inception_v2_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
                with tf.variable_scope('stn'):
                    with tf.variable_scope('localization'):
                        transformer_theta = localization_net_alpha(inputs, num_transformer, NUM_THETA_PARAMS)
                        transformer_theta_split = tf.split(transformer_theta, num_transformer, axis=1)
                        end_points['stn/localization/transformer_theta'] = transformer_theta

                    transformer_outputs = []
                    transformer_output_size = [transformer_output_height, transformer_output_width]
                    for theta in transformer_theta_split:
                        transformer_outputs.append(
                            transformer(inputs, theta, transformer_output_size, sampling_kernel='bilinear'))

                    inception_outputs = []
                    transformer_outputs_shape = [batch_size, transformer_output_size[0],
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
                        classification_logits = _inception_logits(multipath_outputs, num_classes, dropout_keep_prob)
                        end_points['stn/classification/logits'] = classification_logits

        return classification_logits, end_points

    network_fn.default_image_size = INPUT_IMAGE_SIZE

    return network_fn


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
        init_biase = tf.constant_initializer([2.0, .0, 2.0, .0] * num_transformer)
        logits = slim.conv2d(net, num_transformer * num_theta_params, [1, 1],
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             biases_initializer=init_biase,
                             normalizer_fn=None, activation_fn=tf.nn.tanh, scope='conv2d_c_1x1')

        return tf.squeeze(logits, [1, 2])


def localization_net_beta(inputs, num_transformer, num_theta_parmas):
    with tf.variable_scope('inception_net'):
        net, _ = inception_v2.inception_v2_base(inputs)
    with tf.variable_scope('logits'):
        with tf.variable_scope('branch_0'):
            branch0 = slim.conv2d(net, 128, [1, 1], scope='conv2d_a_1x1')
            branch0 = slim.conv2d(branch0, 144, [3, 3], stride=2, scope='conv2d_b_3x3')
        with tf.variable_scope('branch_1'):
            branch1 = slim.conv2d(net, 144, [1, 1], scope='conv2d_a_1x1')
            branch1 = slim.max_pool2d(branch1, [3, 3], stride=2, padding='SAME', scope='max_pool_b_3x3')
        net = tf.concat([branch0, branch1], axis=-1)

        kernel_size = inception_v2._reduced_kernel_size_for_small_input(net, [7, 7])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='avg_pool_a_{}x{}'.format(*kernel_size))
        init_biase = tf.constant_initializer([2.0, .0, 2.0, .0] * num_transformer)
        logits = slim.conv2d(net, num_transformer * num_theta_parmas, [1, 1],
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             biases_initializer=init_biase,
                             normalizer_fn=None, activation_fn=tf.nn.tanh, scope='conv2d_b_1x1')

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
