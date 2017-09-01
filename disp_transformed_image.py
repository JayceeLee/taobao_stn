import os
import csv
import numpy as np
import tensorflow as tf
import inception_v2

from PIL import Image
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from spatial_transformer_alpha import transformer

num_transformer = 1
NUM_THETA_PARAMS = 4

dataset_dir = '/home/deepinsight/tongzhen/data-set/taobao_stn'
eval_filename = 'taobao_eval_tf.csv'
train_filename = 'taobao_train_tf.csv'

ckpt_dir = '/home/deepinsight/tongzhen/vars/taobao_stn'
save_dir = '/home/deepinsight/tongzhen/log/taobao_stn'


# dataset_dir = '/home/tze'
# eval_filename = 'image_for_eval.csv'
# train_filename = 'image_for_train.csv'
#
# ckpt_dir = '/home/tze/Workspace/vars/stntriplet'
# save_dir = '/home/tze/Learning/tmp'

inputs_height = 224
inputs_width = 224
transformed_height = 224
transformed_width = 224

batch_size = 32
num_threads = 16
capacity = 2 * batch_size


def inputs(filename_dir, file):
    name_list = _get_filename_list(filename_dir, file)
    name_queue = tf.train.string_input_producer(name_list, num_epochs=1)

    reader = tf.WholeFileReader()
    _, content = reader.read(name_queue)

    image = tf.image.decode_jpeg(content, channels=3)
    image = image_preprocessing(image, inputs_height, inputs_width)

    image_batch = tf.train.batch([image], batch_size, num_threads, capacity)

    return image_batch


def transformer_inference(image):
    arg_scope = inception_v2.inception_v2_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=False):
            with tf.variable_scope('stn'):
                with tf.variable_scope('localization'):
                    transformer_theta = localization_net_alpha(image, num_transformer, NUM_THETA_PARAMS)
                    transformer_theta_split = tf.split(transformer_theta, num_transformer, axis=1)

                transformer_outputs = []
                transformer_output_size = [transformed_height, transformed_width]
                for theta in transformer_theta_split:
                    transformer_outputs.append(
                        transformer(image, theta, transformer_output_size, sampling_kernel='bilinear'))

    return transformer_outputs


def localization_net_alpha(image, num_transformer, num_theta_params):
    """
    Utilize inception_v2 as the localization net of spatial transformer
    """
    # outputs 7*7*1024: default final_endpoint='Mixed_5c' before full connection layer
    with tf.variable_scope('inception_net'):
        net, _ = inception_v2.inception_v2_base(image)

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


def _get_filename_list(filename_dir, file):
    name_list = []
    # label_list = []
    # id_list = []
    # attr_list = []
    with open(os.path.join(filename_dir, file), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            name_list.append(row[0])
            # label_list.append(row[1])
            # id_list.append(row[2])
            # attr_list.append(row[3])

    return name_list


def image_preprocessing(image, out_height, out_width):
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


def decode_image(file_content):
    image = file_content / 2.0
    image = image + 0.5
    image = image * 255.0
    image = np.array(image, dtype=np.uint8) 
    return Image.fromarray(image)


def main():
    images = inputs(dataset_dir, eval_filename)
    transformed_images = transformer_inference(images)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the checkpoint: {}'.format(ckpt.model_checkpoint_path))
        else:
            print('checkpoint not found')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        batch_round = 4
        try:
            for round_idx in range(batch_round):
                io_image_batch = sess.run([images, transformed_images[0]])
                for batch_idx, image_batch in enumerate(io_image_batch):
                    for image_idx, image in enumerate(image_batch):
                        decoded_image = decode_image(image)
                        image_saved_id = round_idx * batch_size + image_idx
                        if batch_idx == 0:
                            saved_image = 'image_{}_original.jpg'.format(image_saved_id)
                        else:
                            saved_image = 'image_{}_transformer_path_{}.jpg'.format(image_saved_id, batch_idx - 1)
                        decoded_image.save(os.path.join(save_dir, saved_image))
        except tf.errors.OutOfRangeError:
            print('Done image spatial transforming')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    main()



