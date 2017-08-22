import os
import csv
import tensorflow as tf
import numpy as np
import inception_v2

from PIL import Image
from tensorflow.contrib import slim
from spatial_transformer_alpha import transformer

is_training = True

filename_dir = '/home/tze'
train_filename = 'image_for_train.csv'

fine_tune_ckpt_dir = '/home/tze/Workspace/data-set/ckpt/inception_ckpt'
save_dir = '/home/tze/Workspace/vars/taobao'

localization_var_to_train_scope = ['stn/localization/logits',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5c',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5b',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5a']

classification_var_to_train_scope = ['stn/classification/logits',
                                     'stn/classification/path_0/InceptionV2/Mixed_5c',
                                     'stn/classification/path_0/InceptionV2/Mixed_5b',
                                     'stn/classification/path_0/InceptionV2/Mixed_5a']


NUM_TRANSFORMER = 1
NUM_THETA_PARAMS = 4

NUM_CLASSES = 120
NUM_ATTR = 10654

input_image_height = 224
input_image_width = 224
channels = 3
transformer_output_size = [224, 224]

num_epochs = 2
batch_size = 4

weight_decay = 2e-5
label_smoothing = 0.1
rmsprop_decay = 0.9
num_epochs_per_decay = 2
num_samples_per_epoch = 5994
init_learning_rate_for_cls = 0.01
init_learning_rate_for_loc = 0.00001
end_learning_rate = 0.0001
learning_rate_decay_factor = 0.94
dropout_keep_prob = 0.7


def inputs_loader(file_dir, file):
    # create the filename queue
    filename = os.path.join(file_dir, file)
    filename_queue = tf.train.string_input_producer([filename], num_epochs)

    # create the reader for the filename queue
    reader = tf.TextLineReader()
    key, records = reader.read(filename_queue)

    # read the csv file row by row
    record_defaults = [[''], [0]]
    image_path, image_label = tf.decode_csv(records, record_defaults)

    # decode and preprocess the image
    file_content = tf.read_file(image_path)
    image = tf.image.decode_image(file_content, channels=3)
    image = preprocessing(image)

    # transform the image_label to one hot encoding
    label = slim.one_hot_encoding(image_label, NUM_CLASSES)

    # batching images and labels
    num_threads = 4
    min_after_dequeue = 13 * batch_size
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size,
                                                      min_after_dequeue=min_after_dequeue,
                                                      capacity=capacity,
                                                      num_threads=num_threads)
    # image_batch, label_batch = tf.train.batch([image, label], batch_size,
    #                                                   capacity=capacity,
    #                                                   num_threads=num_threads)

    return image_batch, label_batch


def inputs_loader_for_taobao(file_dir, file):
    # create the filename and label example
    filename_list, label_list = _get_filename_list(file_dir, file)
    filename, label = tf.train.slice_input_producer([filename_list, label_list], num_epochs)

    # decode and preprocess the image
    file_content = tf.read_file(filename)
    image = tf.image.decode_image(file_content, channels=3)
    image = preprocessing(image)

    # transform the image_label to one hot encoding
    label = slim.one_hot_encoding(label, NUM_CLASSES)

    # batching images and labels
    num_threads = 4
    min_after_dequeue = 13 * batch_size
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=capacity, num_threads=num_threads)

    return image_batch, label_batch


# define the inference graph
def inference(inputs, dropout_keep_prob):
    """Fine grained classification with multiplex spatial transformation channels utilizing inception nets

        """
    arg_scope = inception_v2.inception_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        with tf.variable_scope('stn'):
            with tf.variable_scope('localization'):
                transformer_theta = localization_net_alpha(inputs, NUM_TRANSFORMER, NUM_THETA_PARAMS)
                transformer_theta_split = tf.split(transformer_theta, NUM_TRANSFORMER, axis=1)

            transformer_outputs = []
            for theta in transformer_theta_split:
                transformer_outputs.append(
                    transformer(inputs, theta, transformer_output_size, sampling_kernel='bilinear'))

            inception_outputs = []
            transformer_outputs_shape = [batch_size, transformer_output_size[0],
                                         transformer_output_size[1], channels]
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

    return classification_logits


def stn_cnn_with_image_output(inputs, transformer_output_size, num_classes):
    """Fine grained classification with multiplex spatial transformation channels utilizing inception nets

    """
    arg_scope = inception_v2.inception_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        with tf.variable_scope('stn'):
            with tf.variable_scope('localization'):
                transformer_theta = localization_net_beta(inputs, NUM_TRANSFORMER, NUM_THETA_PARAMS)
                transformer_theta_split = tf.split(transformer_theta, NUM_TRANSFORMER, axis=1)

            transformer_outputs = []
            for theta in transformer_theta_split:
                transformer_outputs.append(transformer(inputs, theta, transformer_output_size, sampling_kernel='bilinear'))

    return transformer_outputs


# define the loss function
def train_loss(labels, logits):
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
    #                                                                name='cross_entropy_per_example')
    # avg_cross_entropy = tf.reduce_mean(cross_entropy)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=label_smoothing)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.add_n([cross_entropy] + regularization_losses, name='total_loss')


def train_accuracy(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    return accuracy


def train_step(total_loss, global_step, localization_scope, classification_scope):
    decay_step = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
    lr_cls = tf.train.exponential_decay(init_learning_rate_for_cls, global_step, decay_step,
                                    learning_rate_decay_factor, staircase=True)
    optimizer_cls = tf.train.GradientDescentOptimizer(lr_cls)

    lr_loc = tf.train.exponential_decay(init_learning_rate_for_loc, global_step, decay_step,
                                    learning_rate_decay_factor, staircase=True)
    optimizer_loc = tf.train.GradientDescentOptimizer(lr_loc)

    # ema_loss = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # ema_loss_op = ema_loss.apply([total_loss])
    tf.summary.scalar(total_loss.op.name + '(raw)', total_loss)
    # tf.summary.scalar(total_loss.op.name, ema_loss.average(total_loss))

    classification_vars = get_vars_to_train(classification_scope)
    localization_vars = get_vars_to_train(localization_scope)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # update_ops.append(ema_loss_op)
    train_op_cls = optimizer_cls.minimize(total_loss, var_list=classification_vars)
    train_op_loc = optimizer_loc.minimize(total_loss, global_step, localization_vars)
    with tf.control_dependencies(update_ops):
        train_op = tf.group(train_op_cls, train_op_loc)

    return train_op


# initialize all the variables
def init_fn(sess, fine_tune_ckpt, save_dir, is_fine_tuning=True):
    """Initialize the variables for fine tuning and training

    """
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    if is_fine_tuning:
        # initialize the vars for fine turning
        fine_tune_var_dict_list = _get_var_to_restore()
        for var_dict in fine_tune_var_dict_list:
            saver = tf.train.Saver(var_dict)
            saver.restore(sess, fine_tune_ckpt)
        print('load the fine tune model successfully')
    else:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            print('load the ckpt file successfully')
            # sess.run(tf.train.get_global_step().assign(0))

    # sess.run(tf.variables_initializer(
    #     list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables()))
    # ))


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
        logits = slim.conv2d(net, num_transformer*num_theta_parmas, [1, 1],
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             biases_initializer=init_biase,
                             normalizer_fn=None, activation_fn=tf.nn.tanh, scope='conv2d_b_1x1')

    return tf.squeeze(logits, [1, 2])


def get_vars_to_train(train_scope):
    vars_to_train = []
    for scope in train_scope:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        vars_to_train.extend(var_list)

    return vars_to_train


def _inception_logits(inputs, num_outputs, dropout_keep_prob=1.0, activ_fn=None):
    with tf.variable_scope('logits'):
        kernel_size = inception_v2._reduced_kernel_size_for_small_input(inputs, [7, 7])
        # shape ?*1*1*?
        net = slim.avg_pool2d(inputs, kernel_size, padding='VALID')
        # drop out neuron before fc conv
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout')
        # [1, 1] fc conv
        logits = slim.conv2d(net, num_outputs, [1, 1], normalizer_fn=None, activation_fn=activ_fn, scope='conv2_a_1x1')

    return tf.squeeze(logits, [1, 2])


def _add_loss_summary(total_loss):
    pass


def _get_var_to_restore():
    """Return the list of var_dict for fine tuning

    """
    scope_prefix = ['stn/localization/inception_net/']
    for path_idx in range(NUM_TRANSFORMER):
        scope_prefix.append('stn/classification/path_{}/'.format(path_idx))

    var_dict_list = []
    for prefix in scope_prefix:
        len_prefix = len(prefix)
        var_dict = {var.op.name[len_prefix:]: var for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, prefix)}
        var_dict_list.append(var_dict)

    return var_dict_list


# def _get_var_to_train():
#     """Return the list of var object for training
#
#     """
#     scope_to_train = ['stn/localization/logits', 'stn/classification/logits']
#     var_to_train = []
#     for scope in scope_to_train:
#         var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope)
#         var_to_train.extend(var_list)
#
#     return var_to_train


# def _get_localization_var_to_fine_tune():
#     """Return the dict of (name.op.name, var) pairs for fine tuning
#
#     """
#     scope_prefix = 'stn/localization/'
#     len_prefix = len(scope_prefix)
#     var_dict = {var.op.name[len_prefix:]: var for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope_prefix)}
#
#     return var_dict
#
#
# def _get_classification_var_to_fine_tune():
#     """Return the list of var_dict of classification net with mulitplex paths
#
#     """
#     var_dict_list = []
#     for path_idx in range(NUM_TRANSFORMER):
#         scope_prefix = 'stn/classification/path_{}'.format(path_idx)
#         len_prefix = len(scope_prefix)
#         var_dict = {var.op.name[len_prefix:]: var
#                     for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope_prefix)}
#         var_dict_list.append(var_dict)
#
#     return var_dict_list


def _theta_activ_fn(theta):
    return tf.clip_by_value(theta, -1.0, 1.0)


def _get_filename_list(file_dir, file):
    filename_path = os.path.join(file_dir, file)
    filename_list = []
    cls_label_list = []
    with open(filename_path, 'r') as f:
        for line in f:
            file_info = line.strip().split(',')
            filename_list.append(file_info[0])
            cls_label_list.append(int(file_info[1]))

    return filename_list, cls_label_list


def preprocessing(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.975)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [input_image_height, input_image_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((input_image_height, input_image_width, channels))
    return image


def decode_image(file_content):
    image = file_content / 2.0
    image = image + 0.5
    image = image * 255.0
    return Image.fromarray(image.astype(np.uint8))


def main():
    # file_dir = '/home/tze'
    # file = 'image_for_train.csv'
    #
    # image_batch, label_batch = inputs_loader_for_taobao(file_dir, file)
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     try:
    #         for batch_idx in range(3):
    #             # Run training steps or whatever
    #             images, labels = sess.run([image_batch, label_batch])
    #             print(images, type(images), images.shape)
    #             print(labels, type(labels), labels.shape)
    #             # print(attrs, type(attrs), attrs.shape)
    #             # for im_idx, im in enumerate(images):
    #             #     de_im = decode_image(im)
    #             #     de_im.save('/home/tze/Tmp/image_{}.jpg'.format(batch_idx * batch_size + im_idx))
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #     finally:
    #         # When done, ask the threads to stop.
    #         coord.request_stop()
    #
    #     # Wait for threads to finish.
    #     coord.join(threads)



    # filename_path = os.path.join(filename_dir, train_filename)
    # filename_list = []
    # cls_label_list = []
    # with open(filename_path, 'r') as f:
    #     for line in f:
    #         filename, label, nid, attr  = line.split(',')
    #         filename_list.append(filename)
    #         cls_label_list.append(int(label))
    #
    # for idx in range(7):
    #     print(filename_list[idx])
    #     print(cls_label_list[idx])

    # save the new ckpt for stn classfication
    if is_training:
        dropout_keep_prob = 0.7
    else:
        dropout_keep_prob = 1.0

    inputs, labels = inputs_loader_for_taobao(filename_dir, train_filename)

    logits = inference(inputs, dropout_keep_prob)

    batch_loss = train_loss(labels, logits)
    batch_accuracy = train_accuracy(labels, logits)

    # var_train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # var_train_list = _get_var_to_train()

    global_step = slim.create_global_step()
    localization_scope = ['stn/localization']
    classification_scope = ['stn/classification']
    # localization_scope = localization_var_to_train_scope
    # classification_scope = classification_var_to_train_scope
    train_op = train_step(batch_loss, global_step, localization_scope, classification_scope)
    # summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        fine_tune_model_path = os.path.join(fine_tune_ckpt_dir, 'inception_v2.ckpt')
        init_fn(sess, fine_tune_model_path, save_dir)
        saver.save(sess, os.path.join(save_dir, 'taobao_init.ckpt'), global_step=global_step)
        print('init fine tune params saved')




    # if is_training:
    #     dropout_keep_prob = 0.7
    # else:
    #     dropout_keep_prob = 1.0
    #
    # inputs, labels = inputs_loader_for_taobao(filename_dir, train_filename)
    #
    # logits = inference(inputs, dropout_keep_prob)
    #
    # batch_loss = train_loss(labels, logits)
    # batch_accuracy = train_accuracy(labels, logits)
    #
    # # var_train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # # var_train_list = _get_var_to_train()
    #
    # global_step = slim.create_global_step()
    # # localization_scope = ['stn/localization']
    # # classification_scope = ['stn/classification']
    # localization_scope = localization_var_to_train_scope
    # classification_scope = classification_var_to_train_scope
    # train_op = train_step(batch_loss, global_step, localization_scope, classification_scope)
    # # summary_op = tf.summary.merge_all()
    #
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     fine_tune_model_path = os.path.join(fine_tune_ckpt_dir, 'inception_v2.ckpt')
    #     init_fn(sess, fine_tune_model_path, save_dir)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     try:
    #         while not coord.should_stop():
    #             # Run training steps or whatever
    #             loss, accuracy, step, _ = sess.run([batch_loss, batch_accuracy, global_step, train_op])
    #             if step % 777 == 0:
    #                 print('step: {:>5}, loss: {:.5f}'.format(step, loss))
    #                 saver.save(sess, os.path.join(save_dir, 'taobao.ckpt'), global_step=global_step)
    #                 print('save model ckpt')
    #             if step % 11 == 0:
    #                 print('step: {:>5}, loss: {:.5f}, accuracy: {:.5f}'.format(step, loss, accuracy))
    #             else:
    #                 print('step: {:>5}, loss: {:.5f}'.format(step, loss))
    #
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #     finally:
    #         # When done, ask the threads to stop.
    #         coord.request_stop()
    #
    #     # Wait for threads to finish.
    #     coord.join(threads)


if __name__ == '__main__':
    main()