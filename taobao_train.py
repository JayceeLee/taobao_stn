import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from taobao_gamma import inputs_loader_for_taobao, inference, train_loss, train_accuracy, train_step, init_fn

filename_dir = '/home/deepinsight/tongzhen'
train_filename = 'taobao_train.csv3'
fine_tune_ckpt_dir = '/home/deepinsight/tongzhen/ckpt'
ckpt_save_dir = '/home/deepinsight/tongzhen'

finetune_localization_var_scope = ['stn/localization/logits',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5c',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5b',
                                   'stn/localization/inception_net/InceptionV2/Mixed_5a']

finetune_classification_var_scope = ['stn/classification/logits',
                                     'stn/classification/path_0/InceptionV2/Mixed_5c',
                                     'stn/classification/path_0/InceptionV2/Mixed_5b',
                                     'stn/classification/path_0/InceptionV2/Mixed_5a',
                                     'stn/classification/path_1/InceptionV2/Mixed_5c',
                                     'stn/classification/path_1/InceptionV2/Mixed_5b',
                                     'stn/classification/path_1/InceptionV2/Mixed_5a']

is_fine_tuning = True
dropout_keep_prob = 0.7


def train():

    inputs, labels = inputs_loader_for_taobao(filename_dir, train_filename)

    logits = inference(inputs, dropout_keep_prob)

    batch_loss = train_loss(labels, logits)
    batch_accuracy = train_accuracy(labels, logits)

    global_step = slim.create_global_step()

    if is_fine_tuning:
        localization_scope = finetune_localization_var_scope
        classification_scope = finetune_classification_var_scope
    else:
        localization_scope = ['stn/localization']
        classification_scope = ['stn/classification']

    train_op = train_step(batch_loss, global_step, localization_scope, classification_scope)
    # summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        fine_tune_model_path = os.path.join(fine_tune_ckpt_dir, 'inception_v2.ckpt')
        init_fn(sess, fine_tune_model_path, ckpt_save_dir, is_fine_tuning=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                loss, accuracy, step, _ = sess.run([batch_loss, batch_accuracy, global_step, train_op])
                if step % 777 == 0:
                    print('step: {:>5}, loss: {:.5f}'.format(step, loss))
                    saver.save(sess, os.path.join(ckpt_save_dir, 'taobao.ckpt'), global_step=global_step)
                    print('model ckpt saved')
                if step % 11 == 0:
                    print('step: {:>5}, loss: {:.5f}, accuracy: {:.5f}'.format(step, loss, accuracy))
                else:
                    print('step: {:>5}, loss: {:.5f}'.format(step, loss))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)