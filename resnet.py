import os
import time

import numpy as np
import tensorflow as tf
from tcv3.utils import predict_to_file, eprint, get_dataset, load_dataset

datasetname = 'data_part1'
path_to_dataset = os.path.join(datasetname)

img_height = 64
img_width = 64
img_channels = 1

num_epochs = 12

log_rate = 1
# write_rate = 5

high_learning_rate = 0.5
low_learning_rate = 0.4

# num_neural_units = 1024
num_neural_units = 512

# model = ['gd',  'dpout', 'xv', 'lrh='+str(high_learning_rate), 'lrl='+str(low_learning_rate), 'nu='+str(num_neural_units)]
# model = ['final']

# train_writer = tf.summary.FileWriter(os.path.join('logdir/conv/t', *model))
# valid_writer = tf.summary.FileWriter(os.path.join('logdir/conv/v', *model))

if __name__ == '__main__':
    if not os.path.exists(path_to_dataset):
        eprint(path_to_dataset, 'not found')
        get_dataset('https://maups.github.io/tcv3/%s.tar.bz2' % datasetname, os.path.join('%s.tar.bz2' % datasetname))
    dataset = load_dataset(path_to_dataset, resize_img=(img_width, img_height), perform_augmentation=False)

    classes = dataset['classes']

    x_train, y_train = dataset['train']
    x_valid, y_valid = dataset['valid']
    x_test = dataset['test'][0]

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape)

    # x_train = x_train.reshape(-1, img_height * img_width * img_channels) / 255
    x_train = x_train.reshape(-1, img_height, img_width, img_channels) / 255
    # x_valid = x_valid.reshape(-1, img_height * img_width * img_channels) / 255
    x_valid = x_valid.reshape(-1, img_height, img_width, img_channels) / 255
    # x_test = x_test.reshape(-1, img_height * img_width * img_channels) / 255
    x_test = x_test.reshape(-1, img_height, img_width, img_channels) / 255

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, img_channels), name='features')

            y = tf.placeholder(tf.int64, shape=(None,), name='labels')
            y_one_hot = tf.one_hot(y, len(classes), name='labels_one_hot')

            training_boolean = tf.placeholder(tf.bool, name='training_boolean')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            batch_size = tf.placeholder(tf.int64, name='batch_size')

        # with tf.name_scope('resize'):
        #     pool = tf.layers.max_pooling2d(
        #         x,
        #         pool_size=(2, 2),
        #         strides=(2, 2)
        #     )
        #     eprint('pool:', pool.shape)

        with tf.name_scope('pre_hidden'):
            conv1 = tf.layers.conv2d(
                x,
                32,
                (8, 8),
                strides=(2, 2),
                padding='same',
                activation=tf.nn.relu
            )
            eprint('conv1:', conv1.shape)
            pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2))
            eprint('pool1:', pool1.shape)
            # pool_dpout1 = tf.layers.dropout(pool1, training=training_boolean)
            # eprint('pool_dpout1:', pool_dpout1.shape)

            conv2 = tf.layers.conv2d(
                pool1,
                64,
                (16, 16),
                strides=(2, 2),
                padding='same',
                activation=tf.nn.relu,
            )
            eprint('conv2:', conv2.shape)
            pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))
            eprint('pool2:', pool2.shape)
            # pool_dpout2 = tf.layers.dropout(pool2, rate=0.2, training=training_boolean)
            # eprint('pool_dpout1:', pool_dpout2.shape)

            proc_x = tf.layers.Flatten()(pool2)
            eprint('preproc_x:', proc_x.shape)

        with tf.name_scope('forward'):
            fc1 = tf.layers.dense(
                proc_x,
                num_neural_units,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='hidden'
            )
            dpout = tf.layers.dropout(fc1, rate=0.7, training=training_boolean)
            out = tf.layers.dense(
                dpout,
                len(classes),
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='out'
            )
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_one_hot, out))

            # tf.summary.scalar('loss_function', loss / tf.cast(batch_size, tf.float32))

        with tf.name_scope('backward'):
            # train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
            # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
            train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('metrics'):
            result = tf.argmax(out, 1)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32), name='accuracy')

    with tf.Session(graph=graph) as session:
        eprint('Inicializando variáveis...')
        session.run(tf.global_variables_initializer())

        tf_saver = tf.train.Saver()
        tf_saver.restore(session, 'models/cnn-mnist.ckpt')
        print(session.run([result], feed_dict={x: x_test[0:1], training_boolean: False}))
