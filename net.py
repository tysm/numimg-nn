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
    dataset = load_dataset(path_to_dataset, resize_img=(img_width, img_height), perform_augmentation=True)

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

            # tf.summary.scalar('accuracy_metric', accuracy / tf.cast(batch_size, tf.float32))
        # merged_summary = tf.summary.merge_all()


    def epoch(session: tf.Session, input_x: np.ndarray, input_y: np.ndarray, input_learning_rate: float, iter_idx: int,
              input_batch_size: int = 32):
        batch = np.random.permutation(len(input_x))

        train_loss = 0.0
        train_accuracy = 0.0
        start = time.time()
        for i in range(0, len(input_x), input_batch_size):
            x_batch = input_x.take(batch[i:i + input_batch_size], axis=0)
            y_batch = input_y.take(batch[i:i + input_batch_size], axis=0)

            ret = session.run(
                [train_op, loss, accuracy],
                feed_dict={
                    x: x_batch,
                    y: y_batch,
                    training_boolean: True,
                    learning_rate: input_learning_rate
                }
            )
            train_loss += ret[1] * min(input_batch_size, len(input_x) - i)
            train_accuracy += ret[2]

        # if iter_idx % write_rate == 0:
        #     ret = session.run(
        #         merged_summary,
        #         feed_dict={
        #             x: input_x,
        #             y: input_y,
        #             training_boolean: True,
        #             batch_size: input_x.shape[0]
        #         }
        #     )
        #     train_writer.add_summary(ret, iter_idx)

        if iter_idx % log_rate == 0:
            eprint('epch: %03d' % iter_idx, 'time: %.5f' % (time.time() - start), 
                   'accuracy: %.5f' % (train_accuracy / len(input_x)),
                   'loss: %.5f' % (train_loss / len(input_x)),
                   'lr:', input_learning_rate)


    def evaluation(session: tf.Session, input_x: np.ndarray, input_y: np.ndarray, iter_idx: int,
                   input_batch_size: int = 32):
        valid_loss = 0.0
        valid_accuracy = 0.0
        start = time.time()
        for i in range(0, len(input_x), input_batch_size):
            ret = session.run(
                [loss, accuracy],
                feed_dict={
                    x: input_x[i:i + input_batch_size],
                    y: input_y[i:i + input_batch_size],
                    training_boolean: False
                }
            )
            valid_loss += ret[0] * min(input_batch_size, len(input_x) - i)
            valid_accuracy += ret[1]

        # if iter_idx % write_rate == 0:
        #     ret = session.run(
        #         merged_summary,
        #         feed_dict={
        #             x: input_x,
        #             y: input_y,
        #             training_boolean: False,
        #             batch_size: input_x.shape[0]
        #         }
        #     )
        #     valid_writer.add_summary(ret, iter_idx)

        if iter_idx % log_rate == 0:
            eprint('eval: %03d' % iter_idx, 'time: %.5f' % (time.time() - start),
                  'accuracy: %.5f' % (valid_accuracy / len(input_x)),
                  'loss: %.5f' % (valid_loss / len(input_x)))


    with tf.Session(graph=graph) as session:
        eprint('Inicializando variáveis...')
        session.run(tf.global_variables_initializer())


        eprint('Rodando as épocas...')
        for i in range(num_epochs):
            lr = (high_learning_rate * (num_epochs - i - 1) + low_learning_rate * i) / (num_epochs - 1)
            epoch(session, x_train, y_train, lr, i)
            evaluation(session, x_valid, y_valid, i)

        # predict_to_file(session.run(result, feed_dict={x: x_test, training_boolean: False}), dataset['test'][2], 'result.txt')

        # train_writer.add_graph(session.graph)
        # valid_writer.add_graph(session.graph)

        tf_saver = tf.train.Saver()
        tf_saver.save(session, 'models/cnn-mnist.ckpt')

