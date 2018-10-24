import os
import time

import numpy as np
import tensorflow as tf
from tcv3.utils import predict_to_file, eprint, get_dataset, load_dataset

datasetname = 'data_part1'
path_to_dataset = os.path.join(datasetname)

img_height = 77
img_width = 71
img_channels = 1

num_epochs = 141

log_rate = 20
write_rate = 5

high_learning_rate = 1
low_learning_rate = 0.85

num_neural_units = 1024

# model = ['gd',  'dpout', 'xv', 'lrh='+str(high_learning_rate), 'lrl='+str(low_learning_rate), 'nu='+str(num_neural_units)]
model = ['final']

train_writer = tf.summary.FileWriter(os.path.join('logdir/mlprcpt/t', *model))
valid_writer = tf.summary.FileWriter(os.path.join('logdir/mlprcpt/v', *model))

if __name__ == '__main__':
    if not os.path.exists(path_to_dataset):
        eprint(path_to_dataset, 'not found')
        get_dataset('https://maups.github.io/tcv3/%s.tar.bz2' % datasetname, os.path.join('%s.tar.bz2' % datasetname))
    dataset = load_dataset(path_to_dataset)

    classes = dataset['classes']

    x_train, y_train = dataset['train']
    x_valid, y_valid = dataset['valid']
    x_test = dataset['test'][0]

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape)

    x_train = x_train.reshape(-1, img_height * img_width * img_channels) / 255
    x_valid = x_valid.reshape(-1, img_height * img_width * img_channels) / 255
    x_test = x_test.reshape(-1, img_height * img_width * img_channels) / 255

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, img_height * img_width * img_channels), name='features')
            y = tf.placeholder(tf.int64, shape=(None,), name='labels')
            y_one_hot = tf.one_hot(y, len(classes), name='labels_one_hot')
            training_boolean = tf.placeholder(tf.bool, name='training_boolean')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            batch_size = tf.placeholder(tf.int64, name='batch_size')

        with tf.name_scope('forward'):
            fc1 = tf.layers.dense(
                x,
                num_neural_units,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='hidden_layer'
            )
            dpout = tf.layers.dropout(
                fc1,
                training=training_boolean,
                name='dropout_layer'
            )
            out = tf.layers.dense(
                dpout,
                len(classes),
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='output_layer'
            )
            loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(y_one_hot, out),
                name='loss'
            )

            tf.summary.scalar('loss_function', loss / tf.cast(batch_size, tf.float32))

        with tf.name_scope('backward'):
            # train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
            # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
            train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('metrics'):
            result = tf.argmax(out, 1)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32), name='accuracy')

            tf.summary.scalar('accuracy_metric', accuracy / tf.cast(batch_size, tf.float32))
        merged_summary = tf.summary.merge_all()


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

        if iter_idx % write_rate == 0:
            ret = session.run(
                merged_summary,
                feed_dict={
                    x: input_x,
                    y: input_y,
                    training_boolean: True,
                    batch_size: input_x.shape[0]
                }
            )
            train_writer.add_summary(ret, iter_idx)

        if iter_idx % log_rate == 0:
            eprint('epoch:', iter_idx, 'lr:', input_learning_rate, 'time: %.5f' % (time.time() - start),
                   'accuracy: %.5f' % (train_accuracy / len(input_x)), 'loss: %.5f' % (train_loss / len(input_x)))


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

        if iter_idx % write_rate == 0:
            ret = session.run(
                merged_summary,
                feed_dict={
                    x: input_x,
                    y: input_y,
                    training_boolean: False,
                    batch_size: input_x.shape[0]
                }
            )
            valid_writer.add_summary(ret, iter_idx)

        if iter_idx % log_rate == 0:
            print('evaluation:', iter_idx, 'time: %.5f' % (time.time() - start),
                  'accuracy: %.5f' % (valid_accuracy / len(input_x)), 'loss: %.5f' % (valid_loss / len(input_x)))


    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        for i in range(num_epochs):
            lr = (high_learning_rate * (num_epochs - i - 1) + low_learning_rate * i) / (num_epochs - 1)
            epoch(session, x_train, y_train, lr, i)
            evaluation(session, x_valid, y_valid, i)

        predict_to_file(session.run(result, feed_dict={x: x_test, training_boolean: False}), dataset['test'][2], 'mlprcpt_result.txt')

        # train_writer.add_graph(session.graph)
        valid_writer.add_graph(session.graph)
