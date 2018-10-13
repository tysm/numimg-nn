import os
import time

import numpy as np
import tensorflow as tf
from tcv3.utils import neuron_step, eprint, get_dataset, load_dataset

datasetname = 'data_part1'
path_to_dataset = os.path.join(datasetname)

img_height = 77
img_width = 71
img_channels = 1

num_epochs = 50
high_learning_rate = 0.05
low_learning_rate = 0.0001

if __name__ == '__main__':
    if not os.path.exists(path_to_dataset):
        eprint(path_to_dataset, 'not found')
        get_dataset('https://maups.github.io/tcv3/%s.tar.bz2' % datasetname, os.path.join('%s.tar.bz2' % datasetname))
    dataset = load_dataset(path_to_dataset)

    classes = dataset['classes']

    x_train, y_train = dataset['train']
    x_valid, y_valid = dataset['valid']

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    x_train = x_train.reshape(-1, img_height * img_width * img_channels) / 255
    x_valid = x_valid.reshape(-1, img_height * img_width * img_channels) / 255

    eprint(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, img_height * img_width * img_channels), name='features')
            y = tf.placeholder(tf.int64, shape=(None,), name='labels')
            y_one_hot = tf.one_hot(y, len(classes), name='labels_one_hot')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        with tf.name_scope('forward'):
            z = neuron_step(x, img_height * img_width * img_channels, len(classes))
            #loss = tf.reduce_mean(tf.reduce_sum((y_one_hot - a) ** 2))
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=z,
                    labels=y_one_hot
                ),
                name='loss'
            )

        with tf.name_scope('backward'):
            train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('metrics'):
            result = tf.argmax(z, 1)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))


    def epoch(session: tf.Session, input_x: np.ndarray, input_y: np.ndarray, input_learning_rate: float, iter_idx: int,
              batch_size: int = 32):
        batch = np.random.permutation(len(input_x))

        train_loss = 0.0
        train_accuracy = 0.0
        start = time.time()
        for i in range(0, len(input_x), batch_size):
            x_batch = input_x.take(batch[i:i + batch_size], axis=0)
            y_batch = input_y.take(batch[i:i + batch_size], axis=0)

            ret = session.run(
                [train_op, loss, accuracy],
                feed_dict={
                    x: x_batch,
                    y: y_batch,
                    learning_rate: input_learning_rate
                }
            )
            train_loss += ret[1] * min(batch_size, len(input_x) - i)
            train_accuracy += ret[2]

        print('epoch:', iter_idx, 'lr:', input_learning_rate, 'time: %f.5' % (time.time() - start), 'accuracy:',
              train_accuracy / len(input_x), 'loss:', train_loss / len(input_x))


    def evaluation(session: tf.Session, input_x: np.ndarray, input_y: np.ndarray, iter_idx: int, batch_size: int = 32):
        valid_loss = 0.0
        valid_accuracy = 0.0
        start = time.time()
        for i in range(0, len(input_x), batch_size):
            ret = session.run(
                [loss, accuracy],
                feed_dict={
                    x: input_x[i:i + batch_size],
                    y: input_y[i:i + batch_size]
                }
            )
            valid_loss += ret[0] * min(batch_size, len(input_x) - i)
            valid_accuracy += ret[1]

        print('evaluation:', iter_idx, 'time: %f.5' % (time.time() - start), 'accuracy:', valid_accuracy / len(input_x),
              'loss:', valid_loss / len(input_x))


    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        for i in range(num_epochs):
            lr = (high_learning_rate*(num_epochs-i-1)+low_learning_rate*i)/(num_epochs-1)
            epoch(session, x_train, y_train, lr, i)
            evaluation(session, x_valid, y_valid, i)
