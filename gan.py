import time

import numpy as np
import tensorflow as tf

from utils import generator, discriminator, eprint, get_dataset, load_dataset

num_epochs = 21
log_rate = 1

num_features = 64
img_height, img_width, img_channels = (32, 32, 1)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, (None, num_features))
    y = tf.placeholder(tf.float32, (None, img_height, img_width, img_channels))

    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)


    norm = tf.layers.batch_normalization(x, 1, training=is_training)

    gimgs = generator(x, is_training)

    df_labels = discriminator(gimgs, is_training)
    dt_labels = discriminator(y, is_training, reuse=True)

    gvars = [v for v in tf.global_variables() if v.name.startswith('GEN')]
    dvars = [v for v in tf.global_variables() if v.name.startswith('DIS')]

    f_loss = tf.losses.log_loss(tf.zeros_like(df_labels), df_labels)
    t_loss = tf.losses.log_loss(tf.ones_like(dt_labels), dt_labels)

    gloss = tf.reduce_mean(tf.losses.log_loss(tf.ones_like(df_labels), df_labels))
    dloss = tf.reduce_mean(0.5 * (f_loss + t_loss))

    gtrain_op = tf.train.RMSPropOptimizer(learning_rate).minimize(gloss, var_list=gvars)
    dtrain_op = tf.train.RMSPropOptimizer(learning_rate).minimize(dloss, var_list=dvars)


def train(sess, ix, iy, lr, epoch, bs=16):
    batch = np.random.permutation(len(ix))
    _gloss, _dloss = 0.0, 0.0
    start = time.time()
    for i in range(0, len(ix), bs):
        bx = ix.take(batch[i:i + bs], axis=0)
        by = iy.take(batch[i:i + bs], axis=0)

        ret = sess.run(
            [dtrain_op, dloss],
            feed_dict={
                x: bx,
                y: by,
                is_training: True,
                learning_rate: lr,
            }
        )
        _dloss += ret[1] * min(bs, len(ix) - i)

        ret = sess.run(
            [gtrain_op, gloss],
            feed_dict={
                x: bx,
                y: by,
                is_training: True,
                learning_rate: lr,
            }
        )
        _gloss += ret[1] * min(bs, len(ix) - i)

    if epoch % log_rate == 0:
        print(
            'epoch: %05d' % epoch,
            'gloss: %07.3f' % _gloss,
            'dloss: %07.3f' % _dloss,
            'time: %07.3f' % (time.time() - start)
        )

dataset = load_dataset('data_part1', resize_img=(img_height, img_width))
datay = dataset['train'][0].reshape((-1, img_height, img_width, img_channels))
datax = np.random.randn(len(datay), 64)

eprint(datax.shape, datay.shape)

# tfwriter = tf.summary.FileWriter('logdir/gan', graph)

with tf.Session(graph=graph) as session:
    print('Inicializando variáveis...:', end=' ')
    session.run(tf.global_variables_initializer())
    print('Done')

    for epoch in range(num_epochs):
        train(session, datax, datay, 0.00015, epoch)
