import time

import numpy as np
import tensorflow as tf

from utils import generator, discriminator, eprint, get_dataset, load_dataset

num_epochs = 60000
log_rate = 1

num_features = 64
img_height, img_width, img_channels = (32, 32, 1)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, (None, num_features))
    y = tf.placeholder(tf.float32, (None, img_height, img_width, img_channels))

    dp_rate = tf.placeholder(tf.float32)

    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)


    norm = tf.layers.batch_normalization(x, 1, training=is_training)

    gimgs = generator(x, dp_rate, is_training)

    summary = tf.summary.image('generated', gimgs, max_outputs=16)

    df_labels = discriminator(gimgs, dp_rate, is_training)
    dt_labels = discriminator(y, dp_rate, is_training, reuse=True)

    gvars = [v for v in tf.global_variables() if v.name.startswith('GEN')]
    dvars = [v for v in tf.global_variables() if v.name.startswith('DIS')]

    f_loss = tf.losses.log_loss(tf.zeros_like(df_labels), df_labels)
    t_loss = tf.losses.log_loss(tf.ones_like(dt_labels), dt_labels)

    gloss = tf.reduce_mean(tf.losses.log_loss(tf.ones_like(df_labels), df_labels))
    dloss = tf.reduce_mean(0.5 * (f_loss + t_loss))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gtrain_op = tf.train.RMSPropOptimizer(learning_rate).minimize(gloss, var_list=gvars)
        dtrain_op = tf.train.RMSPropOptimizer(learning_rate).minimize(dloss, var_list=dvars)

tfwriter = tf.summary.FileWriter('logdir/gan', graph)


def train(sess, ix, iy, lr, epoch, bs=16):
    batch = np.random.permutation(len(ix))
    _gloss, _dloss = 0.0, 0.0
    gtimes, dtimes = 0, 0
    start = time.time()
    for i in range(0, len(ix), bs):
        bx = ix.take(batch[i:i + bs], axis=0)
        by = iy.take(batch[i:i + bs], axis=0)

        feed_dict = {
            x: bx,
            y: by,
            dp_rate: 0.4,
            is_training: True,
            learning_rate: lr,
        }

        f_ls, t_ls, g_ls, d_ls = sess.run([f_loss, t_loss, gloss, dloss], feed_dict=feed_dict)

        train_g = g_ls * 1.5 >= d_ls
        train_d = d_ls * 2 >= g_ls

#        print('g %07.3f' % g_ls, 'd %07.3f' % d_ls)

        if train_d:
#            print('%05d' % epoch, 'I\'ll train discriminator for now.')
            ret = sess.run([dtrain_op, dloss], feed_dict=feed_dict)
            _dloss += ret[1] * bx.shape[0]
            dtimes += bx.shape[0]

        if train_g:
#            print('%05d' % epoch, 'I\'ll train generator for now.')
            ret = sess.run([gtrain_op, gloss], feed_dict=feed_dict)
            _gloss += ret[1] * bx.shape[0]
            gtimes += bx.shape[0]

        tfwriter.add_summary(sess.run(summary, feed_dict={x: bx, dp_rate: 0.4, is_training: False}))

    if gtimes != 0:
        _gloss /= gtimes
    if dtimes != 0:
        _dloss /= dtimes

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


with tf.Session(graph=graph) as session:
    tfsaver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
    # tfsaver.restore(sess, 'models/gan.ckpt')

    print('Inicializando variáveis...:', end=' ')
    session.run(tf.global_variables_initializer())
    print('Done')


    for epoch in range(num_epochs):
        train(session, datax, datay, 0.00015, epoch)

    tfsaver.save(session, 'models/gan.ckpt')
