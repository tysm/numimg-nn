import random

import tensorflow as tf


num_features = 64
img_height, img_width, img_channels = (32, 32, 1)
random_seed = random.randint(1, 101)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, (None, num_features))
    y = tf.placeholder(tf.float32, (None, img_height, img_width, img_channels))

    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    batch_size = tf.placeholder(tf.int64)

    norm = tf.layers.batch_normalization(x, 1, training=is_training)

    with tf.variable_scope('GEN'):
        gfc = tf.layers.dense(norm, 4*4*512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        gx = tf.reshape(gfc, (-1, 4, 4, 512))
        gnorm = tf.layers.batch_normalization(gx, 1, training=is_training)

        gconv = tf.layers.conv2d_transpose(gnorm, 256, 5, strides=2, padding='same', activation=tf.nn.relu)
        gnorm = tf.layers.batch_normalization(gconv, 1, training=is_training)

        gconv = tf.layers.conv2d_transpose(gnorm, 128, 5, strides=2, padding='same', activation=tf.nn.relu)
        gnorm = tf.layers.batch_normalization(gconv, 1, training=is_training)

        gimg = tf.layers.conv2d_transpose(gnorm, 1, 5, strides=2, padding='same', activation=tf.nn.sigmoid)

    false_labels = tf.zeros((batch_size))
    true_labels = tf.ones((batch_size))
    labels = tf.concat([false_labels, true_labels], axis=0)
    labels = tf.random.shuffle(labels, random_seed)

    imgs = tf.concat([gimg, y], axis=0)
    imgs = tf.random.shuffle(imgs, random_seed)

    with tf.variable_scope('DIS'):
        dconv = tf.layers.conv2d(imgs, 64, 5, strides=2, padding='same', activation=tf.nn.relu)
        dnorm = tf.layers.batch_normalization(dconv, 1, training=is_training)

        dconv = tf.layers.conv2d(dnorm, 128, 5, strides=2, padding='same', activation=tf.nn.relu)
        dnorm = tf.layers.batch_normalization(dconv, 1, training=is_training)

        dconv = tf.layers.conv2d(dnorm, 256, 5, strides=2, padding='same', activation=tf.nn.relu)
        dnorm = tf.layers.batch_normalization(dconv, 1, training=is_training)

        flat = tf.reshape(tf.nn.relu(dnorm), (-1, 4*4*256))
        out = tf.layers.dense(flat, 1, activation=tf.sigmoid)


    gen_vars = [v for v in tf.global_variables() if v.name.startswith('GEN')]
    dis_vars = [v for v in tf.global_variables() if v.name.startswith('DIS')]

    loss = tf.reduce_mean(tf.reduce_sum((out-labels)**2))

    gen_train_op = tf.train.AdamOptimizer(learning_rate).minimize(-loss, var_list=gen_vars)
    dis_train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=dis_vars)
