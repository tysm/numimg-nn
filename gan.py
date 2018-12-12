import tensorflow as tf


num_features = 64

# batch_size, height, width, channels
x_shape = [-1, 8, 8, 1]
y_shape = (None, 64, 64, 1)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, (None, num_features))
    y = tf.placeholder(tf.float32, y_shape)
    training = tf.placeholder(tf.bool)

    norm = tf.layers.batch_normalization(x, 1, training=training)

    x = tf.reshape(x, x_shape)

    with tf.name_scope('GEN'):
        gconv = tf.layers.conv2d_transpose(x, 32, 2, strides=2, padding='same', activation=tf.nn.relu)
        gconv = tf.layers.conv2d_transpose(gconv, 16, 4, strides=2, padding='same', activation=tf.nn.relu)
        gconv = tf.layers.conv2d_transpose(gconv, 8, 8, strides=2, padding='same', activation=tf.nn.relu)
        gconv = tf.layers.conv2d_transpose(gconv, 4, 16, padding='same', activation=tf.nn.relu)
        gimg = tf.layers.conv2d_transpose(gconv, 1, 32, padding='same', activation=tf.nn.relu)

    
