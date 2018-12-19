import tensorflow as tf
import numpy as np

graph = tf.Graph()

with graph.as_default():
    X = np.random.randn(3, 1)
    print(X)
    Y = np.random.randn(3)
    print(Y)
    Xl = tf.random.shuffle(X, 11)
    Yl = tf.random.shuffle(Y, 11)

with tf.Session(graph=graph) as sess:
    print(sess.run([Xl, Yl]))
