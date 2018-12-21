import os
import sys
import time
import zipfile
from urllib.request import urlretrieve

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


# ML functions.

def lrelu(x, alpha=0.2):
    return tf.maximum(x, tf.multiply(x, alpha))


def generator(noise, dp_rate, is_training, reuse=False):
    iprint('GEN')
    print(noise.shape)
    with tf.variable_scope('GEN', reuse=reuse):
        noise = tf.layers.dense(noise, 4*4*128, activation=lrelu)
        norm = tf.layers.batch_normalization(noise, training=is_training)
        dp = tf.layers.dropout(norm, dp_rate, training=is_training)
        print(dp.shape)

        noise = tf.reshape(dp, (-1, 4, 4, 128))
        print(noise.shape)

        conv = tf.layers.conv2d_transpose(
            noise,
            64,
            5,
            strides=2,
            padding='same',
            activation=lrelu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)
        dp = tf.layers.dropout(norm, dp_rate, training=is_training)
        print(dp.shape)

        conv = tf.layers.conv2d_transpose(
            norm,
            32,
            5,
            strides=2,
            padding='same',
            activation=lrelu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)
        dp = tf.layers.dropout(norm, dp_rate, training=is_training)
        print(dp.shape)

        conv = tf.layers.conv2d_transpose(
            norm,
            16,
            5,
            strides=2,
            padding='same',
            activation=lrelu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)
        dp = tf.layers.dropout(norm, dp_rate, training=is_training)
        print(dp.shape)

        imgs = tf.layers.conv2d_transpose(
            norm,
            1,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.sigmoid
        )
        print(imgs.shape)
    print()
    return imgs


def discriminator(x, dp_rate, is_training, reuse=False):
    iprint('DIS')
    print(x.shape)
    with tf.variable_scope('DIS', reuse=reuse):
        conv = tf.layers.conv2d(
            x,
            16,
            5,
            strides=2,
            padding='same',
            activation=lrelu
        )
        dp = tf.layers.dropout(conv, dp_rate, training=is_training)
        print(dp.shape)

        conv = tf.layers.conv2d(
            dp,
            32,
            5,
            padding='same',
            activation=lrelu
        )
        dp = tf.layers.dropout(conv, dp_rate, training=is_training)
        print(dp.shape)

        conv = tf.layers.conv2d(
            dp,
            64,
            5,
            padding='same',
            activation=lrelu
        )
        dp = tf.layers.dropout(conv, dp_rate, training=is_training)
        print(dp.shape)

        conv = tf.layers.conv2d(
            dp,
            128,
            5,
            padding='same',
            activation=lrelu
        )
        dp = tf.layers.dropout(conv, dp_rate, training=is_training)
        print(dp.shape)

        flat = tf.layers.flatten(dp)
        fc = tf.layers.dense(flat, 4*4*128, activation=lrelu)
        print(fc.shape)

        out = tf.layers.dense(fc, 1, activation=tf.sigmoid)
        print(out.shape)
    print()
    return out


# Help functions.

def iprint(*args, **kwargs):
    """Print INFO

    :param *args: *args.
    :param **kwargs: **kwargs.
    :return:
    """

    print('INFO:', *args, **kwargs)


def get_dataset(url, filename):
    """Download and extract the archive from @url

    :param url: zipped dataset url.
    :param filename: zipped dataset filename.
    :return:
    """

    if not os.path.exists(filename):
        iprint(filename, 'not found')
        iprint('downloading', filename)
        start = time.time()
        filename, _ = urlretrieve(url, filename)
        iprint('download completed - %f.5' % (time.time() - start))

    iprint('extracting', filename)
    start = time.time()
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()
    iprint('extraction completed - %f.5' % (time.time() - start))


def load_dataset(path_to_dataset, x_dtype=np.float32, y_dtype=np.int64,
                 cv_flag=f'IMREAD_GRAYSCALE', resize_img=None,
                 training_percentage=0.8, seed=None):
    """Load a MNIST dataset

    The dataset should be in the format:
    ---
    path_to_dataset/
        labeled/
            label0/
                img0
                ...
            ...
        unlabeled/
            img0
            ...
    ---
    Where the labels are numbers.

    :param path_to_dataset: the dataset path.
    :param x_dtype: specifies wich type x values should be.
    :param y_dtype: specifies wich type y values should be.
    :param cv_flag: specifies how images should be read.
    :param resize_img: image final size (img_width, img_height).
    :param training_percentage: percentage for the train_set.
    :param seed: a shuffle seed.
    :return: the dataset.
    """

    # Read labeled data.
    x, y = [], []
    path_to_labeled = os.path.join(path_to_dataset, 'labeled')
    classes = sorted(os.listdir(path_to_labeled))
    for label in classes:
        acc = []
        path_to_label = os.path.join(path_to_labeled, label)
        for imgname in sorted(os.listdir(path_to_label)):
            path_to_img = os.path.join(path_to_label, imgname)
            img = cv.imread(path_to_img, getattr(cv, cv_flag))

            if resize_img is not None:
                img = cv.resize(img, resize_img)

            acc.append(img)

        x.extend(acc)
        y.extend([int(label)] * len(acc))

    # Shuffle both in the same way.
    x, y = shuffle(x, y, random_state=seed)

    # Split the labeled data into train and valid.
    x_train = x[:int(len(x) * training_percentage)]
    x_valid = x[int(len(x) * training_percentage):]

    y_train = y[:int(len(y) * training_percentage)]
    y_valid = y[int(len(y) * training_percentage):]

    # Convert into numpy array.
    x_train = np.array(x_train, dtype=x_dtype)
    x_valid = np.array(x_valid, dtype=x_dtype)

    y_train = np.array(y_train, dtype=y_dtype)
    y_valid = np.array(y_valid, dtype=y_dtype)


    # Read unlabeled data.
    x_test = []
    imgnames = []
    path_to_unlabeled = os.path.join(path_to_dataset, 'unlabeled')
    for imgname in sorted(os.listdir(path_to_unlabeled)):
        path_to_img = os.path.join(path_to_unlabeled, imgname)
        img = cv.imread(path_to_img, getattr(cv, cv_flag))

        if resize_img is not None:
            img = cv.resize(img, resize_img)

        x_test.append(img)
        imgnames.append(imgname)

    # Convert into numpy array.
    x_test = np.array(x_test, dtype=x_dtype)

    # Final dataset format.
    dataset = {
        'classes': classes,
        'train': (x_train, y_train),
        'valid': (x_valid, y_valid),
        'test': (x_test, path_to_unlabeled, imgnames)
    }
    return dataset

