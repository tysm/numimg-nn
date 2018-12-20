import os
import sys
import time
import tarfile
from urllib.request import urlretrieve
from typing import Tuple

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


# ML functions.

def generator(noise, is_training, reuse=False):
    with tf.variable_scope('GEN', reuse=reuse):
        fc = tf.layers.dense(
            noise,
            4*4*512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        dp = tf.layers.dropout(fc, rate=0.2, training=is_training)

        x = tf.reshape(dp, (-1, 4, 4, 512))
        norm = tf.layers.batch_normalization(x, training=is_training)

        conv = tf.layers.conv2d_transpose(
            norm,
            256,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.relu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)

        conv = tf.layers.conv2d_transpose(
            norm,
            128,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.relu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)

        imgs = tf.layers.conv2d_transpose(
            norm,
            1,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.sigmoid
        )
    return imgs


def discriminator(x, is_training, reuse=False):
    with tf.variable_scope('DIS', reuse=reuse):
        conv = tf.layers.conv2d(
            x,
            64,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.relu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)

        conv = tf.layers.conv2d(
            norm,
            128,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.relu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)

        conv = tf.layers.conv2d(
            norm,
            256,
            5,
            strides=2,
            padding='same',
            activation=tf.nn.relu
        )
        norm = tf.layers.batch_normalization(conv, training=is_training)

        flat = tf.reshape(norm, (-1, 4*4*256))
        out = tf.layers.dense(
            flat,
            1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
    return out


# Help functions.

def eprint(*args, **kwargs):
    """
    Print to stderr.

    :param args: args to print.
    :param kwargs: kwargs to print.
    :return:
    """
    print('INFO:', *args, file=sys.stderr, **kwargs)


def get_dataset(url: str, filename: str):
    """
    Download and extract the archive from url.

    :param url: archive url.
    :param filename: tar.bz2 archive filename.
    :return:
    """
    if not os.path.exists(filename):
        eprint(filename, 'not found')
        eprint('downloading', filename)
        start = time.time()
        filename, headers = urlretrieve(url, filename)
        eprint('download completed - %f.5' % (time.time() - start))

    eprint('extracting', filename)
    start = time.time()
    with tarfile.open(filename, 'r:bz2') as tarbz2file:
        tarbz2file.extractall()
    eprint('extraction completed - %f.5' % (time.time() - start))


def load_dataset(path_to_dataset: str, cv_flag: str=f'IMREAD_GRAYSCALE', x_dtype: np.dtype=np.float32,
                 y_dtype: np.dtype=np.int64, training_percentage: float=1, seed: int=7,
                 resize_img: Tuple[int, int]=None) -> dict:
    """
    Load a dataset in the format:
        "
            path_to_dataset/
                train/
                    label0/
                        img0
                        ...
                    ...
                test/
                    img0
                    ...
        "

    :param path_to_dataset: the dataset path.
    :param cv_flag: specifies how images should be read.
    :param x_dtype: specifies wich type x values should be.
    :param y_dtype: specifies wich type y values should be.
    :param training_percentage: percentage for the train_set.
    :param seed: a shuffle seed.
    :param resize_img: resize (img_width, img_height)
    :param perform_augmentation: if True, performs augmentation on training set.
    :return: the dataset.
    """
    x, y = [], []
    path_to_train = os.path.join(path_to_dataset, 'train')
    classes = sorted(os.listdir(path_to_train))
    for label in classes:
        acc = []
        path_to_label = os.path.join(path_to_train, label)
        for imgname in sorted(os.listdir(path_to_label)):
            path_to_img = os.path.join(path_to_label, imgname)
            img = cv.imread(path_to_img, getattr(cv, cv_flag))

            if resize_img is not None:
                img = cv.resize(img, resize_img)

            acc.append(img)

        x.extend(acc)
        y.extend([int(label)] * len(acc))

    # x, y = shuffle(np.array(x, dtype=x_dtype), np.array(y, dtype=y_dtype), random_state=seed)
    x, y = shuffle(x, y, random_state=seed)

    x_train, y_train = x[:int(len(x) * training_percentage)], y[:int(len(y) * training_percentage)]
    x_valid, y_valid = x[int(len(x) * training_percentage):], y[int(len(y) * training_percentage):]

    x_train, y_train = np.array(x_train, dtype=x_dtype), np.array(y_train, dtype=y_dtype)
    x_valid, y_valid = np.array(x_valid, dtype=x_dtype), np.array(y_valid, dtype=y_dtype)

    x_test = []
    imgnames = []
    path_to_test = os.path.join(path_to_dataset, 'test')
    for imgname in sorted(os.listdir(path_to_test)):
        path_to_img = os.path.join(path_to_test, imgname)
        img = cv.imread(path_to_img, getattr(cv, cv_flag))

        if resize_img is not None:
            img = cv.resize(img, resize_img)

        x_test.append(img)
        imgnames.append(imgname)
    x_test = np.array(x_test, dtype=x_dtype)

    dataset = {
        'classes': classes,
        'train': (x_train, y_train),
        'valid': (x_valid, y_valid),
        'test': (x_test, path_to_test, imgnames)
    }
    return dataset
