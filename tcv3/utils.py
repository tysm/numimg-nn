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

def neuron_step(x: tf.Tensor, channels_in: int, channels_out: int, name: str = 'neuron') -> tf.Tensor:
    """
    Build a neuron and return a linear operation on x.

    :param x: input.
    :param channels_in: features dimension.
    :param channels_out:  output dimension
    :param name: scope name.
    :return: linear operation on x.
    """
    with tf.name_scope(name):
        # w = tf.Variable(
        #     # tf.random_uniform((channels_in, channels_out), maxval=0.001),
        #     tf.truncated_normal((channels_in, channels_out), stddev=0.1),
        #     name='weight'
        # )
        w = tf.get_variable(
            'weight',
            shape=(channels_in, channels_out),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.Variable(
            tf.zeros((channels_out,)),
            # tf.constant(0.1, shape=(channels_out,)),
            name='bias'
        )
        a = tf.sigmoid(tf.matmul(x, w) + b)
        # tf.summary.histogram('weights', w)
        # tf.summary.histogram('biases', b)
        # tf.summary.histogram('activation', a)
        return a


def predict_to_file(results: list, paths_to_imgs: list, filename):
    with open(filename, 'w') as out_file:
        for i in range(len(results)):
            out_file.write('{} {}\n'.format(paths_to_imgs[i], results[i]))


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
                 y_dtype: np.dtype=np.int64, training_percentage: float=0.8, seed: int=7,
                 resize_img: Tuple[int, int]=None, perform_augmentation: bool=False) -> dict:
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

    if perform_augmentation:
        rows, cols = x_train[0].shape[:2]

        M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 0, 1.10)
        scaling_imgs = - len(x_train)
        for i in range(len(x_train)):
            img = cv.warpAffine(x_train[i], M, (cols, rows))
            if not is_valid_img(img):
                continue
            x_train.append(img)
            y_train.append(y_train[i])
        scaling_imgs += len(x_train)
        eprint('scaling_imgs={}'.format(scaling_imgs))

        clock_M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), -5, 1)
        anti_M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 5, 1)
        rotated_imgs = - len(x_train)
        for i in range(len(x_train)):
            img = cv.warpAffine(x_train[i], clock_M, (cols, rows))
            if is_valid_img(img):
                x_train.append(img)
                y_train.append(y_train[i])

            img = cv.warpAffine(x_train[i], anti_M, (cols, rows))
            if not is_valid_img(img):
                continue
            x_train.append(img)
            y_train.append(y_train[i])
        rotated_imgs += len(x_train)
        eprint('rotated_imgs={}'.format(rotated_imgs))

        M = np.float32([[1, 0, 0], [0, 1, 4]])
        translated_imgs = - len(x_train)
        for i in range(len(x_train)):
            img = cv.warpAffine(x_train[i], M, (cols, rows))
            if not is_valid_img(img):
                continue
            x_train.append(img)
            y_train.append(y_train[i])
        translated_imgs += len(x_train)
        eprint('translated_imgs={}'.format(translated_imgs))

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

def is_valid_img(img: np.ndarray):
    rows, cols = img.shape[:2]
    if any(i > 127.0 for i in img[0]):
        return False
    if any(i > 127.0 for i in img[-1]):
        return False
    if any(i > 127.0 for i in img[:,0]):
        return False
    if any(i > 127.0 for i in img[:,-1]):
        return False
    if any(i > 127.0 for i in img.reshape(-1)):
        return True
    return False
