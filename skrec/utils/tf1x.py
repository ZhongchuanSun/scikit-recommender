__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["inner_product", "bpr_loss", "l2_loss",
           "euclidean_distance", "l2_distance",
           "hinge_loss"]

import tensorflow as tf


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def bpr_loss(y_pos, y_neg, name="bpr_loss"):
    """ bpr loss
    """
    with tf.name_scope(name):
        return -tf.log_sigmoid(y_pos - y_neg)


def l2_loss(*params):
    """
        tf.nn.l2_loss = sum(t ** 2) / 2
    """
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def euclidean_distance(a, b, dim: int=-1):
    return tf.norm(a - b, ord='euclidean', axis=dim)


def l2_distance(a, b, dim: int=-1):
    return euclidean_distance(a, b, dim)


def hinge_loss(yij, margin=1.0):
    return tf.nn.relu(margin - yij)
