import tensorflow as tf
import math
import numpy as np


def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # Replace nan values in the true and predicted images with 0
        imgs_true = tf.where(tf.math.is_nan(imgs_true), tf.zeros_like(imgs_true), imgs_true)
        imgs_pred = tf.where(tf.math.is_nan(imgs_pred), tf.zeros_like(imgs_pred), imgs_pred)

        # return (1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)) / 2

        result = 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

        # check if nan
        #if math.isnan(result):
        #    result = 0.0

        # return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))
        return result

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # Replace nan values in the true and predicted images with 0
        imgs_true = tf.where(tf.math.is_nan(imgs_true), tf.zeros_like(imgs_true), imgs_true)
        imgs_pred = tf.where(tf.math.is_nan(imgs_pred), tf.zeros_like(imgs_pred), imgs_pred)

        result = 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

         # check if nan
        #if math.isnan(result):
        #    result = 0.0

        return result

    return loss


def l2_loss(imgs_true, imgs_pred):

    # Replace nan values in the true and predicted images with 0
    imgs_true = tf.where(tf.math.is_nan(imgs_true), tf.zeros_like(imgs_true), imgs_true)
    imgs_pred = tf.where(tf.math.is_nan(imgs_pred), tf.zeros_like(imgs_pred), imgs_pred)

    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    result = tf.nn.l2_loss(imgs_true - imgs_pred)

    #if tf.is_nan(result):
    #    result = tf.constant([0.0])

    return result
