#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn8_vgg_ours as fcn8_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

img1 = scp.misc.imread("./test_data/tabby_cat.png")
fake_mask = scp.misc.imread("./test_data/tabby_cat.png",'L')
msk_layer = np.expand_dims(fake_mask, axis = 2)
print(msk_layer.shape,img1.shape)
input_img = np.append(img1, msk_layer, 2)
print("shape",input_img.shape)
with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: input_img}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn8_downsampled.png', down_color)
    scp.misc.imsave('fcn8_upsampled.png', up_color)
