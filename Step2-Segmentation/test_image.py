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
#img1 = scp.misc.imread("ECSSD/images/0005.jpg")
#fake_mask = scp.misc.imread("ECSSD/ground_truth_mask/0005.png",'L')
name = "dance-twirl/00003"
savename = "dance-twirl-00003"

img1 = scp.misc.imread("DAVIS/images/"+name+".jpg")
fake_mask = scp.misc.imread("DAVIS/box_mask/"+name+".png",'L')
#fake_mask = scp.misc.imread("DAVIS/ground_truth_mask/boat/00039.png",'L')

msk_layer = np.expand_dims(fake_mask, axis = 2)
#print(msk_layer.shape,img1.shape)
input_img = np.append(img1, msk_layer, 2)
#print("shape",input_img.shape)
with tf.Session() as sess:
    images = tf.placeholder("float")
    
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    ckpt_dir = "./checkpoints"

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    
    start = 0
    if ckpt and ckpt.model_checkpoint_path:
        start = int(ckpt.model_checkpoint_path.split("-")[1])
        print("start by epoch: %d"%(start))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

        print('Finished building Network.')

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    feed_dict = {images: input_img}
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])
    scp.misc.imsave(savename+'input.jpg', img1)
    scp.misc.imsave(savename+'output.png', up_color)
    scp.misc.imsave(savename+'input_box.png', fake_mask)
