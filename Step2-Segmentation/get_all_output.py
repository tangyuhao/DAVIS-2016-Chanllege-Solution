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


test_clips_filename = "./testset.txt"
f = open(test_clips_filename, "r")
test_clips = f.read().splitlines()
test_img_prefix = "./DAVIS/images/"
test_img_folders = []
test_box_prefix = "./DAVIS/SSD_box/"
test_box_folders = []
output_seg_prefix = "./DAVIS/result/"
output_seg_folders = []
for clip in test_clips:
    test_img_folders.append(os.path.join(test_img_prefix,clip))
    test_box_folders.append(os.path.join(test_box_prefix,clip))
    output_seg_folders.append(os.path.join(output_seg_prefix,clip))

##### create the output folders #####
for output_seg_folder in output_seg_folders:
    if not os.path.exists(output_seg_folder):
        os.makedirs(output_seg_folder)

# now we get all test folders
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)
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
else:
    print('Fail! Cannot find checkpoint file')
    exit()
print('start running the Network')

# start get the output
for idx in range(len(test_clips)):
    image_names = sorted([file for file in os.listdir(test_img_folders[idx]) if file.endswith(".jpg")])
    box_names = sorted([file for file in os.listdir(test_box_folders[idx]) if file.endswith(".png")])
    box_paths = []
    for box_name in box_names:
        box_paths.append(os.path.join(test_box_folders[idx], box_name))

    print(box_names)
    for j in range(0,len(image_names)):
        img_path = os.path.join(test_img_folders[idx], image_names[j])
        cur_img = scp.misc.imread(img_path)
        need_box_path = os.path.join(test_box_folders[idx], image_names[j][:-4]+".png")
        output_seg_path = os.path.join(output_seg_folders[idx], image_names[j][:-4]+".png")
        print("outputSegPath:",output_seg_path)
        if need_box_path in box_paths:
            continue
            # good case
            cur_mask = scp.misc.imread(need_box_path,'L')
            msk_layer = np.expand_dims(cur_mask, axis = 2)
            input_tensor = np.append(cur_img, msk_layer, 2)
            tensors = [vgg_fcn.pred_up]
            feed_dict = {images: input_tensor}
            [up_result] = sess.run(tensors, feed_dict=feed_dict)
            output_segmentation = utils.color_image(up_result[0])
            #import pdb; pdb.set_trace()

            scp.misc.imsave(output_seg_path, output_segmentation)
        else:
            # bad case
            last_msk_name = image_names[j-1][:-4] + ".png"
            last_msk_path = os.path.join(output_seg_folders[idx], last_msk_name)
            cur_mask = scp.misc.imread(last_msk_path,'L')
            msk_layer = np.expand_dims(cur_mask, axis = 2)
            input_tensor = np.append(cur_img, msk_layer, 2)
            tensors = [vgg_fcn.pred_up]
            feed_dict = {images: input_tensor}
            [up_result] = sess.run(tensors, feed_dict=feed_dict)
            output_segmentation = utils.color_image(up_result[0])

            scp.misc.imsave(output_seg_path, output_segmentation)


