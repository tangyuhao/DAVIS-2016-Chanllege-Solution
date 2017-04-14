import os
import math
import random
import glob

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('./')
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
import visualization
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.55, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def get_clip_info(clip_path):
    image_names = sorted([file for file in os.listdir(clip_path) if file.endswith(".jpg")])
    clip_info = []
    for i in range(0,len(image_names)):
        img = mpimg.imread(os.path.join(clip_path, image_names[i]))
        frame_name = os.path.join(os.path.basename(clip_path),image_names[i])
        [height, width] = img.shape[:2]
        rclasses, rscores, rbboxes =  process_image(img)
        frame_info = []
        for k in range(len(rclasses)):
            box_info = {
                "class": rclasses[k],
                "score": rscores[k],
                "box": rbboxes[k]
            }
            frame_info.append(box_info)
        clip_info.append({"frame_name":frame_name,"frame_info":frame_info})
    return clip_info, [height, width]
# output clip_info structure:
# [   
#     {
#         "frame_name":<foldername>/<filename>.jpg,
#         "frame_info": [
#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },
#                 ...
#             ]
#     }
#     {
#         "frame_name":<foldername>/<filename>.jpg,
#         "frame_info": [
#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },
#                 ...
#             ]
#     }
#     {
#         "frame_name":<foldername>/<filename>.jpg,
#         "frame_info": [
#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },

#                 {
#                     "class": int,
#                     "score": value from 0 to 1,
#                     "box": [ymin, xmin, ymax, xmax]
#                 },
#                 ...
#             ]
#     }
#     ...
# ]
#Person: person
# Animal: bird, cat, cow, dog, horse, sheep
# Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
# Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor


if __name__ == "__main__":
    # Test on some demo image and visualize output.
    path = '../BF_Segmentation/DAVIS/images/scooter-black'
    path = 'testtttt'
    image_names = sorted([file for file in os.listdir(path) if file.endswith(".jpg")])
    max_len = len(image_names) + 1
    need_frames = 30
    for i in range(0,min(need_frames-1,max_len-1)):
        img = mpimg.imread(os.path.join(path, image_names[i]))
        rclasses, rscores, rbboxes =  process_image(img)

    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
