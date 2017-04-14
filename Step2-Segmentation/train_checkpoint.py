from datetime import datetime
import logging
import sys
import time
import random
import tensorflow as tf
import fcn8_vgg_ours
import numpy as np
import pdb
import os, glob

RGB_IMG = 0
GT_IMG = 1
BOX_IMG = 2
NUM_EPOCHS = 40000
BATCH_SIZE = 10
WIDTH = 500
HEIGHT = 500
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    filename="train.log")

def load_training_set(train_file, valid_file):
    '''
    return train_set and val_set
    '''
    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    random.shuffle(trainlines)
    train_img = []
    train_gt = []
    train_box = []
    for line in trainlines:
        #print(line)
        line = line.split(",")
        train_img.append(line[RGB_IMG])
        train_gt.append(line[GT_IMG])
        train_box.append(line[BOX_IMG])
    ftrain.close()
    fvalid = open(valid_file, "r")
    validlines = fvalid.read().splitlines()
    random.shuffle(validlines)
    valid_img = []
    valid_gt = []
    valid_box = []
    for line in validlines:
        line = line.split(",")
        valid_img.append(line[RGB_IMG])
        valid_gt.append(line[GT_IMG])
        valid_box.append(line[BOX_IMG])
    fvalid.close()
    return [train_img, train_gt, train_box], [valid_img, valid_gt, valid_box]

def toOneHot(array):
    '''
    input: array with size : [BATCHES,HEIGHT,WIDTH]
    output: array with size : [BATCHES,HEIGHT,WIDTH,2]
    '''
    gt = (np.array([0,255]) == array[:,:,:,None]).astype(float)
    return gt

def single_JPEGimage_reader(filename_queue):
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = (tf.to_float(tf.image.decode_jpeg(image_file, channels=3)))
    image = tf.image.resize_images(image,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

def single_PNGimage_reader(filename_queue):
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.to_float(tf.image.decode_png(image_file, channels=1))
    image = tf.image.resize_images(image,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # pixel distribution ground truth 
    return image


def train_label_reader(inputrgb_queue, inputmsk_queue, label_queue):
    min_queue_examples = 128
    num_threads = 4
    input_ = single_JPEGimage_reader(inputrgb_queue)
    mask = single_PNGimage_reader(inputmsk_queue)
    label = single_PNGimage_reader(label_queue)
    input_.set_shape([HEIGHT,WIDTH,3])
    mask.set_shape([HEIGHT,WIDTH,1])
    label.set_shape([HEIGHT,WIDTH,1])
    input_batch, mask_batch, label_batch = tf.train.shuffle_batch(
        [input_, mask, label],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE,
        seed=3,
        min_after_dequeue=min_queue_examples)
    return input_batch, mask_batch, label_batch

def next_batch(inputrgb_queue, inputmsk_queue, label_queue):
    input_batch, mask_batch, label_batch = train_label_reader(inputrgb_queue, inputmsk_queue, label_queue)
    batch = tf.concat([input_batch, mask_batch, label_batch], axis=3)
    return batch


sess = tf.InteractiveSession()
images = tf.placeholder("float")
batch_images = tf.expand_dims(images, 0)

vgg_fcn = fcn8_vgg_ours.FCN8VGG()
with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images, train = True, debug=True)



labels = tf.placeholder("int32", [None, HEIGHT, WIDTH])
loss = fcn8_vgg_ours.pixel_wise_cross_entropy(vgg_fcn.upscore32, labels, num_classes = 2)    
tf.summary.scalar('pixel_wise_cross_entropy_loss', loss)
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train_summary', sess.graph)

train_step = tf.train.AdamOptimizer(1e-6,0.9).minimize(loss)
logging.info("********* CNN constructed *********")

train_file = "./all_train_imgs.csv"
valid_file = "./all_val_imgs.csv"

train_set, val_set = load_training_set(train_file, valid_file)
inputrgb_queue = tf.train.string_input_producer(train_set[RGB_IMG], shuffle=False)
label_queue = tf.train.string_input_producer(train_set[GT_IMG], shuffle=False)
inputmsk_queue = tf.train.string_input_producer(train_set[BOX_IMG], shuffle=False)
batch = next_batch(inputrgb_queue, inputmsk_queue, label_queue)
logging.info("********* pipeline constructed *********")
init_glb = tf.global_variables_initializer()
init_loc = tf.local_variables_initializer()
sess.run(init_glb)
sess.run(init_loc)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
ckpt_dir = "./checkpoints"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
start = 0
if ckpt and ckpt.model_checkpoint_path:
    start = int(ckpt.model_checkpoint_path.split("-")[1])
    print("start by epoch: %d"%(start))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)
last_save_epoch = start
try:
    for i in range(start, NUM_EPOCHS):
        print('epoch %d ...'%i)
        tmp = batch.eval()
        input_ = tmp[:,:,:,:4]
        label_ = tmp[:,:,:,4]
        start_time = time.time()
        train_step.run(feed_dict={batch_images:input_, labels:label_})
        print('Train: time elapsed: %.3fs.'%(time.time()-start_time))
        if i % 40 == 0 and i > start:
            summary = sess.run(merged_summary, feed_dict={batch_images:input_, labels:label_})
            train_writer.add_summary(summary, i)
        if i % 1000 == 0 and i > start:
            # delete old checkpoints
            files = glob.glob("checkpoints/checkpoint.ckpt-*")
            for filename in files:
                os.remove(filename)
            save_ckpt = "checkpoint.ckpt"
            model_saver = tf.train.Saver()
            last_save_epoch = i
            model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i+1)
except KeyboardInterrupt:
    print("get keyboard interrupt")
    if (i - last_save_epoch > 100):
        model_saver = tf.train.Saver()
        save_ckpt = "checkpoint.ckpt"
        model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i+1)

print("done")
