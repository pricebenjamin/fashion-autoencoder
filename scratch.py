import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

from glob import glob

# Local imports
from input_pipeline import get_dataset_iterator
from k_folds import KFolds

# TODO: Accept the following constants via commandline arguments
learning_rate = 0.0001
num_epochs = 10

data_dir = '~/data/hessel/small'
data_dir = os.path.expanduser(data_dir)
image_ext = '*.png'

# model_dir = '~/tf-saves/fashion-net'
# model_dir = os.path.expanduser(model_dir)

# ckpt_name = 'naive-dense.ckpt'
# ckpt_path = os.path.join(model_dir, ckpt_name)

# Construct list of image filenames
glob_string = os.path.join(data_dir, image_ext)
image_filenames = sorted(glob(glob_string))
print('Found {} image files.'.format(len(image_filenames)))

# Initialize KFolds object
# folded_data = KFolds(image_filenames)
# training_images, eval_images = folded_data.get_fold(0)

# Create dataset nodes
training_iterator = get_dataset_iterator(
    image_filenames,
    training=True,
    image_type='png',
    num_repeats=1,
    batch_size=32,
    num_parallel_calls=4,
    prefetch_buffer_size=10,
    compute_shape=True)

# Construct the graph
x = training_iterator.get_next()

h2  = tf.layers.flatten(x)
h1  = tf.layers.dense(h2, 128, activation=tf.nn.leaky_relu)
h0  = tf.layers.dense(h1,  64, activation=None)
h1_ = tf.layers.dense(h0, 128, activation=tf.nn.leaky_relu)
h2_ = tf.layers.dense(h1_, h2.shape[-1], activation=None)

reconstructed_images = tf.reshape(h2_, tf.shape(x))

with tf.name_scope('loss'):
    error = h2_ - h2
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op  = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def main():
    # Create session configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Launch configured session
    with tf.Session(config=config) as sess:
        sess.run(training_iterator.initializer)
        sess.run(init)

        step = 0
        while True:
            try:
                if step % 100 == 0:
                    _, loss = sess.run([train_op, mse])
                    print('loss = {}'.format(loss))
                else:
                    sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
            step += 1

    return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
