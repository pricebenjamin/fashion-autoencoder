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

input_shape = [120, 96, 3] # H, W, C
flat_size = np.product(input_shape)

data_dir = '~/data/hessel/small'
data_dir = os.path.expanduser(data_dir)
image_ext = '*.png'

# model_dir = '~/tf-saves/fashion-net'
# model_dir = os.path.expanduser(model_dir)

# ckpt_name = 'naive-dense.ckpt'
# ckpt_path = os.path.join(model_dir, ckpt_name)


# Construct the graph
# x = tf.placeholder(tf.float32, [None, *input_shape])

# h2  = tf.layers.flatten(x)
# h1  = tf.layers.dense(h2, 2048, activation=tf.nn.leaky_relu)
# h0  = tf.layers.dense(h1,  128, activation=None)
# h1_ = tf.layers.dense(h0, 2048, activation=tf.nn.leaky_relu)
# h2_ = tf.layers.dense(h1_, flat_size, activation=None)

# reconstructed_images = tf.reshape(h2_, tf.shape(x))

# with tf.name_scope('loss'):
#     error = h2_ - h2
#     mse = tf.reduce_mean(tf.square(error), name='mse')

# optimizer = tf.train.AdamOptimizer(learning_rate)
# train_op  = optimizer.minimize(mse)

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()


def parser(image_filename_tensor):
    image_contents = tf.read_file(image_filename_tensor)
    image = tf.image.decode_png(image_contents, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [120, 96])
    return image

ds = tf.data.Dataset.from_tensor_slices(image_filenames)

ds = ds.apply(
    tf.contrib.data.map_and_batch(
        map_func=parser,
        batch_size=4,
        num_parallel_calls=4))

ds = ds.prefetch(10)
ds = ds.make_initializable_iterator()
ds_init = ds.initializer

image_batch = ds.get_next()


def main():

    # Construct list of image filenames
    glob_string = os.path.join(data_dir, image_ext)
    image_filenames = sorted(glob(glob_string))
    print('Found {} image files.'.format(len(image_filenames)))

    # Initialize KFolds object
    folded_data = KFolds(image_filenames)
    training_images, eval_images = folded_data.get_fold(0)

    # Create dataset nodes
    training_iterator = get_dataset_iterator(
        training_images,
        training=True,
        image_type='png',
        num_repeats=1,
        batch_size=4,
        num_parallel_calls=4,
        prefetch_buffer_size=10)

    evaluation_iterator = get_dataset_iterator(
        eval_images,
        training=False,
        image_type='png',
        batch_size=4,
        num_parallel_calls=4,
        prefetch_buffer_size=10)

    train_batch = training_iterator.get_next()
    eval_batch = evaluation_iterator.get_next()

    # Create session configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Launch configured session
    with tf.Session(config=config) as sess:
        sess.run(training_iterator.initializer)
        sess.run(evaluation_iterator.initializer)

        for i in range(3):
            print('Fetching batches...')
            tb, eb = sess.run([train_batch, eval_batch])
            fig, axes = plt.subplots(2, 4)
            row1, row2 = axes
            for j in range(4):
                row1[j].imshow(tb[j])
                row2[j].imshow(eb[j])
            plt.show()
            print('Reinitializing...')
            sess.run(training_iterator.initializer)
            sess.run(evaluation_iterator.initializer)



    return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
