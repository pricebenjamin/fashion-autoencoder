import tensorflow as tf
import numpy as np

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

model_dir = '~/tf-saves/fashion-net'
model_dir = os.path.expanduser(model_dir)

ckpt_name = 'naive-dense.ckpt'
ckpt_path = os.path.join(model_dir, ckpt_name)

# Construct list of image filenames
glob_string = os.path.join(data_dir, image_ext)
image_filenames = sorted(glob(glob_string))
print('Found {} image files.'.format(len(image_filenames)))

# Initialize KFolds object
# folded_data = KFolds(image_filenames)

# Create dataset nodes
training_iterator = get_dataset_iterator(
    image_filenames,
    training=True,
    image_type='png',
    num_repeats=1,
    batch_size=128,
    num_parallel_calls=4,
    prefetch_buffer_size=10,
    compute_shape=True)

# Construct the graph
x = training_iterator.get_next()

h2  = tf.layers.flatten(x)
h1  = tf.layers.dense(h2, 2048, activation=None)
h0  = tf.layers.dense(h1,  128, activation=None)
h1_ = tf.layers.dense(h0, 2048, activation=None)
h2_ = tf.layers.dense(h1_, h2.shape[-1], activation=tf.nn.sigmoid)

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
        # Initialize iterator and model weights
        sess.run(training_iterator.initializer)
        sess.run(init)

        # Save the initialized model
        saver.save(sess, ckpt_path)

        # Training loop
        for epoch in range(num_epochs):
            step = 0
            # Iterate batch by batch over the entire dataset
            # until the iterator runs out of values.
            while True:
                try:
                    if step % 10 == 0:
                        _, loss = sess.run([train_op, mse])
                        print('epoch = {}, step = {}, loss = {}'.format(
                            epoch, step, loss))
                    else: 
                        sess.run(train_op)
                    step += 1
                except tf.errors.OutOfRangeError:
                    print('Reached end of dataset. Reinitializing iterator...')
                    # Break out of the while loop
                    break

            # Reset the training iterator before the next epoch
            sess.run(training_iterator.initializer)

        # Save the model after all epochs are complete
        print('Saving model...')
        saver.save(sess, ckpt_path)
    return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
