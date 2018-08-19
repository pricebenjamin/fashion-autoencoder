import tensorflow as tf
import numpy as np

import sys
import os

from glob import glob
from PIL import Image

# Local imports
from input_pipeline import get_dataset_iterator

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
evaluation_iterator = get_dataset_iterator(
    image_filenames,
    training=False,
    image_type='png',
    num_repeats=1,
    batch_size=16,
    num_parallel_calls=4,
    prefetch_buffer_size=10,
    compute_shape=True)

# Construct the graph
x = evaluation_iterator.get_next()

h2  = tf.layers.flatten(x)
h1  = tf.layers.dense(h2, 2048, activation=None)
h0  = tf.layers.dense(h1,  128, activation=None)
h1_ = tf.layers.dense(h0, 2048, activation=None)
h2_ = tf.layers.dense(h1_, h2.shape[-1], activation=tf.nn.sigmoid)

reconstructed_images = tf.reshape(h2_, tf.shape(x))

saver = tf.train.Saver()

def main():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(evaluation_iterator.initializer)
        saver.restore(sess, ckpt_path)

        encodings = []

        # Compute the outputs of the very first batch
        print('Encoding the first batch of images...')
        enc, outputs = sess.run([h0, reconstructed_images])
        encodings.append(enc)
        outputs = (outputs * 255.0).astype(np.uint8)
        print('Saving reconstructed images for the first batch...')
        for i, output in enumerate(outputs):
            im = Image.fromarray(output)
            im.save(image_filenames[i].split('/')[-1])

        # Compute the remaining encodings
        print('Encoding the remaining images...')
        while True:
            try:
                enc = sess.run(h0)
                encodings.append(enc)
            except tf.errors.OutOfRangeError:
                print('Done encoding.')
                break

        encodings = np.concatenate(encodings)
        print('Saving the encodings...')
        np.save('encodings.npy', encodings)

    return

if __name__ == '__main__':
    main()
