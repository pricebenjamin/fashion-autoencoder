import tensorflow as tf
import numpy as np
from PIL import Image

def get_image_shape(image_filename):
    '''Determines the height, width, and number of channels of an image.

    Args:
        image_filename: string, a complete path to an image file

    Returns:
        A tuple of the form (height, width, num_channels).
    '''
    try:
        image = Image.open(image_filename)
    except FileNotFoundError as error:
        error_message = 'Unable to find the image `{}`.'.format(image_filename)
        raise Exception(error_message) from error

    image_as_array = np.array(image)
    return image_as_array.shape

def parse_jpeg_image(image_filename_tensor, shape=None):
    '''Reads and decodes a 3-channel JPEG image file.

    Args:
        image_filename_tensor: tf.string tensor containing the complete path
                               to a single image
        shape: tuple of integers, (height, width)

    Returns:
        A tf.float32 tensor containing scaled RGB image data
    '''

    image_contents = tf.read_file(
        image_filename_tensor,
        name='Read_Image_File')
    image = tf.image.decode_jpeg(
        image_contents,
        channels=3,
        # ratio=1, # integer downscaling ratio
        # dct_method='', # decompression method
        name='Decode_JPEG')

    if shape is not None:
        image = tf.image.resize_images(image, shape)

    # Convert image and scale to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image

def parse_png_image(image_filename_tensor, shape=None):
    '''Reads and decodes a 3-channel PNG image file.

    Args:
        image_filename_tensor: tf.string tensor containing the complete path
                               to a single PNG image
        shape: tuple of integers, (height, width)

    Returns:
        A tf.float32 tensor containing scaled RGB image data
    '''

    image_contents = tf.read_file(
        image_filename_tensor,
        name='Read_PNG_Image')
    image = tf.image.decode_png(
        image_contents,
        channels=3)

    if shape is not None:
        image = tf.image.resize_images(image, shape)

    # Convert image and scale to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image

def get_dataset_iterator(
    image_filenames, 
    training=False,
    image_type='png',
    num_repeats=1,
    batch_size=16,
    num_parallel_calls=None,
    prefetch_buffer_size=4,
    compute_shape=True):
    '''Creates a tf.data.Iterator from a list of image file paths.

    # TODO: Rewrite
    Args:
        image_filenames: python list of paths to image files
        training: bool; if True, the dataset will be shuffled and repeated
        args: object containing commandline arguments which specify batch size,
              epochs, parallelization parameters, etc.

    Returns:
        A tf.data.Iterator for iterating over the dataset. Make sure to 
        initialize the iterator before calling its get_next() method.
    '''

    image_type = image_type.lower()
    assert image_type in ['png', 'jpg', 'jpeg']

    n_images = len(image_filenames)
    assert n_images > 0

    image_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)
    if training:
        shuffle_buffer_size = n_images * num_repeats

        image_dataset = image_dataset.apply(
            tf.contrib.data.shuffle_and_repeat(
                shuffle_buffer_size,
                count=num_repeats)) # Num. times to repeat dataset

    # If the network architecture being used needs to know all dimensions
    # of the input, we can compute the shape of the first image in the list
    # and pass this shape to the parser.
    if compute_shape:
        h, w, c = get_image_shape(image_filenames[0])
        shape = (h, w)
        
        def parse_png_with_shape(image_filename):
            return parse_png_image(image_filename, shape)

        def parse_jpeg_with_shape(image_filename):
            return parse_jpeg_image(image_filename, shape)

        map_func = (
            parse_png_with_shape if image_type == 'png' else \
            parse_jpeg_with_shape)
    else:
        map_func = (
            parse_png_image if image_type == 'png' else \
            parse_jpeg_image)

    image_dataset = image_dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=map_func,
            batch_size=batch_size,
            # num_parallel_batches=None,
            # drop_remainder=False,
            num_parallel_calls=num_parallel_calls))

    image_dataset = image_dataset.prefetch(prefetch_buffer_size)
    image_dataset = image_dataset.make_initializable_iterator()

    return image_dataset
