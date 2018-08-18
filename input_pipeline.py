import tensorflow as tf

def parse_jpeg_image(image_filename_tensor):
    '''Reads and decodes a 3-channel JPEG image file.

    Args:
        image_filename_tensor: tf.string tensor containing the complete path
                               to a single image

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
    image = tf.cast(image, tf.float32, name='Cast_Image_to_Float')
    image = tf.divide(image, 255.0, name='Scale_RGB_Values')

    # Insert augmentation routine here, if desired

    return image

def parse_png_image(image_filename_tensor):
    '''Reads and decodes a 3-channel PNG image file.

    Args:
        image_filename_tensor: tf.string tensor containing the complete path
                               to a single PNG image

    Returns:
        A tf.float32 tensor containing scaled RGB image data
    '''

    image_contents = tf.read_file(
        image_filename_tensor,
        name='Read_PNG_Image')
    image = tf.image.decode_png(
        image_contents,
        channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    return image

def get_dataset_iterator(
    image_filenames, 
    training=False,
    image_type='png',
    num_repeats=1,
    batch_size=16,
    num_parallel_calls=None,
    prefetch_buffer_size=4):
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

    assert image_type.lower() in ['png', 'jpg', 'jpeg']

    n_images = len(image_filenames)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

    if training:
        shuffle_buffer_size = n_images * num_repeats

        image_dataset = image_dataset.apply(
            tf.contrib.data.shuffle_and_repeat(
                shuffle_buffer_size,
                count=num_repeats)) # Num. times to repeat dataset

    map_func = parse_png_image if image_type == 'png' else parse_jpeg_image

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
