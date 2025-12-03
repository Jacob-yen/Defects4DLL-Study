import tensorflow as tf


def correct_image_flip_left_right(image):
    return tf.reverse(image, axis=[-2])


PATCH_TF = False  # Change this to True to fix the bug
if PATCH_TF:
    tf.image.flip_left_right = correct_image_flip_left_right

image_input = tf.convert_to_tensor([
    # batch: 0
    [
        # y: 0
        [[0, 1, 2], [3, 4, 5]],  # x=0,1 channels=0,1,2
        # y: 1
        [[6, 7, 8], [9, 10, 11]],  # x=0,1 channels=0,1,2
    ],
])

image_flipped_directly = tf.image.flip_left_right(image_input)

expected_output = tf.convert_to_tensor([
    # batch: 0
    [
        # y: 0
        [[3, 4, 5], [0, 1, 2]],  # x=0,1 channels=0,1,2
        # y: 1
        [[9, 10, 11], [6, 7, 8]],  # x=0,1 channels=0,1,2
    ],
])

tf.assert_equal(image_flipped_directly, expected_output)


def generator():  yield image_input


dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int32)


def flip_it(image, do_flip: bool):
    if do_flip:
        return tf.image.flip_left_right(image)
    else:
        return image


dataset = dataset.map(lambda image: flip_it(image, tf.constant(True)))

image_flipped_via_dataset_map = next(iter(dataset))

tf.assert_equal(image_flipped_via_dataset_map, expected_output)
