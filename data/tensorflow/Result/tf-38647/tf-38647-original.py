# codeStart
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
import numpy as np


class RandomFlipTest(keras_parameterized.TestCase):
    def _run_test(self, mode, expected_output=None, mock_random=None):
        np.random.seed(1337)
        num_samples = 2
        orig_height = 5
        orig_width = 8
        channels = 3
        if mock_random is None:
            mock_random = [1 for _ in range(num_samples)]
            mock_random = np.reshape(mock_random, [2, 1, 1, 1])
        inp = np.random.random((num_samples, orig_height, orig_width, channels))
        if expected_output is None:
            expected_output = inp
            if mode == 'horizontal' or mode == 'horizontal_and_vertical':
                expected_output = np.flip(expected_output, axis=2)
            if mode == 'vertical' or mode == 'horizontal_and_vertical':
                expected_output = np.flip(expected_output, axis=1)
        with test.mock.patch.object(
                random_ops, 'random_uniform', return_value=mock_random):
            with tf_test_util.use_gpu():
                layer = image_preprocessing.RandomFlip(mode)
                actual_output = layer(inp, training=1)
                self.assertAllClose(expected_output, actual_output)

    def test_random_flip_horizontal_half(self):
        with CustomObjectScope({'RandomFlip': image_preprocessing.RandomFlip}):
            np.random.seed(1337)
            mock_random = [1, 0]
            mock_random = np.reshape(mock_random, [2, 1, 1, 1])
            input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
            expected_output = input_images.copy()
            expected_output[0, :, :, :] = np.flip(input_images[0, :, :, :], axis=1)
            self._run_test('horizontal', expected_output, mock_random)



fx = RandomFlipTest()
fx.test_random_flip_horizontal_half()
# codeEnd
