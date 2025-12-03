import tensorflow
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.preprocessing import image as preprocessing_image
from tensorflow.python.data import Dataset
import numpy as np


class TestImage(keras_parameterized.TestCase):
    def test_smart_resize_tf_dataset(self):
        test_input_np = np.random.random((2, 20, 40, 3))
        test_ds = Dataset.from_tensor_slices(test_input_np)

        resize = lambda img: preprocessing_image.smart_resize(img, size=size)

        size = (50, 50)
        test_ds.map(resize)


fx = TestImage()
fx.test_smart_resize_tf_dataset()
