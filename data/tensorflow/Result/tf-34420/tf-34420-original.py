import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import text_vectorization
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
import tensorflow


class TextVectorizationPreprocessingTest(keras_parameterized.TestCase,
                                         preprocessing_test_utils.PreprocessingLayerTest):
    def test_standardize_with_no_identical_argument(self):
        standardize = "".join(["lower", "_and_strip_punctuation"])
        layer = text_vectorization.TextVectorization(standardize=standardize)

        input_data = keras.Input(shape=(1,), dtype=dtypes.string)
        layer(input_data)


fx = TextVectorizationPreprocessingTest()
fx.test_standardize_with_no_identical_argument()
