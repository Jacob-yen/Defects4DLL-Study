import tensorflow
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import text_vectorization


class TextVectorizationLayerTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                 ):

    def test_scalar_input_int_mode_trim_to_len_limit(self):
        vocab_data = [
            "fire earth earth", "earth earth", "wind wind", "and wind and"
        ]
        input_data = "earth wind and fire fire and earth michigan"
        layer = text_vectorization.TextVectorization(output_sequence_length=3)
        layer.adapt(vocab_data)
        layer(input_data)



fx = TextVectorizationLayerTest()
fx.test_scalar_input_int_mode_trim_to_len_limit()