import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import core


class LambdaLayerTest(keras_parameterized.TestCase):

    def test_dynamic(self):
        inp = keras.Input(shape=(10,))
        keras.layers.Lambda(lambda x_input: x_input, dynamic=True)(inp)


testClass = LambdaLayerTest()
testClass.test_dynamic()
# codeEnd
