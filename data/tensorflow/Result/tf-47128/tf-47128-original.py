import tensorflow
from tensorflow.python import keras
from tensorflow.python.platform import test


class SerializeKerasObjectTest(test.TestCase):
    def test_serialize_type_object_initializer(self):
        layer = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.ones,
            bias_initializer=keras.initializers.zeros)
        keras.layers.serialize(layer)


fx = SerializeKerasObjectTest()
fx.test_serialize_type_object_initializer()
# codeEnd
