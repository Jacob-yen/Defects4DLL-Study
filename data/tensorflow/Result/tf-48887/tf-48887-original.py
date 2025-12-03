# codeStart
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import recurrent as rnn_v1
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
import numpy as np


class RNNTest(keras_parameterized.TestCase):

    def test_rnn_with_ragged_input(self, layer):
        ragged_data = ragged_factory_ops.constant(
            [[[1., 1., 1., 1., 1.], [1., 2., 3., 1., 1.]],
             [[2., 4., 1., 3., 1.]],
             [[2., 3., 4., 1., 5.], [2., 3., 1., 1., 1.], [1., 2., 3., 4., 5.]]],
            ragged_rank=1)
        dense_tensor, row_lengths = keras.backend.convert_inputs_if_ragged(
            ragged_data)
        np.random.seed(100)
        returning_rnn_layer = layer(4, go_backwards=True, return_sequences=True)

        x_ragged = keras.Input(shape=(None, 5), ragged=True)
        y_ragged = returning_rnn_layer(x_ragged)
        model = keras.models.Model(x_ragged, y_ragged)
        output_ragged = model.predict(ragged_data, steps=1)

        x_dense = keras.Input(shape=(3, 5))
        masking = keras.layers.Masking()(x_dense)
        y_dense = returning_rnn_layer(masking)
        model_2 = keras.models.Model(x_dense, y_dense)
        dense_data = ragged_data.to_tensor()
        output_dense = model_2.predict(dense_data, steps=1)

        output_dense = keras.backend.reverse(output_dense, [1])
        output_dense = ragged_tensor.RaggedTensor.from_tensor(
            output_dense, lengths=row_lengths)

        self.assertAllClose(keras.backend.reverse(output_ragged, [1]), output_dense)


fx = RNNTest()
fx.test_rnn_with_ragged_input(layer=rnn_v1.SimpleRNN)
# codeEnd
