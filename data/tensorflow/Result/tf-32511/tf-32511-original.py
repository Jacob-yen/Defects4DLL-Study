import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class TestModelCloning(keras_parameterized.TestCase):

    def test_functional_cloning_with_tensor_kwarg(self):
        """Test that cloning works with models that use Tensor kwargs."""

        class LayerWithTensorKwarg(keras.layers.Layer):

            def call(self, inputs, tensor=None):
                if tensor is not None:
                    return inputs * math_ops.cast(tensor, dtypes.float32)
                else:
                    return inputs

        inputs = keras.layers.Input(shape=(3))
        t = array_ops.sequence_mask(array_ops.shape(inputs)[1])
        model = keras.models.Model(inputs, LayerWithTensorKwarg()(inputs, tensor=t))
        model.add_loss(math_ops.reduce_sum(model.outputs))

        input_arr = np.random.random((1, 3)).astype(np.float32)
        with ops.Graph().as_default():
            with self.session() as sess:
                keras.models.clone_model(model)


partitionedCallTest = TestModelCloning()
partitionedCallTest.test_functional_cloning_with_tensor_kwarg()
