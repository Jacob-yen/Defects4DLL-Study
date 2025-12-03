# codeStart
from tensorflow.python.eager import def_function
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops


class EmbeddingLookupTest(test_util.TensorFlowTestCase):

    def testEmbeddingLookupGradientsHaveKnownShape(self):
        x = resource_variable_ops.ResourceVariable(
            initial_value=np.zeros([3, 3]),
            trainable=True,
            shape=[3, 3],
            dtype=dtypes.float32)

        @def_function.function(input_signature=[])
        def _call():
            with gradients.GradientTape() as tape:
                y = embedding_ops.embedding_lookup_v2(x, [0])
                loss = math_ops.reduce_sum(y)
            grads = tape.gradient(loss, x)
            self.assertAllEqual(grads.shape, [3, 3])
            return ops.convert_to_tensor(grads)

        concrete_call = _call.get_concrete_function()
        self.evaluate(concrete_call())


testClass = EmbeddingLookupTest()
testClass.testEmbeddingLookupGradientsHaveKnownShape()
# codeEnd