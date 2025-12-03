import tensorflow
from absl.testing import parameterized
import numpy as np
import unittest
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class KerasActivationsTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(("1d", (5,)),
                                    ("2d", (2, 5)),
                                    ("3d", (2, 2, 3)))
    def test_softmax(self, shape):
        x = backend.placeholder(ndim=len(shape))
        f = backend.function([x], [activations.softmax(x, axis=-1)])


if __name__ == '__main__':
    unittest.main()
# codeEnd
