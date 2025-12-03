from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
import tensorflow

class OnesTest(test.TestCase):

    def testQintDtype(self):
        @def_function.function(autograph=False)
        def f():
            ones = array_ops.ones([2, 3], dtype=dtypes_lib.quint8)
            return math_ops.cast(ones, dtypes_lib.int32)

        self.evaluate(f())


fx = OnesTest()
fx.testQintDtype()
