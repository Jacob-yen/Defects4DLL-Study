import tensorflow
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test


class AssertAllCloseTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_doesnt_raise_complex(self):
        x = constant_op.constant(1. + 0.1j, name="x")
        y = constant_op.constant(1.1 + 0.1j, name="y")
        check_ops.assert_near(x, y, atol=0., rtol=0.5, message="failure message")


fx = AssertAllCloseTest()
fx.test_doesnt_raise_complex()
