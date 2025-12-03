from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op


class LoopIntegrationTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, test_fn, *inputs):
        with self.converted(test_fn, [break_statements,
                                      continue_statements,
                                      control_flow],
                            {}, (constant_op.constant,)) as result:
            self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

    def test_while_loop_with_else(self):
        def test_fn(x):
            while x > 2:
                x /= 2
            else:
                x += 1
            return x

        self.assertTransformedEquivalent(test_fn, 4)


partitionedCallTest = LoopIntegrationTest()
partitionedCallTest.test_while_loop_with_else()
