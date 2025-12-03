# codeStart
from tensorflow.python.platform import test
from tensorflow.python.tpu import bfloat16


class BFloat16ScopeTest(test.TestCase):
    def testScopeName(self):
        """Test if name for the variable scope is propagated correctly."""
        with bfloat16.bfloat16_scope() as bf:
            self.assertEqual(bf.name, "bfloat16")


fx = BFloat16ScopeTest()
fx.testScopeName()
# codeEnd
