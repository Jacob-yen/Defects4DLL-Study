from absl.testing import parameterized
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class AdditiveAttentionTest(test.TestCase, parameterized.TestCase):
    def test_mixed_float16_policy(self):
        # Test case for GitHub issue:
        # https://github.com/tensorflow/tensorflow/issues/46064
        with policy.policy_scope('mixed_float16'):
            q = math_ops.cast(random_ops.random_uniform((2, 3, 4), seed=1), 'float16')
            v = math_ops.cast(random_ops.random_uniform((2, 3, 4), seed=2), 'float16')
            k = math_ops.cast(random_ops.random_uniform((2, 3, 4), seed=3), 'float16')
            layer = dense_attention.AdditiveAttention(causal=True)
            layer([q, v, k])


partitionedCallTest = AdditiveAttentionTest()
partitionedCallTest.test_mixed_float16_policy()
