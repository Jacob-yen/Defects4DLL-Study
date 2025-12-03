import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ExtractGlimpseTest(test.TestCase):

    def testGlimpseNonNormalizedNonCentered(self):
        img = constant_op.constant(np.arange(25).reshape((1, 5, 5, 1)),
                                   dtype=dtypes.float32)
        with self.test_session():
            result1 = image_ops.extract_glimpse_v2(img, [3, 3], [[0, 0]],
                                                   centered=False, normalized=False)
            self.assertAllEqual(np.asarray([[0, 1, 2], [5, 6, 7], [10, 11, 12]]),
                                self.evaluate(result1)[0, :, :, 0])


partitionedCallTest = ExtractGlimpseTest()
partitionedCallTest.testGlimpseNonNormalizedNonCentered()
