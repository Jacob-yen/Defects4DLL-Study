import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
import tensorflow

class FFTShiftTest(test.TestCase):
    @test_util.run_deprecated_v1
    def testPlaceholder(self):
        x = array_ops.placeholder(shape=[None, None, None], dtype='float32')
        axes = None
        fft_ops.fftshift(x, axes=axes)

fx = FFTShiftTest()
fx.testPlaceholder()
# codeEnd
