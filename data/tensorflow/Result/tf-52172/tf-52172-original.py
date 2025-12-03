from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_ops_impl


class SelectDistortedCropBoxTest(test_util.TensorFlowTestCase):

    def testDeterminismExceptionThrowing(self):
        with test_util.deterministic_ops():
            image_ops_impl.sample_distorted_bounding_box_v2(image_size=[50, 50, 1], bounding_boxes=[[[0., 0., 1., 1.]]], )


partitionedCallTest = SelectDistortedCropBoxTest()
partitionedCallTest.testDeterminismExceptionThrowing()
