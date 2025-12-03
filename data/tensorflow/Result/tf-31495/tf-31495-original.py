import re
from tensorflow.python.framework import test_util
from tensorflow.python.saved_model.model_utils import export_utils


class ExportTest(test_util.TensorFlowTestCase):
    def test_get_temp_export_dir(self):
        export_dir_base = "/tmp/export/"
        export_dir_1 = export_utils.get_timestamped_export_dir(export_dir_base)
        temp_export_dir = export_utils.get_temp_export_dir(export_dir_1).decode("utf-8")
        expected_1 = re.compile(export_dir_base + "temp-[\d]{10}")
        self.assertTrue(expected_1.match(temp_export_dir))


partitionedCallTest = ExportTest()
partitionedCallTest.test_get_temp_export_dir()
