import tensorflow
from tensorflow.python.distribute import collective_util
from tensorflow.python.eager import test


class OptionsTest(test.TestCase):

    def testCreateOptionsViaExportedAPI(self):
        options = collective_util._OptionsExported(bytes_per_pack=1)
        options.bytes_per_pack



fx = OptionsTest()
fx.testCreateOptionsViaExportedAPI()
# codeEnd
