import imp
import sys

from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator import create_python_api
from tensorflow.python.util.tf_export import tf_export


@tf_export('test_op', 'test_op1', 'test.test_op2')
def test_op():
    pass


@tf_export('test1.foo', v1=['test.foo'])
def deprecated_test_op():
    pass


@tf_export('TestClass', 'NewTestClass')
class TestClass(object):
    pass


_TEST_CONSTANT = 5
_MODULE_NAME = 'tensorflow.python.test_module'


class CreatePythonApiTest(test.TestCase):

    def setUp(self):
        # Add fake op to a module that has 'tensorflow' in the name.
        sys.modules[_MODULE_NAME] = imp.new_module(_MODULE_NAME)
        setattr(sys.modules[_MODULE_NAME], 'test_op', test_op)
        setattr(sys.modules[_MODULE_NAME], 'deprecated_test_op', deprecated_test_op)
        setattr(sys.modules[_MODULE_NAME], 'TestClass', TestClass)
        test_op.__module__ = _MODULE_NAME
        TestClass.__module__ = _MODULE_NAME
        tf_export('consts._TEST_CONSTANT').export_constant(
            _MODULE_NAME, '_TEST_CONSTANT')

    def testConstantIsAdded(self):
        imports, _, _ = create_python_api.get_api_init_text(packages=[create_python_api._DEFAULT_PACKAGE], output_package='tensorflow', api_name='tensorflow', api_version=1)
        print("success")

partitionedCallTest = CreatePythonApiTest()
partitionedCallTest.setUp()
partitionedCallTest.testConstantIsAdded()
