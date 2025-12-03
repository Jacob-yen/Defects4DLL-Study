import tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


class GetDependentVariablesTest(test_util.TensorFlowTestCase):

    def testGetVariableByName(self):
        with context.graph_mode():
            init = constant_op.constant(100.0)
            var = variable_scope.variable(init, name='a/replica_1')
            if isinstance(var, variables.RefVariable):
                var._variable = array_ops.identity(var, name='a')
            else:
                var._handle = array_ops.identity(var, name='a')
            custom_gradient.get_variable_by_name('a')


fx = GetDependentVariablesTest()
fx.testGetVariableByName()
# codeEnd
