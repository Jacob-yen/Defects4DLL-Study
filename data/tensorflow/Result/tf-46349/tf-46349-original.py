import tensorflow
from tensorflow.python.eager import context
from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def get_var(val, dtype, name=None):
    return variables.VariableV1(val, use_resource=True, dtype=dtype, name=name)


class AutoCastVariableTest(test.TestCase, parameterized.TestCase):

    def test_optimizer(self, optimizer_class=rmsprop.RMSprop, use_tf_function=False):
        x = get_var(1., dtypes.float32)
        x = autocast_variable.create_autocast_variable(x)
        y = get_var(1., dtypes.float32)
        opt = optimizer_class(learning_rate=1.)

        opt.minimize(lambda: x + y, var_list=[x, y])


atestClass = AutoCastVariableTest()
atestClass.test_optimizer()
