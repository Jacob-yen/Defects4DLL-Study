from tensorflow.python.framework import test_util
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import slot_creator


class SlotCreatorTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testCreateSlotFromFirstMDimensionVariable(self):
        with self.test_session():
            s = variables.Variable([1.0, 2.5], name="var")
            p_v = variable_scope.get_variable(
                "var",
                shape=[2, 2],
                partitioner=partitioned_variables.fixed_size_partitioner(2))
            i, v = list(enumerate(p_v))[0]
            slot = slot_creator.create_slot(v, s.initialized_value(), name="slot")
            si = slot._save_slice_info
            variables.global_variables_initializer().run()
            self.assertAllEqual([2], si.full_shape)


fx = SlotCreatorTest()
fx.testCreateSlotFromFirstMDimensionVariable()
