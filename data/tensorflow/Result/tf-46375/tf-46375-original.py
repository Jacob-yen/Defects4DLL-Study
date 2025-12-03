import tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import summary_ops_v2


class SummaryOpsTest(test_util.TensorFlowTestCase):

    def testTrace_withProfiler(self):
        @def_function.function
        def f():
            x = constant_op.constant(2)
            y = constant_op.constant(3)
            return x ** y

        assert context.executing_eagerly()
        logdir = self.get_temp_dir()
        writer = summary_ops_v2.create_file_writer(logdir)
        summary_ops_v2.trace_on(graph=True, profiler=True)
        profiler_outdir = self.get_temp_dir()
        with writer.as_default():
            f()
            summary_ops_v2.trace_export(name='foo', step=1, profiler_outdir=profiler_outdir)


fx = SummaryOpsTest()
fx.testTrace_withProfiler()
