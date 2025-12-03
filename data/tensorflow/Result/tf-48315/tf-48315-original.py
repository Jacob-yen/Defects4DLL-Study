# codeStart
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.callbacks import BackupAndRestore

CALLBACK_HOOKS = [
    'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
    'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
    'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
    'on_test_begin', 'on_test_end', 'on_train_batch_begin',
    'on_train_batch_end', 'on_train_begin', 'on_train_end'
]


class CallAllHooks(keras.callbacks.Callback):
    """A callback that calls self._run for all hooks"""

    def __init__(self):
        for method_name in CALLBACK_HOOKS:
            setattr(self, method_name, self._run)

    def _run(self, *args, logs=None):
        raise NotImplementedError


class KerasCallbacksTest(keras_parameterized.TestCase):
    def test_logs_conversion(self):
        assert_dict_equal = self.assertDictEqual

        class MutateNumpyLogs(CallAllHooks):
            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                logs["numpy"] = 1

        class MutateTensorFlowLogs(CallAllHooks):
            def __init__(self):
                super(MutateTensorFlowLogs, self).__init__()
                self._supports_tf_logs = True

            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                logs["tf"] = 2

        class AssertNumpyLogs(CallAllHooks):
            def _run(self, *args, logs=None):
                logs = logs or args[-1]
                assert_dict_equal(logs, {"all": 0, "numpy": 1, "tf": 2})

        class AssertTensorFlowLogs(AssertNumpyLogs):
            def __init__(self):
                super(AssertTensorFlowLogs, self).__init__()
                self._supports_tf_logs = True

        cb_list = keras.callbacks.CallbackList([
            MutateNumpyLogs(),
            MutateTensorFlowLogs(),
            AssertNumpyLogs(),
            AssertTensorFlowLogs()])


        cb_list.on_epoch_begin(0, logs={"all": 0})



fx = KerasCallbacksTest()
fx.test_logs_conversion()
# codeEnd