import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class TestCTC(test.TestCase):

    def test_ctc_decode(self):
        depth = 6
        seq_len_0 = 5
        input_prob_matrix_0 = np.asarray(
            [[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
             [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
             [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
             [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
             [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
             [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]],
            dtype=np.float32)

        inputs = ([input_prob_matrix_0[t, :][np.newaxis, :]
                   for t in range(seq_len_0)] +
                  2 * [np.zeros((1, depth), dtype=np.float32)])

        inputs = keras.backend.variable(np.asarray(inputs).transpose((1, 0, 2)))

        input_length = keras.backend.variable(
            np.array([seq_len_0], dtype=np.int32))

        decode_truth = [
            np.array([1, 0, -1, -1, -1, -1, -1]),
            np.array([0, 1, 0, -1, -1, -1, -1])
        ]
        beam_width = 2
        top_paths = 2

        decode_pred_tf, log_prob_pred_tf = keras.backend.ctc_decode(
            inputs,
            input_length,
            greedy=False,
            beam_width=beam_width,
            top_paths=top_paths)

        self.assertTrue(np.alltrue(decode_truth[0] == keras.backend.eval(decode_pred_tf[0])))


fx = TestCTC()
fx.test_ctc_decode()
# codeEnd
