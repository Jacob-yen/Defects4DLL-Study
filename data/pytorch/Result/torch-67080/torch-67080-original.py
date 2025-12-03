import unittest
import onnxruntime
import torch
import numpy as np
import io
import os


class TestONNXRuntime(unittest.TestCase):
    opset_version = 9
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    onnx_shape_inference = True

    def setUp(self):
        torch.manual_seed(0)
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        self.is_script_test_enabled = True

    def test_large_model_with_non_str_file(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                dim = 5
                n = 40 * 4 * 10 ** 6
                self.emb = torch.nn.Embedding(n, dim)
                self.lin1 = torch.nn.Linear(dim, 1)
                self.seq = torch.nn.Sequential(
                    self.emb,
                    self.lin1,
                )

            def forward(self, input):
                return self.seq(input)

        x = torch.tensor([2], dtype=torch.long)
        f = io.BytesIO()
        err_msg = ("The serialized model is larger than the 2GiB limit imposed by the protobuf library. "
                   "Therefore the output file must be a file path, so that the ONNX external data can be written to "
                   "the same directory. Please specify the output file name.")
        with self.assertRaisesRegex(RuntimeError, err_msg):
            torch.onnx.export(LargeModel(), x, f)


fx = TestONNXRuntime()
fx.setUp()
fx.test_large_model_with_non_str_file()
