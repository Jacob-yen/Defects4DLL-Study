import io
import onnx
import onnx.numpy_helper
import torch
from torch.testing._internal import common_utils

class TestONNXExport(common_utils.TestCase):

    @common_utils.skipIfCaffe2
    def test_onnx_aten(self):
        class ModelWithAtenFmod(torch.nn.Module):
            def forward(self, x, y):
                return torch.fmod(x, y)

        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        f = io.BytesIO()
        torch.onnx.export(
            ModelWithAtenFmod(),
            (x, y),
            f,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertAtenOp(onnx_model, "fmod", "Tensor")


fx = TestONNXExport()
fx.test_onnx_aten()
