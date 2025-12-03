import torch
import torch.onnx
from torch.onnx import (utils,
                        OperatorExportTypes,
                        TrainingMode)
from torch.onnx.symbolic_helper import _set_onnx_shape_inference
import onnx
import io
import unittest


class _BaseTestCase(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def _model_to_graph(self, model, input,
                        do_constant_folding=True,
                        training=TrainingMode.EVAL,
                        operator_export_type=OperatorExportTypes.ONNX,
                        input_names=None,
                        dynamic_axes=None):
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        # Need disable onnx_shape_inference for this test because it puts const node to initializers.
        _set_onnx_shape_inference(False)
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(model, input,
                                                              do_constant_folding=do_constant_folding,
                                                              _disable_torch_constant_prop=True,
                                                              operator_export_type=operator_export_type,
                                                              training=training,
                                                              input_names=input_names,
                                                              dynamic_axes=dynamic_axes)
        _set_onnx_shape_inference(True)
        return graph, params_dict, torch_out


class TestUtilityFuns_opset9(_BaseTestCase):
    opset_version = 9

    def _test_deduplicate_initializers(self, torchscript=False):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 3)
                self.layer2 = torch.nn.Linear(3, 3)

                # Reusing layers.
                self.layer3 = self.layer1

                # Reusing parameters.
                self.layer2.weight = self.layer1.weight
                self.layer1.bias = self.layer2.bias

                # Parameter with different tensors equal in value.
                self.param1 = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
                self.param2 = torch.nn.Parameter(torch.tensor([1., 2., 3.]))

            def forward(self, x):
                return self.layer3(self.layer2(self.layer1(x))) + self.param1 + self.param2

        model = torch.jit.script(MyModule()) if torchscript else MyModule()

        x = torch.randn(3, 3)
        param_name_set = set([k for k, _ in model.named_parameters()])

        # Test eval mode.
        model.eval()
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f,
                          opset_version=self.opset_version)
        graph = onnx.load(io.BytesIO(f.getvalue()))
        param_name_set.remove("param2")
        self.assertSetEqual(set([i.name for i in graph.graph.initializer]), param_name_set)

    def test_deduplicate_initializers(self):
        self._test_deduplicate_initializers(torchscript=False)


fx = TestUtilityFuns_opset9()
fx.test_deduplicate_initializers()
