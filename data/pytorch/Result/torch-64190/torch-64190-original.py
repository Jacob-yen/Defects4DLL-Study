import torch
import unittest
import onnx
import io


class TestUtilityFuns(unittest.TestCase):

    def test_duplicated_output_node(self):


        class DuplicatedOutputNet(torch.nn.Module):

            def __init__(self, input_size, num_classes):
                super(DuplicatedOutputNet, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, num_classes)

            def forward(self, input0, input1):
                out1 = self.fc1(input0)
                out2 = self.fc1(input1)
                return out1, out1, out2, out1, out2
        N, D_in, H, D_out = 64, 784, 500, 10
        pt_model = DuplicatedOutputNet(D_in, D_out)
        f = io.BytesIO()
        x = torch.randn(N, D_in)
        dynamic_axes = {'input0': {(0): 'input0_dim0', (1): 'input0_dim1'},
            'input1': {(0): 'input1_dim0', (1): 'input1_dim1'}, 'output-0':
            {(0): 'output-0_dim0', (1): 'output-0_dim1'}, 'output-1': {(0):
            'output-1_dim0', (1): 'output-1_dim1'}, 'output-2': {(0):
            'output-2_dim0', (1): 'output-2_dim1'}, 'output-3': {(0):
            'output-3_dim0', (1): 'output-3_dim1'}, 'output-4': {(0):
            'output-4_dim0', (1): 'output-4_dim1'}}
        torch.onnx.export(pt_model, (x, x), f, input_names=['input0',
            'input1'], output_names=['output-0', 'output-1', 'output-2',
            'output-3', 'output-4'], do_constant_folding=False, training=
            torch.onnx.TrainingMode.TRAINING, dynamic_axes=dynamic_axes,
            verbose=True, keep_initializers_as_inputs=True)
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.output[0].name, 'output-' + str(0))


fx = TestUtilityFuns()
fx.test_duplicated_output_node()
