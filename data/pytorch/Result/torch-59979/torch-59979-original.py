import unittest
import onnxruntime
import torch
import numpy as np
import io
import copy
import os


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def convert_to_onnx(model, input=None, opset_version=9, example_outputs=None,
                    do_constant_folding=True, keep_initializers_as_inputs=True,
                    dynamic_axes=None, input_names=None, output_names=None,
                    fixed_batch_size=False, training=None,
                    onnx_shape_inference=True):
    # export the model to ONNX
    f = io.BytesIO()
    input_copy = copy.deepcopy(input)
    torch.onnx._export(model, input_copy, f,
                       opset_version=opset_version,
                       example_outputs=example_outputs,
                       do_constant_folding=do_constant_folding,
                       keep_initializers_as_inputs=keep_initializers_as_inputs,
                       dynamic_axes=dynamic_axes,
                       input_names=input_names, output_names=output_names,
                       fixed_batch_size=fixed_batch_size, training=training,
                       onnx_shape_inference=onnx_shape_inference)

    # compute onnxruntime output prediction
    ort_sess = onnxruntime.InferenceSession(f.getvalue())
    return ort_sess


def inline_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else inline_flatten_list(i, res_list)
    return res_list


def run_ort(ort_sess, input):
    input_copy = copy.deepcopy(input)
    input, _ = torch.jit._flatten(input_copy)
    inputs = [to_numpy(inp) for inp in input]
    num_model_inputs = len(ort_sess.get_inputs())
    # 自动裁剪 inputs
    if len(inputs) > num_model_inputs:
        inputs = inputs[:num_model_inputs]  # 裁剪多余的输入
    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)
    return inline_flatten_list(ort_outs, [])


def ort_compare_with_pytorch(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = [to_numpy(outp) for outp in output]

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(self, model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False, dict_check=True,
                   training=None, remained_onnx_input_idx=None):
    model.eval()
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        if isinstance(input, dict):
            input = (input,)
        input_args = copy.deepcopy(input)
        input_kwargs = {}
        if dict_check and isinstance(input_args[-1], dict):
            input_kwargs = input_args[-1]
            input_args = input_args[:-1]
        try:
            model_copy = copy.deepcopy(model)
            output = model_copy(*input_args, **input_kwargs)
        except Exception:
            output = model(*input_args, **input_kwargs)
        if isinstance(output, torch.Tensor):
            output = (output,)

        if not dict_check and isinstance(input[-1], dict):
            input = input + ({},)

        ort_sess = convert_to_onnx(model, input=input, opset_version=self.opset_version,
                                   example_outputs=output, do_constant_folding=do_constant_folding,
                                   keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                                   dynamic_axes=dynamic_axes, input_names=input_names,
                                   output_names=output_names, fixed_batch_size=fixed_batch_size, training=training,
                                   onnx_shape_inference=self.onnx_shape_inference)
        # compute onnxruntime output prediction
        if remained_onnx_input_idx is not None:
            input_onnx = []
            for idx in remained_onnx_input_idx:
                input_onnx.append(input[idx])
            input = input_onnx
        ort_outs = run_ort(ort_sess, input)
        ort_compare_with_pytorch(ort_outs, output, rtol, atol)

        # if additional test inputs are provided run the onnx
        # model with these inputs and check the outputs
        if test_with_inputs is not None:
            for test_input in test_with_inputs:
                if isinstance(test_input, torch.Tensor):
                    test_input = (test_input,)
                test_input_copy = copy.deepcopy(test_input)
                output = model(*test_input_copy)
                if isinstance(output, torch.Tensor):
                    output = (output,)
                if remained_onnx_input_idx is not None:
                    test_input_onnx = []
                    for idx in remained_onnx_input_idx:
                        test_input_onnx.append(test_input[idx])
                    test_input = test_input_onnx
                ort_outs = run_ort(ort_sess, test_input)
                ort_compare_with_pytorch(ort_outs, output, rtol, atol)


class TestONNXRuntime(unittest.TestCase):
    opset_version = 13
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

    def run_test(self, model, input, rtol=1e-3, atol=1e-7, do_constant_folding=True,
                 batch_size=2, use_gpu=True, dynamic_axes=None, test_with_inputs=None,
                 input_names=None, output_names=None, fixed_batch_size=False, dict_check=True,
                 training=None, remained_onnx_input_idx=None):
        def _run_test(m, remained_onnx_input_idx):
            return run_model_test(self, m, batch_size=batch_size,
                                  input=input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                                  do_constant_folding=do_constant_folding,
                                  dynamic_axes=dynamic_axes, test_with_inputs=test_with_inputs,
                                  input_names=input_names, output_names=output_names,
                                  fixed_batch_size=fixed_batch_size, dict_check=dict_check,
                                  training=training, remained_onnx_input_idx=remained_onnx_input_idx)

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx['scripting']
            tracing_remained_onnx_input_idx = remained_onnx_input_idx['tracing']
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        if self.is_script_test_enabled:
            script_model = torch.jit.script(model)
            _run_test(script_model, scripting_remained_onnx_input_idx)

        _run_test(model, tracing_remained_onnx_input_idx)

    def test_dynamic_repeat_interleave(self):

        class DynamicRepeatsModel(torch.nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(DynamicRepeatsModel(), (x, repeats), test_with_inputs=[(another_x, another_repeats)],
                      input_names=["input_1", "repeats_1"],
                      dynamic_axes={"input_1": {1: "w"}, "repeats_1": {0: "r"}})


fx = TestONNXRuntime()
fx.setUp()
fx.test_dynamic_repeat_interleave()
