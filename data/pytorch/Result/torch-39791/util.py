import unittest
import onnxruntime
import torch

import numpy as np
import io
import copy

opset_version = 11
keep_initializers_as_inputs = True  # For IR version 3 type export.


def setUp():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(seed=0)


def run_test(model, input, rtol=1e-3, atol=1e-7, do_constant_folding=True,
             batch_size=2, use_gpu=True, dynamic_axes=None, test_with_inputs=None,
             input_names=None, output_names=None, fixed_batch_size=False):
    def _run_test(m):
        return run_model_test(m, batch_size=batch_size,
                              input=input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                              do_constant_folding=do_constant_folding,
                              dynamic_axes=dynamic_axes, test_with_inputs=test_with_inputs,
                              input_names=input_names, output_names=output_names,
                              fixed_batch_size=fixed_batch_size)

    _run_test(model)


def ort_test_with_input(ort_sess, input, output, rtol, atol):
    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    inputs = list(map(to_numpy, input))
    outputs = list(map(to_numpy, output))
    num_model_inputs = len(ort_sess.get_inputs())
    # 自动裁剪 inputs
    if len(inputs) > num_model_inputs:
        inputs = inputs[:num_model_inputs]  # 裁剪多余的输入
    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False):
    model.eval()

    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_copy = copy.deepcopy(input)
        output = model(*input_copy)
        if isinstance(output, torch.Tensor):
            output = (output,)

        # export the model to ONNX
        f = io.BytesIO()
        input_copy = copy.deepcopy(input)
        torch.onnx._export(model, input_copy, f,
                           opset_version=opset_version,
                           example_outputs=output,
                           do_constant_folding=do_constant_folding,
                           keep_initializers_as_inputs=keep_initializers_as_inputs,
                           dynamic_axes=dynamic_axes,
                           input_names=input_names, output_names=output_names,
                           fixed_batch_size=fixed_batch_size)

        # compute onnxruntime output prediction
        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        input_copy = copy.deepcopy(input)
        ort_test_with_input(ort_sess, input_copy, output, rtol, atol)

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
                ort_test_with_input(ort_sess, test_input, output, rtol, atol)
