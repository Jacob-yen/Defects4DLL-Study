import io
import copy
import unittest
import onnxruntime
import torch
import numpy as np
import os
from typing import List

opset_version = 11
keep_initializers_as_inputs = False
onnx_shape_inference = True


def setUp():
    torch.manual_seed(0)
    onnxruntime.set_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(seed=0)
    os.environ['ALLOW_RELEASED_ONNX_OPSET_ONLY'] = '0'


def run_test(model, input, rtol=0.001, atol=1e-07,
             do_constant_folding=True, batch_size=2, use_gpu=True, dynamic_axes=
             None, test_with_inputs=None, input_names=None, output_names=None,
             fixed_batch_size=False, dict_check=True, training=None,
             remained_onnx_input_idx=None):
    def _run_test(m, remained_onnx_input_idx, flatten=True):
        return run_model_test(m, batch_size=batch_size, input=
        input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                              do_constant_folding=do_constant_folding, dynamic_axes=
                              dynamic_axes, test_with_inputs=test_with_inputs,
                              input_names=input_names, output_names=output_names,
                              fixed_batch_size=fixed_batch_size, dict_check=dict_check,
                              training=training, remained_onnx_input_idx=
                              remained_onnx_input_idx, flatten=flatten)

    if isinstance(remained_onnx_input_idx, dict):
        scripting_remained_onnx_input_idx = remained_onnx_input_idx[
            'scripting']
    else:
        scripting_remained_onnx_input_idx = remained_onnx_input_idx
    script_model = torch.jit.script(model)
    _run_test(script_model, scripting_remained_onnx_input_idx,
              flatten=False)

def flatten_tuples(elem):
    tup = []
    for t in elem:
        if isinstance(t, (tuple)):
            tup += flatten_tuples(t)
        else:
            tup += [t]
    return tup


def to_numpy(elem):
    if isinstance(elem, torch.Tensor):
        if elem.requires_grad:
            return elem.detach().cpu().numpy()
        else:
            return elem.cpu().numpy()
    elif isinstance(elem, list) or isinstance(elem, tuple):
        return [to_numpy(inp) for inp in elem]
    elif isinstance(elem, bool):
        return np.array(elem, dtype=bool)
    elif isinstance(elem, int):
        return np.array(elem, dtype=int)
    elif isinstance(elem, float):
        return np.array(elem, dtype=float)
    elif isinstance(elem, dict):
        dict_ = []
        for k in elem:
            dict_ += [to_numpy(k)] + [to_numpy(elem[k])]
        return dict_
    else:
        return RuntimeError("Input has unknown type.")


def convert_to_onnx(model, input=None, opset_version=11, example_outputs=None,
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
    input = flatten_tuples(input)
    inputs = to_numpy(input)
    num_model_inputs = len(ort_sess.get_inputs())
    # 自动裁剪 inputs
    if len(inputs) > num_model_inputs:
        inputs = inputs[:num_model_inputs]  # 裁剪多余的输入
    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs) if i < len(ort_sess.get_inputs()))
    ort_outs = ort_sess.run(None, ort_inputs)
    return inline_flatten_list(ort_outs, [])


def ort_compare_with_pytorch(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = [to_numpy(outp) for outp in output]

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False, dict_check=True,
                   training=None, remained_onnx_input_idx=None,
                   flatten=True):
    if training is not None and training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training is None or training == torch.onnx.TrainingMode.EVAL:
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

        ort_sess = convert_to_onnx(model, input=input, opset_version=opset_version,
                                   example_outputs=output, do_constant_folding=do_constant_folding,
                                   keep_initializers_as_inputs=keep_initializers_as_inputs,
                                   dynamic_axes=dynamic_axes, input_names=input_names,
                                   output_names=output_names, fixed_batch_size=fixed_batch_size, training=training,
                                   onnx_shape_inference=onnx_shape_inference)
        # compute onnxruntime output prediction
        if remained_onnx_input_idx is not None:
            input_onnx = []
            for idx in remained_onnx_input_idx:
                input_onnx.append(input[idx])
            input = input_onnx

        input_copy = copy.deepcopy(input)
        if flatten:
            input_copy, _ = torch.jit._flatten(input_copy)

        ort_outs = run_ort(ort_sess, input_copy)
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
                if flatten:
                    test_input, _ = torch.jit._flatten(test_input)
                ort_outs = run_ort(ort_sess, test_input)
                ort_compare_with_pytorch(ort_outs, output, rtol, atol)