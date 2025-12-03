import unittest
import onnxruntime
import torch
import numpy as np
import io
import copy
import os



opset_version = 9
keep_initializers_as_inputs = True  # For IR version 3 type export.
onnx_shape_inference = True


def setUp():
    torch.manual_seed(0)
    onnxruntime.set_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(seed=0)
    os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"


def run_model_test_with_external_data(model, input, rtol=0.001, atol=1e-7,
                                      do_constant_folding=True, dynamic_axes=None,
                                      input_names=None, output_names=None,
                                      ort_optim_on=True, training=None, use_external_data_format=None):
    import os
    import tempfile

    if training is not None and training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training is None or training == torch.onnx.TrainingMode.EVAL:
        model.eval()
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
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_file_name = os.path.join(tmpdirname, "model.onnx")
            input_copy = copy.deepcopy(input)
            torch.onnx.export(model, input_copy, model_file_name,
                              opset_version=opset_version,
                              verbose=False,
                              do_constant_folding=do_constant_folding,
                              keep_initializers_as_inputs=keep_initializers_as_inputs,
                              dynamic_axes=dynamic_axes,
                              input_names=input_names, output_names=output_names)
            # compute onnxruntime output prediction
            ort_sess_opt = onnxruntime.SessionOptions()
            ort_sess_opt.graph_optimization_level = \
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if ort_optim_on else \
                    onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            ort_sess = onnxruntime.InferenceSession(model_file_name, sess_options=ort_sess_opt)
            input_copy = copy.deepcopy(input)
            ort_outs = run_ort(ort_sess, input_copy)
            ort_compare_with_pytorch(ort_outs, output, rtol, atol)

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


def convert_to_onnx(model, input=None, opset_version=9, do_constant_folding=True,
                    keep_initializers_as_inputs=True, dynamic_axes=None,
                    input_names=None, output_names=None,
                    fixed_batch_size=False, training=None,
                    onnx_shape_inference=True):
    # export the model to ONNX
    f = io.BytesIO()
    input_copy = copy.deepcopy(input)
    torch.onnx._export(model, input_copy, f,
                       opset_version=opset_version,
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