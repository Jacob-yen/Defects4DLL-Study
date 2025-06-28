import ast
import astor
import re
import numpy as np
import random
from src.tools import param_utils
from src.tools.logger_utils import LoggerUtils
from typing import  List

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger

MAX_ATTEMPTS = 80


def is_torch_dtype_expression(arg):
    """
    This function is used to handle the args such as Attribute(value=Name(id='torch', ctx=Load()), attr='float32', ctx=Load())
    """
    if isinstance(arg, ast.Attribute):
        if isinstance(arg.value, ast.Name) and arg.value.id == 'torch' and arg.attr in param_utils.torch_dtypes:
            return True
    return False


def change_torch_dtype(arg):
    # get current dtype value
    current_dtype = arg.attr
    # get a new different dtype value from param_utils.torch_dtypes
    new_dtype = random.choice(list(set(param_utils.torch_dtypes) - set([current_dtype])))
    arg.attr = new_dtype
    return arg


def is_tf_dtype_expression(arg):
    """
    This function is used to handle the args such as Attribute(value=Name(id='dtypes', ctx=Load()), attr='float32', ctx=Load())
    """
    if isinstance(arg, ast.Attribute):
        if isinstance(arg.value, ast.Name) and (
                arg.value.id == 'tf' or arg.value.id == 'dtypes_lib' or arg.value.id == 'dtypes') and arg.attr in param_utils.tf_dtypes:
            return True
    return False


def change_tf_dtype(arg):
    # get current dtype value
    current_dtype = arg.attr
    # get a new different dtype value from param_utils.tf_dtypes
    new_dtype = random.choice(list(set(param_utils.tf_dtypes) - set([current_dtype])))
    arg.attr = new_dtype
    return arg


def create_new_tensor(dtype, framework):
    if framework == "pytorch":
        shape = tuple([random.randint(1, param_utils.MAX_SHAPE_NUMBER)])
        import torch
        new_tensor = torch.rand(shape)
        new_tensor = new_tensor.to(eval(dtype) if dtype.startswith('torch.') else eval('torch.' + dtype))
    elif framework == "tensorflow":
        shape = tuple([random.randint(1, param_utils.MAX_SHAPE_NUMBER)])
        import tensorflow as tf
        new_tensor = tf.constant(shape)
        new_tensor = new_tensor.to(
            eval(dtype) if dtype.startswith('tf.') or dtype.startswith('dtypes.') or dtype.startswith(
                'dtypes_lib.') else eval('dtypes.' + dtype))
    else:
        raise ValueError("No such framework: {}".format(framework))
    return new_tensor


def parse_range_str(range_str):
    def parse_num(s):
        if s == '-inf':
            return -1000
        elif s == 'inf' or s == '+inf':
            return 1000
        else:
            return eval(s)

    pattern = r'\[([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\]|\[([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\)|\(([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\]|\(([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\)'
    match = re.fullmatch(pattern, range_str)
    if not match:
        raise ValueError(f"{range_str} is not a valid range string.")

    start = parse_num(range_str.split(',')[0][1:].strip())
    end = parse_num(range_str.split(',')[1][:-1].strip())

    return start, end


def range_generator(s, is_integer):
    pattern = r'\[([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\]|\[([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\)|\(([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\]|\(([-+]?\d+|[-+]?inf)\s*,\s*([-+]?\d+|[-+]?inf)\)'
    match = re.fullmatch(pattern, s)
    if not match:
        raise ValueError(f"{s} is not a valid range string.")

    start = s.split(',')[0][1:].strip()
    end = s.split(',')[1][:-1].strip()

    if start == "inf":
        start = 1e7
    else:
        start = eval(start)
    if end == "inf":
        end = 1e7
    else:
        end = eval(end)
    if is_integer:
        return random.randint(int(start), int(end))
    else:
        return random.uniform(float(start), float(end))


def parse_default_value(value):
    def is_integer(_value):
        _value = _value.replace(" ", "")
        return _value.isdigit()

    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    if str(value) in ['None', 'True', 'False'] or is_integer(str(value)) or is_float(str(value)):
        return eval(str(value))
    else:
        return value


def default_generator(value_type, s=None, e=None):
    if "int" in value_type:
        s = -10 if s is None else s
        e = 10 if e is None else e
        return random.randint(-10, 10)
    elif "float" in value_type:
        s = -1 if s is None else s
        e = 1 if e is None else e
        return random.uniform(-1, 1)
    elif "bool" in value_type:
        return random.choice([True, False])
    else:
        raise NotImplementedError(f"Unsupported type: {value_type}")


def create_node(value):
    if value is None or isinstance(value, bool):
        return ast.NameConstant(value=value)
    elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
        return ast.Constant(value=value)
    elif isinstance(value, tuple):
        return ast.Tuple(elts=[ast.Constant(value=val) for val in value], ctx=ast.Load())
    elif isinstance(value, list):
        return ast.List(elts=[ast.Constant(value=val) for val in value], ctx=ast.Load())
    else:
        raise NotImplementedError(f"Unsupported type: {type(value)}")


def generate_data(dtype, ndim, cons, framework):
    def flatten_list(lst):
        flattened = []
        for item in lst:
            if isinstance(item, list):
                flattened.extend(flatten_list(item))
            else:
                flattened.append(item)
        return flattened

    start = end = None
    if 'range' in cons.keys():
        range_str = random.choice(cons['range'])
        range_str = str(range_str)
        start, end = parse_range_str(range_str)
    generator = param_utils.fetch_generator(dtype)
    if eval(ndim) == 0:
        return generator(start, end)
    elif eval(ndim) == 1:
        new_value = generator(start, end)
        # NOTE: we can currently only convert to tuple and list.
        if 'structure' in cons.keys() and ('list' in cons['structure'] or 'tuple' in cons['structure']):
            if 'tuple' in cons['structure']:
                return (new_value,)
            if 'list' in cons['structure']:
                return [new_value]
    elif eval(ndim) > 1:
        # if tensor, generate a tensor with ndim
        if 'tensor_t' in cons.keys():
            return create_new_tensor(dtype, framework)
        if 'list' in cons['structure'] or 'tuple' in cons['structure']:
            # generate a list or tuple with ndim
            new_values = [generator(start, end) for _ in range(eval(ndim))]
            if 'tuple' in cons['structure']:
                new_values = tuple(new_values)
            return new_values


def is_mutable(n):
    if isinstance(n, (
            ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant, ast.NameConstant, ast.keyword, ast.Attribute)):
        return True
    else:
        return False


def is_iterable(n):
    if isinstance(n, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return True
    else:
        return False


def equal_type(old_param, old_param_types, param_name, param_types):
    # we do not match the star arg and the normal arg
    if ("*" in old_param and "*" not in param_name) or ("*" not in old_param and "*" in param_name):
        return False
    for old_param_type in old_param_types:
        for param_type in param_types:
            if old_param_type == param_type:
                return True
            else:
                for name in ["float", "int", "tensor", "string", "object", "device", "bool", "module", "pickle"]:
                    if name in old_param_type.lower() and name in param_type.lower():
                        return True
    return False


# Some basic types are randomly generated ["float", "int", "device", "bool"]
def generated_basic_type(new_replace_api, param_name, param_types):
    # the parameter can have multiple types
    for param_type in param_types:
        param_val = ""
        param_type = str(param_type)
        if "float" in param_type:
            random_float = random.random() * 10
            param_val = str(random_float)
        elif "int" in param_type:
            random_int = random.randint(1, 10)
            param_val = str(random_int)
        elif "device" in param_type:
            random_element = random.choice(["'cuda'", "'cpu'"])
            param_val = random_element
        elif "bool" in param_type:
            random_element = random.choice(["True", "False"])
            param_val = random_element
        elif "array_like" in param_type:
            # randomly initialize a two-dimensional array (2<=column<=4, 2<=row<=4)
            column = random.randint(2, 4)
            row = random.randint(2, 4)
            # integer array
            array = np.random.randint(0, 10, (column, row))
            # convert the ndarray to list format
            param_val = array.tolist()
            param_val = str(param_val)
        elif "None" in param_type:
            param_val = "None"
        elif "torch.tensor" in param_type:
            # generate a tensor
            param_val = "torch.randn(1,2,3)"
        elif "tf.tensor" in param_type:
            # generate a tensor
            param_val = "tf.random.normal([1,2,3])"
        elif "nn.Module" in param_type:
            # generate a tensor
            param_val = "nn.Linear(1,1)"
        elif "torch.dtype" in param_type:
            # generate a tensor
            param_val = random.choice(["torch.float32", "torch.float64", "torch.int32", "torch.int64", "torch.bool"])
        elif "tf.dtype" in param_type:
            # generate a tensor
            param_val = random.choice(["tf.float32", "tf.float64", "tf.int32", "tf.int64", "tf.bool"])
        elif "string" in param_type:
            # generate a tensor
            param_val = "str('test')"
        else:
            # TODO: to support more types
            # raise ValueError(f"Unsupported type {param_type}")
            logger.warn(f"Unsupported type {param_type}. Skip")
            continue

        if param_name.startswith("*"):
            new_replace_api = f"{new_replace_api}{param_val}, "
        else:
            new_replace_api = f"{new_replace_api}{param_name}={param_val}, "
        return new_replace_api, True
    return new_replace_api, False


def join_new_api(relation_api, old_param_dtype_dict, new_require_param_dtype_dict,alignments):
    def get_value_by_alignments(alignments, param_name):
        align_tuple = alignments[param_name]
        if isinstance(align_tuple, str) and align_tuple == "N/A":
            return None
        else:
            arg_obj,_,_ = align_tuple
            return _parse_value(arg_obj)

    if relation_api == "tensorflow.keras.layers.Lambda":
        return "tensorflow.keras.layers.Lambda(lambda inputs: inputs[0])", True

    new_replace_api = f"{relation_api}("
    used_old_param = set()
    for param_name, param_type in new_require_param_dtype_dict.items():
        match_success = False
        for old_param, old_param_type in old_param_dtype_dict.items():
            if equal_type(old_param,old_param_type, param_name, param_type) and old_param not in used_old_param:
                # get the value of the old_param
                if old_param.startswith("*"):
                    star_arg_values = get_value_by_alignments(alignments,old_param)
                    if star_arg_values.startswith("[") and star_arg_values.endswith("]"):
                        star_arg_values = star_arg_values[1:-1]
                    new_replace_api = f"{new_replace_api}{star_arg_values}, "
                else:
                    if param_name.startswith("*"):
                        param_name = param_name[1:]
                    new_replace_api = f"{new_replace_api}{param_name}={get_value_by_alignments(alignments,old_param)}, "
                used_old_param.add(old_param)
                match_success = True
                break
        if not match_success:
            # Some basic types are randomly generated ["float", "int", "device", "bool"]
            new_replace_api, generate_success = generated_basic_type(new_replace_api, param_name, param_type)
            if not generate_success:
                return new_replace_api, False
    new_replace_api = new_replace_api.strip()
    if new_replace_api.endswith(","):
        new_replace_api = new_replace_api[:-1]
    new_replace_api = f"{new_replace_api})"
    return new_replace_api, True


def _parse_alignments(api_full_name, alignments):
    s = api_full_name + "("
    for arg_name, arg_tuple in alignments.items():
        if isinstance(arg_tuple, str) and arg_tuple == "N/A":
            s += f"{arg_name}=N/A,"
        else:
            arg_obj, arg_type, arg_idx = arg_tuple
            # if isinstance(arg_obj, list):
            #     s += f"{arg_name}=({','.join([str(_parse_value(item)) for item in arg_obj])}),"
            # else:
            if arg_name.startswith("*"):
                arg_name = arg_name[1:]
                star_arg_values = _parse_value(arg_obj)
                if star_arg_values.startswith("[") and star_arg_values.endswith("]"):
                    star_arg_values = star_arg_values[1:-1]
                s += f"{star_arg_values},"
            else:
                s += f"{arg_name}={_parse_value(arg_obj)},"
    if s.endswith(","):
        s = s[:-1]
    s += ")"
    return s


def _parse_value(n):
    if isinstance(n, ast.Constant):
        return n.value
    elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Num):
        return -n.operand.n
    elif isinstance(n, ast.List):
        # return [_parse_value(item) for item in n.elts]
        res = [str(_parse_value(item)) for item in n.elts]
        return "[" + ",".join(res) + "]"
    elif isinstance(n, ast.Tuple):
        res = [str(_parse_value(item)) for item in n.elts]
        return "(" + ",".join(res) + ")"
    elif isinstance(n, ast.Dict):
        res_dict = {str(_parse_value(item[0])): str(_parse_value(item[1])) for item in zip(n.keys, n.values)}
        res_str = "{"
        for k, v in res_dict.items():
            res_str += f"{k}:{v},"
        res_str += "}"
        return res_str
    elif isinstance(n, ast.Name):
        # append the name
        return n.id
    elif isinstance(n, ast.Call):
        # get the function name of the call
        func_name = _parse_value(n.func)
        return f"{func_name}({str(_parse_value(n.args))})"
        # logger.debug(f"meet function call in parameters: {n.func}")
        # return "@FuncCall@"
    elif isinstance(n, ast.Attribute):
        return f"{_parse_value(n.value)}.{n.attr}"
    elif isinstance(n, ast.keyword):
        return str(_parse_value(n.value))
    elif isinstance(n,List):
        # for positional star
        res = [str(_parse_value(item)) for item in n]
        return "[" + ",".join(res) + "]"
    elif isinstance(n,ast.Subscript):
        return f"{_parse_value(n.value)}[{_parse_value(n.slice)}]"
    else:
        # raise ValueError(f"Unsupported type {type(n)}: {ast.unparse(n)}")
        return ast.unparse(n)


def _fetch_args_value(args):
    values = []
    for arg in args:
        values.append(arg)
    return values


def signature_alignments(alignments, func_args, func_keyword_args, param_category):
    arg_ptr = 0
    # step1. align the args.
    for para in param_category['positional']['require']:
        if arg_ptr < len(func_args):
            # alignments[para] = func_args[arg_ptr]
            alignments[para] = (func_args[arg_ptr], "args", arg_ptr)
            arg_ptr += 1
    for para in param_category['positional']['star']:
        if arg_ptr < len(func_args):
            alignments[para] = (func_args[arg_ptr:], "args", arg_ptr)
            arg_ptr = len(func_args)
    for para in param_category['positional']['optional']:
        if arg_ptr < len(func_args):
            alignments[para] = (func_args[arg_ptr], "args", arg_ptr)
            arg_ptr += 1
    for para in param_category["keyword"]["require"]:
        if arg_ptr < len(func_args):
            alignments[para] = (func_args[arg_ptr], "args", arg_ptr)
            arg_ptr += 1
    for para in param_category["keyword"]["optional"]:
        if arg_ptr < len(func_args):
            alignments[para] = (func_args[arg_ptr], "args", arg_ptr)
            arg_ptr += 1

    for i in range(len(func_keyword_args)):
        param_name = func_keyword_args[i].arg
        if param_name:
            alignments[param_name] = (func_keyword_args[i], "keyword", i)
    return alignments


def construct_api_value(cons, framework):
    # get dtype and assert the type
    if "dtype" not in cons.keys():
        # we should infer from the invocation
        logger.error("Not implemented. We should infer dtype from the invocation")
    else:
        dtypes = set(cons["dtype"])
        available_dtypes = dtypes.intersection(set(param_utils.supportted_types))
        if len(available_dtypes) > 0:
            # randomly choose one type
            dtype = random.choice(list(available_dtypes))
            # get the ndim
            if "ndim" not in cons.keys() or cons['ndim'] == 'any':
                ndim = '0'
            else:
                # NOTE: convert to string since DocTer mixd the integer and string (0 and '0' both existed)
                if isinstance(cons['ndim'], int) or isinstance(cons['ndim'], str):
                    ndim = str(cons['ndim'])
                elif isinstance(cons['ndim'], list):
                    ndim = random.choice(cons['ndim'])
                    ndim = str(ndim)
                else:
                    raise ValueError("Can't recoginize the ndim: {}".format(cons['ndim']))
            assert ndim.isdigit(), f"The ndim: {ndim} is not a digit"
            data = generate_data(dtype, ndim, cons, framework)
            return data


class APIHandler:
    def __init__(self, invoke_name, api_full_name, signature_data, constraint_data) -> None:
        self.invoke_name = invoke_name
        self.api_full_name = api_full_name
        self.signature_data = signature_data
        self.constraint_data = constraint_data
        self.has_start_variables = False
        self.alignments = None

    def mutate_constraint(self, arg, val_range, constraint, framework):
        # mutate the value based on its type and the value range
        if isinstance(arg.value, bool):
            return ast.Constant(value=not arg.value)
        elif isinstance(arg.value, (int, float)):
            # 30% to sample the boundary value
            sample_boundary = random.random() < 0.3
            if not sample_boundary and val_range:
                # sample the value from the range
                start, end = parse_range_str(val_range)
                generator = param_utils.fetch_generator(str(type(arg.value)))
                return ast.Constant(value=generator(start, end))
            else:
                # sample the value from the boundary
                if isinstance(arg.value, int):
                    # boundary = [-1, 0, 1]
                    boundary = [0, 1]
                else:
                    # boundary = [-1.0, -0.5, 0.0, 0.5, 1.0]
                    boundary = [0.0, 0.5, 1.0]

                if val_range:
                    start, end = parse_range_str(val_range)
                    intersec_boundary = [i for i in boundary if start <= i <= end]
                    if len(intersec_boundary) > 0:
                        return ast.Constant(value=random.choice(intersec_boundary))
                    else:
                        return arg
                else:
                    return ast.Constant(value=random.choice(boundary))
        elif isinstance(arg.value, str):
            # currently we do not mutate string
            return arg
        elif arg.value is None:
            # we need to change it based on its type and range
            new_value = construct_api_value(constraint,framework)

            return create_node(new_value) if new_value else arg
        else:
            raise ValueError(f"Unknown type {type(arg.value)}")

    def mutate_api_value(self, arg, val_range, constraint, framework):
        if is_mutable(arg):
            if isinstance(arg, ast.Attribute):
                if framework == "pytorch":
                    if is_torch_dtype_expression(arg):
                        arg = change_torch_dtype(arg)
                elif framework == "tensorflow":
                    if is_tf_dtype_expression(arg):
                        arg = change_tf_dtype(arg)
                else:
                    raise ValueError("No such framework: {}".format(framework))
            elif isinstance(arg, (ast.Constant, ast.NameConstant)):
                arg = self.mutate_constraint(arg, val_range, constraint,framework)
            elif isinstance(arg, ast.keyword):
                # mutate the keyword.value
                arg.value = self.mutate_api_value(arg.value, val_range, constraint, framework)
            else:
                # recursively mutate the value
                # 30% chance to set to empty set/list/dict
                assert is_iterable(arg), f"Unknown type {type(arg)}"
                if random.random() < 0.3:
                    if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
                        arg.elts = []
                    else:
                        arg.keys = []
                        arg.values = []
                elif 0.3 <= random.random() < 0.6:
                    # randomly remove some elements
                    if isinstance(arg, (ast.List, ast.Tuple, ast.Set)) and len(arg.elts) != 0:
                        # randomly remove some elements
                        ridx = random.randint(0, len(arg.elts) - 1)
                        arg.elts = arg.elts[:ridx] + arg.elts[ridx + 1:]
                    elif isinstance(arg, ast.Dict) and len(arg.keys) != 0:
                        # randomly remove some elements
                        ridx = random.randint(0, len(arg.keys) - 1)
                        arg.keys = arg.keys[:ridx] + arg.keys[ridx + 1:]
                        arg.values = arg.values[:ridx] + arg.values[ridx + 1:]

                else:
                    # randomly choose one element to mutate
                    if isinstance(arg, (ast.List, ast.Tuple, ast.Set)) and len(arg.elts) != 0:
                        # randomly choose one elt
                        ridx = random.randint(0, len(arg.elts) - 1)
                        arg.elts[ridx] = self.mutate_api_value(arg.elts[ridx], val_range, constraint, framework)
                    elif isinstance(arg, ast.Dict) and len(arg.keys) != 0:
                        # randomly choose one key
                        ridx = random.randint(0, len(arg.keys) - 1)
                        arg.keys[ridx] = self.mutate_api_value(arg.keys[ridx], val_range, constraint, framework)
                        arg.values[ridx] = self.mutate_api_value(arg.values[ridx], val_range, constraint, framework)
        return arg

class APIReplacer(ast.NodeTransformer):
    def __init__(self, new_replace_api, target_position):
        self.target_position = target_position
        self.new_replace_api = new_replace_api

    def visit_Call(self, node):
        if (hasattr(node, "lineno") and hasattr(node, "end_lineno")
                and hasattr(node, "col_offset") and hasattr(node,"end_col_offset") and
                self.target_position == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)):
            new_node = ast.parse(self.new_replace_api, mode='eval').body
            new_node.lineno = node.lineno
            new_node.col_offset = node.col_offset
            return new_node
        return self.generic_visit(node)

    def conduct_mutation(self, source_code):
        tree = ast.parse(source=source_code, mode='exec')
        self.visit(tree)
        return astor.to_source(tree)


class APIFinder(ast.NodeVisitor, APIHandler):
    def __init__(self, target, invoke_name, api_full_name, signature_data, constraint_data, target_position,para_alignments, param_category):
        # APIHandler.__init__()
        super().__init__(invoke_name, api_full_name, signature_data, constraint_data)
        self.target = target
        assert len(self.target) == 1, "Target should only have one key"
        self.target_position = target_position
        self.param_dtype_dict = {}
        self.alignments = para_alignments
        self.param_category = param_category

    def visit_Call(self, node):
        if (hasattr(node, "lineno") and hasattr(node, "end_lineno") and 
                hasattr(node, "col_offset") and hasattr(node,"end_col_offset") and
                self.target_position == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)):
            call_code = ast.unparse(node).strip()
            # analyze the current parameters
            func_args = _fetch_args_value(node.args)
            func_keyword_args = _fetch_args_value(node.keywords)
            self.alignments = signature_alignments(self.alignments, func_args, func_keyword_args,self.param_category)

            self.param_dtype_dict = self.param_dtype_analysis(self.alignments, self.constraint_data)
            # print the alignment
            logger.debug(f"Function alignment: {_parse_alignments(self.api_full_name,self.alignments)}")

        self.generic_visit(node)

    def param_dtype_analysis(self, alignments, constraint_data):
        param_dtype_dict = {}
        for align_key, align_tuple in alignments.items():
            try:
                dtype_value = constraint_data['constraints'][align_key]['dtype']
            except KeyError:
                logger.debug(f"Can't find the dtype for {align_key} from {self.api_full_name}. Skip")
                continue
            if isinstance(align_tuple, str) and align_tuple == "N/A":
                continue
            param_dtype_dict[align_key] = dtype_value
        return param_dtype_dict


class APIParamModifier(ast.NodeVisitor, APIHandler):
    def __init__(self, target, invoke_name, api_full_name, signature_data, constraint_data, mutation_point, alignments, param_category):
        # APIHandler.__init__()
        super().__init__(invoke_name, api_full_name, signature_data, constraint_data)
        self.target = target
        assert len(self.target) == 1, "Target should only have one key"
        self.target_point = mutation_point
        self.param_category = param_category
        self.alignments = alignments

    def visit_Call(self, node):
        if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
            # analyze the current parameters
            func_args = _fetch_args_value(node.args)
            func_keyword_args = _fetch_args_value(node.keywords)
            self.alignments = signature_alignments(self.alignments, func_args, func_keyword_args, self.param_category)
            # print the alignment
            logger.debug(f"Function alignment: {_parse_alignments(self.api_full_name,self.alignments)}")

        self.generic_visit(node)

    def conduct_mutation(self, source_code, constraints, framework, param_category):
        tree = ast.parse(source=source_code, mode='exec')
        self.param_category = param_category
        # get the function alignment
        self.visit(tree)
        mutant_cnt = 0
        # get the list of input_names
        input_names = list(self.alignments.keys())
        iter_ptr = tik_cnt = 0
        attempts = max(MAX_ATTEMPTS, len(input_names))
        # since our mutation has randomness, we randomly try 5 times
        while tik_cnt < attempts:
            if iter_ptr >= len(input_names):
                # reset the iter_ptr
                iter_ptr = 0
            input_name = input_names[iter_ptr]
            copied_tree = ast.parse(source=source_code, mode='exec')
            # re-get the
            self.visit(copied_tree)
            # we need to copy the tree to avoid high-order mutation
            for node in ast.walk(copied_tree):
                if isinstance(node,ast.Call) and self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                    call_code = ast.unparse(node).strip()
                    # mutate the code
                    logger.debug(f"Code: Original {call_code} ")
                    if input_name.startswith("tmp_arg"):
                        constraint = {}
                    else:
                        if input_name not in constraints["constraints"].keys():
                            logger.error(f"Can't find the constraints for {input_name} from {self.api_full_name}")
                            continue
                        constraint = constraints["constraints"][input_name]
                    new_node = self.mutate(node, input_name, constraint,framework)
                    new_call_code = ast.unparse(new_node).strip()
                    logger.debug(f"Code: Mutation {mutant_cnt}: {new_call_code}")
                    if new_call_code != call_code:
                        return astor.to_source(copied_tree)
            iter_ptr += 1
            tik_cnt += 1
        return None

    def mutate(self, node, input_name, constraint, framework):
        align_tuple = self.alignments[input_name]
        # we initialize the parameter that is not set
        if isinstance(align_tuple, str) and align_tuple == "N/A":
            # generate new value by the constraint
            new_value = construct_api_value(constraint, framework)
            if new_value:
                # construct a node
                arg_node = create_node(new_value)
                new_kwarg = ast.keyword(arg=input_name, value=arg_node)
                node.keywords.append(new_kwarg)
        else:
            arg_obj, arg_type, arg_idx = align_tuple
            # check whether the default value exists
            # we have 30% chance to change it to default
            if "default" in constraint.keys() and constraint["default"] == 'None' and random.random() < 0.3:
                if isinstance(arg_obj, ast.keyword):
                    arg_obj.value = ast.Constant(value=None)
                    node.keywords[arg_idx] = arg_obj
                else:
                    arg_obj = ast.Constant(value=None)
                    node.args[arg_idx] = arg_obj
            else:
                # mutate the arg, according to their type
                val_range = constraint["range"] if "range" in constraint.keys() else None
                val_range = random.choice(val_range) if isinstance(val_range, list) else val_range
                # if the arg_obj is a list, we randomly mutate one element
                if isinstance(arg_obj, list):
                    arg_idx = random.randint(0, len(arg_obj) - 1)
                    arg_obj[arg_idx] = self.mutate_api_value(arg_obj[arg_idx], val_range, constraint, framework)
                else:
                    arg_obj = self.mutate_api_value(arg_obj, val_range, constraint, framework)
                if arg_type == "args":
                    if isinstance(arg_obj,list):
                        node.args = arg_obj
                    else:
                        node.args[arg_idx] = arg_obj
                elif arg_type == "keywords":
                    node.keywords[arg_idx] = arg_obj

        return node

