import os
import random

import yaml
import ast
import astor
from src import FUNC_SIG_PATH, DOCTER_PATH, root_path
from src.tools import utils
from src.mutation.base import mutation_utils,api_utils
from src.tools.enum_class import Framework
from src.tools.utils import AnalysisUtils
from src.tools.logger_utils import LoggerUtils
from collections import defaultdict
from src.mutation.base.api_utils import APIParamModifier

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


def categorize_parameters(input_param_dict):
    alignments = {}
    category = defaultdict(lambda: defaultdict(list))
    for param_dict in input_param_dict:
        param_name = param_dict["name"]
        alignments[param_name] = "N/A"
        param_require = param_dict["require"] if "require" in param_dict.keys() else False
        param_style = param_dict["style"] if "style" in param_dict.keys() else "positional"

        if param_name.startswith("*"):
            category['positional']["star"].append(param_name)
        else:
            param_require_type = "require" if param_require else "optional"
            category[param_style][param_require_type].append(param_name)
    return alignments, category


class APIMutator:
    def __init__(self, **kwargs):
        self.framework = kwargs.get("framework", None)
        self.supported_apis = kwargs.get("supported_apis", None)

    @staticmethod
    def api_decorator_scan(tree):
        def has_abnormal_decorator(decorator_list):
            for dec in decorator_list:
                if not isinstance(dec, ast.Name):
                    return True
                else:
                    if dec.id not in ["staticmethod", "classmethod", "property", "abstractmethod"]:
                        return True
            else:
                return False

        mutation_points = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Remove annotation from function definition
                decorator_list = node.decorator_list
                if isinstance(decorator_list, list) and len(decorator_list) > 0 and has_abnormal_decorator(
                        decorator_list):
                    mutation_points.append({"target": node.name,
                                            "point": (node.lineno, node.end_lineno,
                                                      node.col_offset, node.end_col_offset)})
        return mutation_points

    @staticmethod
    def api_decorator_mutation(source_code, mutation_point_item):
        slineno, elineno, col_offset, ecol_offset = mutation_point_item["point"]
        target = mutation_point_item["target"]
        tree = ast.parse(source=source_code, mode='exec')

        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == target
                    and node.lineno == slineno
                    and node.end_lineno == elineno
                    and node.col_offset == col_offset
                    and node.end_col_offset == ecol_offset):
                # remove the decorators
                node.decorator_list = []
                break
        # save the mutation
        return astor.to_source(tree)

    def get_available_api_calls(self, source_code):
        if self.framework == Framework.TENSORFLOW:
            targets = ["tensorflow"]
        elif self.framework == Framework.PYTORCH:
            targets = ["torch","onnx"]
        elif self.framework == Framework.JITTOR:
            targets = ["jittor"]
        elif self.framework == Framework.MXNET:
            targets = ["mxnet"]
        elif self.framework == Framework.MINDSPORE:
            targets = ["mindspore"]
        elif self.framework == Framework.PADDLE:
            targets = ["paddle"]
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        # Obtain all mutable APIs in test cases
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets)
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        invoke_name2full_name = {}
        available_api_calls = []
        for api_call in api_calls:
            if api_call["full_name"] in self.supported_apis:
                if api_call["alias_library"]:
                    invoke_name2full_name[f'{api_call["alias_library"]}.{api_call["call_name"]}'] = api_call[
                        "full_name"]
                    available_api_calls.append({api_call["alias_library"]: api_call["call_name"]})
                else:
                    invoke_name2full_name[api_call["call_name"]] = api_call["full_name"]
                    available_api_calls.append({api_call["call_name"]: ""})
        # logger.debug(f"available_api_calls: {available_api_calls}")
        return available_api_calls, invoke_name2full_name

    def api_invocation_scan(self, source_code, targets):
        if self.framework == Framework.TENSORFLOW:
            from src.mutation.tensorflow import tensorflow_utils as framework_utils
        elif self.framework == Framework.PYTORCH:
            from src.mutation.pytorch import pytorch_utils as framework_utils
        elif self.framework == Framework.JITTOR:
            from src.mutation.jittor import jittor_utils as framework_utils
        elif self.framework == Framework.MXNET:
            from src.mutation.mxnet import mxnet_utils as framework_utils
        elif self.framework == Framework.MINDSPORE:
            from src.mutation.mindspore import mindspore_utils as framework_utils
        elif self.framework == Framework.PADDLE:
            from src.mutation.paddle import paddle_utils as framework_utils
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        tree = ast.parse(source_code, mode='exec')
        visitor = framework_utils.FunctionCallVisitor(targets)
        visitor.visit(tree)
        return visitor.mutation_points

    def general_api_invocation_scan(self, source_code):
        def check_keyword_value(k):
            if isinstance(k.value, ast.Constant):
                return True
            else:
                return False
        # we need to scan the api calls in the source code and get the keyword position
        tree = ast.parse(source_code, mode='exec')
        mutation_points = []
        for node in ast.walk(tree):
            # if node is call, we get the position of each arg
            if isinstance(node, ast.Call):
                # for arg in node.args:
                #     position = (arg.lineno, arg.col_offset)
                #     mutation_points.append({"target": node.func, "point": position, "type": "arg"})
                for keyword in node.keywords:
                    if check_keyword_value(keyword):
                        position = (keyword.lineno, keyword.col_offset)
                        mutation_points.append({"target": node.func, "point": position, "type": "keyword"})
        return mutation_points

    def api_replace_mutation(self,source_code, mutation_point_item, invoke_name2full_name):
        raise NotImplementedError("api_replace_mutation is not implemented for BaseMutator")

    def api_parameter_mutation(self, source_code, mutation_point_item, invoke_name2full_name):
        tree = ast.parse(source=source_code, mode='exec')
        mutation_point = mutation_point_item["point"]
        target = mutation_point_item["target"]
        target_tuple = list(target.items())[0]
        invoke_name = target_tuple[0] if target_tuple[1] == "" else ".".join(list(target.items())[0])
        if invoke_name not in invoke_name2full_name.keys():
            return astor.to_source(tree)
        else:
            api_full_name = invoke_name2full_name[invoke_name]

            # # get the function signature
            signature_data = yaml.safe_load(
                utils.read_text(os.path.join(root_path, FUNC_SIG_PATH[self.framework], f"{api_full_name}.yaml")))
            constraint_data = yaml.safe_load(
                utils.read_text(os.path.join(root_path, DOCTER_PATH[self.framework], f"{api_full_name}.yaml".lower())))
            para_alignments, param_category = categorize_parameters(signature_data["input_params"])
            finder = APIParamModifier(target, invoke_name, api_full_name, signature_data, constraint_data,
                                      mutation_point, para_alignments, param_category)

            mutated_code = finder.conduct_mutation(source_code, constraint_data, framework=self.framework,
                                                   param_category=param_category)
            return mutated_code

    def general_api_parameter_mutation(self, source_code, mutation_point_item):
        def change_general_value(keyword):
            # if the value is an int/float constant, we change it to another value
            if isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, int):
                    special_values = {0, 1, -1}
                    if keyword.value.value in special_values:
                        # we set to another value
                        keyword.value.value = random.choice(list(special_values - {keyword.value.value}))
                    else:
                        keyword.value.value = random.randint(0, 100)
                elif isinstance(keyword.value.value, float):
                    special_values = {0., 1., -1.}
                    if keyword.value.value in special_values:
                        # we set to another value
                        keyword.value.value = random.choice(list(special_values - {keyword.value.value}))
                    else:
                        keyword.value.value = random.uniform(0, 100)
            return keyword.value

        mutation_point = mutation_point_item["point"]
        tree = ast.parse(source_code, mode='exec')
        for node in ast.walk(tree):
            # if node is call, we get the position of each arg
            if isinstance(node, ast.Call):
                # for arg in node.args:
                #     position = (arg.lineno, arg.col_offset)
                #     mutation_points.append({"target": node.func, "point": position, "type": "arg"})
                for keyword_arg in node.keywords:
                    if mutation_point == (keyword_arg.lineno, keyword_arg.col_offset):
                        print(ast.unparse(node))
                        if hasattr(keyword_arg, "value"):
                            keyword_arg.value = change_general_value(keyword_arg)
                        # mutate the keyword
        return astor.to_source(tree)


    def mutation_point_scan(self, source_code, mode=None):
        # we have four kinds of api mutation points
        # api_parameter, api_replace, api_decorator, keyword_removing
        mutation_points = []
        # get the mutation points of api_parameter and api replace
        # get the available api calls
        available_api_calls, _ = self.get_available_api_calls(source_code)
        # get the mutation points
        if mode in ["api_parameter", "api_replace"]:
            if len(available_api_calls) > 0:
                mutation_items = self.api_invocation_scan(source_code, available_api_calls)
                for item in mutation_items:
                    mutation_item = item.copy()
                    if mode == "api_parameter":
                        mutation_item["mode"] = "api_parameter"
                    else:
                        mutation_item["mode"] = "api_replace"
                    mutation_points.append(mutation_item)
            else:
                # conduct the general mutation: 1) replace keyword parameter
                mutation_items = self.general_api_invocation_scan(source_code)
                for mutation_item in mutation_items:
                    mutation_item["mode"] = "general_api_parameter"
                    mutation_points.append(mutation_item)

        elif mode == "api_decorator":
            tree = ast.parse(source=source_code, mode='exec')
            mutation_items = self.api_decorator_scan(tree)
            for item in mutation_items:
                mutation_item = item.copy()
                mutation_item["mode"] = "api_decorator"
                mutation_points.append(mutation_item)
        elif mode == "keyword_removing":
            tree = ast.parse(source=source_code, mode='exec')
            visitor = mutation_utils.RemoveAPIKeyword()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({"mode":'keyword_removing', "point":point})
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")
        return mutation_points

    def mutate(self, source_code, mutation_point_item):
        """
        mutate_mode:
        - api_parameter: change the parameter of api
        - api_replace: replace the api
        - api_decorator: add or remove the decorator
        - keyword_removing: remove the keyword
        """
        _, invoke_name2full_name = self.get_available_api_calls(source_code)
        mutate_mode, mutation_point = mutation_point_item['mode'], mutation_point_item['point']

        if mutate_mode == "api_parameter":
            return self.api_parameter_mutation(source_code, mutation_point_item, invoke_name2full_name)
        elif mutate_mode == "general_api_parameter":
            return self.general_api_parameter_mutation(source_code, mutation_point_item)
        elif mutate_mode == "api_replace":
            return self.api_replace_mutation(source_code, mutation_point_item, invoke_name2full_name)
        elif mutate_mode == "api_decorator":
            return self.api_decorator_mutation(source_code, mutation_point_item)
        elif mutate_mode == "keyword_removing":
            tree = ast.parse(source=source_code, mode='exec')
            visitor = mutation_utils.RemoveAPIKeyword()
            visitor.mutate(tree, mutation_point)
            return astor.to_source(tree)
        else:
            raise ValueError(f"Unknown mutation mode: {mutate_mode}")

    def get_unused_api_calls(self, source_code):
        _,invoke_name2full_name = self.get_available_api_calls(source_code)
        # get the api calls in supported_apis but not in available_api_calls
        total_full_name = set(invoke_name2full_name.values())
        unused_api_calls = set(self.supported_apis) - total_full_name
        return list(unused_api_calls)

    def generate_api_call(self,candidate_api):

        signature_path = os.path.join(root_path, FUNC_SIG_PATH[self.framework], f"{candidate_api}.yaml")
        signature_data = yaml.safe_load(utils.read_text(signature_path))
        constraint_path = os.path.join(root_path, DOCTER_PATH[self.framework], f"{candidate_api}.yaml".lower())
        constraint_data = yaml.safe_load(utils.read_text(constraint_path))

        require_param_dtype_dict = {}
        for signature in signature_data['input_params']:
            if signature['require']:
                param_name = signature['name']
                if param_name not in constraint_data['constraints'] or 'dtype' not in constraint_data['constraints'][param_name]:
                    logger.warn(f"Failed to parse the constraint data for the parameter: {param_name} for api {candidate_api}")
                    return None, False
                dtype_name = constraint_data['constraints'][param_name]['dtype']
                # for the torch.Tensor, we do not need the input parameter
                if param_name == "input" and "torch.Tensor" in candidate_api:
                    continue
                require_param_dtype_dict[param_name] = dtype_name

        api_call_str = f"{candidate_api}("
        for param_name,para_type in require_param_dtype_dict.items():
            api_call_str,success = api_utils.generated_basic_type(api_call_str,param_name,para_type)
            if not success:
                return None,False
        else:
            api_call_str = api_call_str.rstrip()[:-1] if api_call_str.rstrip().endswith(",") else api_call_str
            api_call_str += ")"
            if 'torch.Tensor' in candidate_api and candidate_api!= "torch.Tensor":
                api_call_str = api_call_str.replace("torch.Tensor","torch.randn(1,2,3)")
            return api_call_str, True









