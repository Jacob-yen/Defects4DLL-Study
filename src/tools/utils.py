from collections import defaultdict
import json
import os
import sys
import random
from src import root_path, FUNC_SIG_PATH

sys.path.append(os.getcwd())
import re
import astunparse
from numpy.linalg import norm
import pickle
import numpy as np
import ast
import hashlib
from src.tools.logger_utils import LoggerUtils
from src.tools.enum_class import Framework
from src.mutation.llm import feedback_template


def check_syntax_error(code):
    # try:
    #     tree = ast.parse(code, mode='exec')
    #     return True
    # except SyntaxError:
    #     return False
    try:
        compile(code, '<string>', 'exec')
        print("mutated code is valid!")
        return True
    except SyntaxError as e:
        return False

def assertion_points(tree):
    assertions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.If,
                                                                ast.While, ast.With, ast.Try, ast.ExceptHandler,
                                                                ast.AsyncFunctionDef)):
            lineno = getattr(node, 'lineno', None)
            end_lineno = getattr(node, 'end_lineno', None)
            if lineno is not None:
                code_line = astunparse.unparse(node)
                if CodeUtils.has_assertion(code_line):
                    print(code_line)
                    assertions.append((lineno, end_lineno, code_line.strip()))
    return assertions


def extract_all_traces(ds_path):
    logger_instance = LoggerUtils.get_instance()
    logger = logger_instance.logger
    # get all traces from the dataset
    # get path of all bug
    bug_ids = os.listdir(ds_path)
    # get content of each trace
    traces = dict()
    for bug_id in bug_ids:
        trace_path = os.path.join(ds_path, bug_id, "stack_trace.txt")
        if os.path.exists(trace_path):
            with open(trace_path, "r") as f:
                trace_content = f.read()
                traces[bug_id] = trace_content
        else:
            logger.error("Trace file not found: {}".format(bug_id))
    return traces


def extract_all_tests(ds_path):
    logger_instance = LoggerUtils.get_instance()
    logger = logger_instance.logger
    # get all traces from the dataset
    # get path of all bug
    bug_ids = os.listdir(ds_path)
    # get content of each trace
    tests = dict()
    for bug_id in bug_ids:
        test_path = os.path.join(ds_path, bug_id, f"{bug_id}.py")
        if os.path.exists(test_path):
            with open(test_path, "r") as f:
                test_content = f.read()
                tests[bug_id] = test_content
        else:
            logger.error("Test file not found: {}".format(bug_id))
    return tests


def save_pkl(obj, f):
    with open(f, "wb") as fw:
        pickle.dump(obj, fw)


def read_pkl(f):
    with open(f, "rb") as fr:
        return pickle.load(fr)


def read_text(f):
    with open(f, "r") as fr:
        return fr.read()


def write_text(f, content):
    with open(f, "w") as fw:
        fw.write(content)


def is_test_case_name(file_name):
    pattern = r'^(torch|tf)[-|_]\d+\.py$'
    return re.match(pattern, file_name) is not None


def get_python_version(string):
    pattern = r'python\d\.\d+'
    # search and return the match object
    matches = re.findall(pattern, string)
    # if matches is not empty, return the matched Result
    if matches:
        return matches[0]
    else:
        return None


def get_call_stack(text):
    pattern1 = r'\s+File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+([^ ]+)'
    matches1 = re.findall(pattern1, text)
    file_func_list = []
    for match in matches1:
        file_path, line_number, function_name = match
        function_name = function_name.strip("\n")
        if function_name == "<module>":
            continue
        # if the file is a test case file, we only print the function name
        if is_test_case_name(file_path):
            file_name = file_path
        else:
            ver = get_python_version(file_path)
            if ver:
                # we spit the file path by the python version
                # we first get the python version
                file_name = file_path.split(ver)[-1]
                file_name = file_name[1:] if file_name.startswith("/") else file_name
                # if the file path starts with 'site-packages', we remove it
                if file_name.startswith("site-packages/"):
                    file_name = file_name.split("site-packages/")[-1]
            else:
                # we get the file name of the path
                file_name = file_path.split("/")[-1]
        file_func_list.append((file_name, function_name))
        # print("File: {}, Function: {}".format(file_name, function_name))
    return file_func_list


def get_error_message(text):
    pattern2 = r'(^[a-z|A-Z]*Error.*$)|(^.*\s\[.*Error\].*$)'
    matches2 = re.findall(pattern2, text, re.MULTILINE)
    for match in matches2:
        error_line = match[0] or match[1]
        return error_line
    else:
        return None


def parse_exception(text):
    # get the call stack
    file_func_list = get_call_stack(text)
    # get the error message
    error_line = get_error_message(text)
    return file_func_list, error_line


class CodeUtils:

    @staticmethod
    def split_code_lines(code_text):
        tree = ast.parse(code_text, mode='exec')
        code_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) \
                    or isinstance(node, ast.AnnAssign) \
                    or isinstance(node, ast.AugAssign) \
                    or isinstance(node, ast.Expr):
                start_line_no = node.lineno
                end_line_no = node.end_lineno

                statement_lines = code_text.splitlines()[start_line_no - 1:end_line_no]
                statement = "\n".join(statement_lines)
                statement += "\n"
                code_lines.append((start_line_no, end_line_no, statement))
        # remove comments

        for sid, eid, line_item in code_lines:
            if line_item.strip().startswith("#") \
                    or line_item.strip().startswith("'''") \
                    or line_item.strip().startswith('"""'):
                code_lines.remove((sid, eid, line_item))
        return code_lines

    @staticmethod
    def has_assertion(code_line):
        # a vague regex expr
        pattern = r".*assert\s*.*?|.*assert_(.*?)|.*self.check|.*gradcheck|.*assert*"
        matches = re.findall(pattern, code_line, re.IGNORECASE)
        if matches:
            return True
        else:
            return False

    @staticmethod
    def identify_assign(code_line):
        logger_instance = LoggerUtils.get_instance()
        logger = logger_instance.logger
        code_line = code_line.lstrip()
        tree = ast.parse(code_line, mode='exec')
        if len(tree.body) > 0:
            root_node = tree.body[0]
            if isinstance(root_node, ast.Assign) or \
                    isinstance(root_node, ast.AnnAssign) or \
                    isinstance(root_node, ast.AugAssign):
                # we only get the assignment contains constant values
                for node in ast.walk(root_node):
                    if isinstance(node, ast.Constant):
                        return True
                else:
                    return False

            else:
                return False
        else:
            logger.info(code_line)
            raise Exception("body length less than zero")


class LLMBot:
    def __init__(self, api_base,api_key, model, system_prompt, temperature=None) -> None:
        self.api_base = api_base
        self.api_key=api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature if temperature else 0
        self.history = []
        logger_instance = LoggerUtils.get_instance()
        self.logger = logger_instance.logger

    def chat_completion(self, prompt, api_key="EMPTY"):
        from openai import OpenAI
        # openai.api_key = api_key
        # openai.api_base = self.api_base
        if self.model == "gpt-3.5-turbo":
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # elif "deepseek" in self.model:
        #     client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        if len(self.history) == 3:
            self.clear_history()
        else:
            for talk_history in self.history:
                history_response = talk_history["response"]
                history_exception = talk_history["exception"]
                messages.append({"role": "assistant", "content": history_response})
                messages.append({"role": "user", "content": feedback_template.format(history_exception)})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        self.logger.info(self.parse_prompt(messages))
        return response.choices[0].message.content

    def parse_prompt(self, messages):
        prompt = ""
        assistant_cnt = 1
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant-{assistant_cnt}: {content}\n"
                assistant_cnt += 1
        return prompt


    def update_history(self, response, exception):
        self.history.append({"response": response, "exception": exception})

    def clear_history(self):
        self.history = []
    def chatgpt_completion(self, messages):

        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://openai.repus.cn/v1")
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
        )
        return response.choices[0].message.content

    @staticmethod
    def parse_response(response):
        # only get the code between ```python and ```
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0]
        else:
            return None


def md5(content=None):
    if content is None:
        return ''
    else:
        md5gen = hashlib.md5()
        md5gen.update(content.encode())
        md5code = md5gen.hexdigest()
        return md5code


class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.definitions = {}
        self.usages = []

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.definitions[target.id] = node.value
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.usages.append(node.id)
        self.generic_visit(node)


class LibImportVisitor(ast.NodeVisitor):
    """Visitor that collects all library imports and aliases."""

    def __init__(self, targets):
        self.target = targets
        self.libraries_alias = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith(tuple(self.target)):
                self.libraries_alias.append((alias.name, alias.asname or alias.name))

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith(tuple(self.target)):
            for alias in node.names:
                self.libraries_alias.append((f"{node.module}.{alias.name}", alias.asname or alias.name))

        self.generic_visit(node)


class APICallVisitor(ast.NodeVisitor):
    """Get the API calls from the given libraries."""

    def __init__(self, targets):
        self.target_libraries = targets
        self.api_calls = []
        self.tensor_api_calls = []

    def visit_ClassDef(self, node):
        if hasattr(node, 'bases'):
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    # situation 2: with library name and function name
                    # from torch import nn
                    # nn.Linear(3,3)
                    library_name = self._get_library_name(base)
                    if library_name and library_name.startswith(tuple(self.target_libraries)):
                        api_name = self._get_api_name(base)
                        self.api_calls.append(
                            {"alias_library": library_name, "call_name": api_name, "lineno": node.lineno,
                             "end_lineno": node.end_lineno})
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.target_libraries:
            # situation 1: only with function name
            # from torch import randn
            # randn(2,3)
            self.api_calls.append({"alias_library": None, "call_name": node.func.id, "lineno": node.lineno,
                                   "end_lineno": node.end_lineno})

        if isinstance(node.func, ast.Attribute):
            # situation 2: with library name and function name
            # from torch import nn
            # nn.Linear(3,3)
            library_name = self._get_library_name(node.func)
            if library_name and library_name.startswith(tuple(self.target_libraries)):
                api_name = self._get_api_name(node.func)
                self.api_calls.append({"alias_library": library_name, "call_name": api_name, "lineno": node.lineno,
                                       "end_lineno": node.end_lineno})
            elif 'torch' in self.target_libraries:
                api_name = f"torch.Tensor.{node.func.attr}"
                tensor_api_list = []
                yaml_list = os.listdir(os.path.join(root_path, FUNC_SIG_PATH[Framework.PYTORCH]))
                for yaml in yaml_list:
                    if 'torch.Tensor.' in yaml:
                        tensor_api_list.append(yaml.replace(".yaml", ""))
                if api_name in tensor_api_list:
                    self.tensor_api_calls.append(
                        {"alias_library": 'torch', "call_name": f"Tensor.{node.func.attr}", "lineno": node.lineno,
                         "end_lineno": node.end_lineno})

        for arg in node.args:
            if isinstance(arg, ast.Attribute):
                # situation 2: with library name and function name
                # from torch import nn
                # nn.Linear(3,3)
                library_name = self._get_library_name(arg)
                if library_name and library_name.startswith(tuple(self.target_libraries)):
                    api_name = self._get_api_name(arg)
                    self.api_calls.append({"alias_library": library_name, "call_name": api_name, "lineno": node.lineno,
                                           "end_lineno": node.end_lineno})

        self.generic_visit(node)

    def _get_library_name(self, node):
        if isinstance(node.value, ast.Name):
            return node.value.id
        elif isinstance(node.value, ast.Attribute):
            return self._get_library_name(node.value)

    def _get_api_name(self, node):
        if isinstance(node.value, ast.Name):
            return node.attr
        elif isinstance(node.value, ast.Attribute):
            return self._get_api_name(node.value) + "." + node.attr


class AnalysisUtils:
    def __init__(self):
        pass

    @staticmethod
    def find_variable_definitions_and_usages(source_code):
        tree = ast.parse(source_code)
        visitor = VariableVisitor()
        visitor.visit(tree)
        return visitor.definitions, visitor.usages

    @staticmethod
    def extract_lib_alias(source_code, targets: list):
        """this code is used to get the alias of torch module
            examples: 
            import torch -> torch
            import torch.nn as nn -> nn
            for torch import nn -> nn
            from torch import nn as mynn -> mynn


        Args:
            source_code (_type_): _description_
        Returns:
            a list of imported packages and their aliases
            [(torch, torch), (torch.nn,nn)]
        """
        tree = ast.parse(source_code, mode='exec')
        visitor = LibImportVisitor(targets)
        visitor.visit(tree)
        return visitor.libraries_alias

    @staticmethod
    def extract_api_calls(source_code, targets):
        """this code is used to get the alias of torch module

        Args:
            source_code (_type_): _description_
            target_libraries (_type_): _description_

        Returns:
            _type_: _description_
        """
        alias_lib_mapping = {t[1]: t[0] for t in targets}
        alias = list(alias_lib_mapping.keys())
        tree = ast.parse(source_code)
        visitor = APICallVisitor(alias)
        visitor.visit(tree)
        api_calls = visitor.api_calls
        # return api_calls

        # concatenate the api calls and the alias
        apis = []
        for api_call in api_calls:

            if api_call["alias_library"]:
                root_package = alias_lib_mapping[api_call["alias_library"]]
                func_call = api_call["call_name"]
                # the full name and the func name in code
                apis.append(f"{root_package}.{func_call}")
            else:
                root_package = alias_lib_mapping[api_call["call_name"]]
                apis.append(root_package)
        return apis

    @staticmethod
    def extract_api_calls_v2(source_code, targets):
        """this code is used to get the alias of torch module

        Args:
            source_code (_type_): _description_
            target_libraries (_type_): _description_

        Returns:
            _type_: _description_
        """
        alias_lib_mapping = {t[1]: t[0] for t in targets}
        alias = list(alias_lib_mapping.keys())
        tree = ast.parse(source_code)
        visitor = APICallVisitor(alias)
        visitor.visit(tree)
        api_calls = visitor.api_calls
        tensor_api_calls = visitor.tensor_api_calls
        api_calls.extend(tensor_api_calls)

        # concatenate the api calls and the alias
        apis = []
        # api_call a tuple of (alias_lib_name, func_name)
        # if len(api_call) == 1, it means we only have the func_name
        for api_call in api_calls:
            if api_call["alias_library"] and api_call["alias_library"] in alias_lib_mapping.keys():
                root_package = alias_lib_mapping[api_call["alias_library"]]
                func_call = api_call["call_name"]
                # the full name and the func name in code
                # apis.append(f"{root_package}.{func_call}")
                apis.append({"full_name": f"{root_package}.{func_call}",
                             "alias_library": api_call["alias_library"],
                             "call_name": api_call["call_name"]})
            elif api_call["call_name"] in alias_lib_mapping.keys():
                root_package = alias_lib_mapping[api_call["call_name"]]
                apis.append({"full_name": root_package,
                             "alias_library": api_call["alias_library"],
                             "call_name": api_call["call_name"]})
        return apis

    @staticmethod
    def extract_api_calls_v3(source_code, targets):
        """this code is used to get the alias of torch module

        Args:
            source_code (_type_): _description_
            target_libraries (_type_): _description_

        Returns:
            _type_: _description_
        """
        alias_lib_mapping = {t[1]: t[0] for t in targets}
        alias = list(alias_lib_mapping.keys())
        tree = ast.parse(source_code)
        visitor = APICallVisitor(alias)
        visitor.visit(tree)
        api_calls = visitor.api_calls

        # concatenate the api calls and the alias
        apis = []
        # api_call a tuple of (alias_lib_name, func_name)
        # if len(api_call) == 1, it means we only have the func_name
        for api_call in api_calls:
            if api_call["alias_library"]:
                if api_call["alias_library"] not in alias_lib_mapping.keys():
                    continue
                root_package = alias_lib_mapping[api_call["alias_library"]]
                func_call = api_call["call_name"]
                # the full name and the func name in code
                # apis.append(f"{root_package}.{func_call}")
                apis.append({"full_name": f"{root_package}.{func_call}",
                             "alias_library": api_call["alias_library"],
                             "call_name": api_call["call_name"],
                             "lineno": api_call["lineno"],
                             "end_lineno": api_call["end_lineno"]})
            else:
                root_package = alias_lib_mapping[api_call["call_name"]]
                apis.append({"full_name": root_package,
                             "alias_library": api_call["alias_library"],
                             "call_name": api_call["call_name"],
                             "lineno": api_call["lineno"],
                             "end_lineno": api_call["end_lineno"]})
        return apis


class DataType:

    def __init__(self, value, name=None, lineno=None):
        self.value = value
        self.var_name = name
        self.lineno = lineno
        assert type(value).__name__ in DataType.supported_types(), f"Unsupported type {type(value).__name__}, {value}"
        self.type = type(value).__name__
        self.wrapped_elements = self.wrap_elements()

    @staticmethod
    def supported_types():
        return ['list', 'tuple', 'dict', 'set', 'int', 'float', 'str', 'bool', 'NoneType', 'Tensor']

    def __repr__(self):
        # recursively print all the datatype in the value
        if self.type in ['int', 'float', 'str', 'bool', 'NoneType']:
            return f"{self.var_name}:{self.type} = {self.value}"
        elif self.type == 'Tensor':
            return f"{self.var_name}:Tensor(shape={self.value.shape}, dtype={self.value.dtype})"
        elif self.type == 'list':
            # we should print each datatype in the list
            return f"{self.var_name}:list = [{', '.join([repr(item) for item in self.wrapped_elements])}]"
        elif self.type == 'tuple':
            return f"{self.var_name}:tuple = ({', '.join([repr(item) for item in self.wrapped_elements])})"
        elif self.type == 'dict':
            return f"{self.var_name}:dict = {{{', '.join([f'{k}: {v}' for k, v in self.wrapped_elements.items()])}}}"
        elif self.type == 'set':
            return f"{self.var_name}:set = {{{', '.join([repr(item) for item in self.wrapped_elements])}}}"
        else:
            return None

    def __str__(self):
        # recursively print all the datatype in the value
        if self.type in ['int', 'float', 'str', 'bool', 'NoneType']:
            return f"{self.var_name}:{self.type} = {self.value}"
        elif self.type == 'Tensor':
            return f"{self.var_name}:Tensor(shape={self.value.shape}, dtype={self.value.dtype})"
        elif self.type == 'list':
            # we should print each datatype in the list
            return f"{self.var_name}:list = [{', '.join([repr(item) for item in self.wrapped_elements])}]"
        elif self.type == 'tuple':
            return f"{self.var_name}:tuple = ({', '.join([repr(item) for item in self.wrapped_elements])})"
        elif self.type == 'dict':
            return f"{self.var_name}:dict = {{{', '.join([f'{k}: {v}' for k, v in self.wrapped_elements.items()])}}}"
        elif self.type == 'set':
            return f"{self.var_name}:set = {{{', '.join([repr(item) for item in self.wrapped_elements])}}}"
        else:
            return None

    def wrap_elements(self):
        if self.type == 'list':
            return [DataType(item, name=f"{self.var_name}_inner{idx}") for idx, item in enumerate(self.value)]
        elif self.type == 'tuple':
            return (DataType(item, name=f"{self.var_name}_inner{idx}") for idx, item in enumerate(self.value))
        elif self.type == 'dict':
            return {DataType(k, name=f"{self.var_name}_inner_k{idx}"): DataType(v, name=f"{self.var_name}_inner_v{idx}")
                    for idx, (k, v) in enumerate(self.value.items())}
        elif self.type == 'set':
            return {DataType(item, name=f"{self.var_name}_inner{idx}") for idx, item in enumerate(self.value)}
        elif self.type in ['int', 'float', 'str', 'bool', 'NoneType', 'Tensor']:
            return self.value
        else:
            return None


class CodeWrapper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def extract_import_statements(code):
        tree = ast.parse(code, mode='exec')
        import_lines = []
        code_splits = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                start_line = node.lineno
                end_line = node.end_lineno
                if start_line == end_line:
                    import_lines.append((start_line - 1, end_line))
                else:
                    import_lines.append((start_line - 1, end_line))
        import_statements = []
        for import_line in import_lines:
            import_line_start, import_line_end = import_line
            import_line = code_splits[import_line_start:import_line_end]
            import_line = '\n'.join(import_line)
            import_statements.append(import_line)
        return import_statements

    # @staticmethod
    # def wrap_code(source_code,mode='var'):
    #     import_prefix = """import os\nimport pickle\nimport types\nimport sys\nsys.path.append(os.getcwd())\nimport inspect\nimport unittest\nfrom approach.utils import DataType, md5"""
    #     # get import statements
    #     import_statements = CodeWrapper.extract_import_statements(source_code)
    #     imports = '\n'.join(import_statements)

    #     # we need to delete the import statements from the code
    #     # for each import statement, we replace it with ""
    #     for import_statement in import_statements:
    #         source_code = source_code.replace(import_statement + "\n", "")

    #     # split the code into lines
    #     lines = source_code.split('\n')
    #     # remove the empty lines
    #     lines = [line for line in lines if line.strip() != '']
    #     # add indents
    #     indented_lines = [' ' * 4 + line for line in lines]
    #     # add the function definition in the front
    #     indented_lines.insert(0, 'def top_function_executor():')
    #     indented_code = '\n'.join(indented_lines)
    #     trace_prefix =  read_text(f'approach/{mode}_trace_code.txt')
    #     # read the suffix code
    #     suffix_code = read_text(f'approach/{mode}_suffix_code.txt')

    #     # wrap the code
    #     wrapped_code = f"{import_prefix}\n{imports}\n{trace_prefix}\n{indented_code}\n{suffix_code}"
    #     return wrapped_code


def gen_md5_id(s, is_16_bit=True):
    md5_machine = hashlib.md5()
    md5_machine.update(s.encode('utf-8'))
    return md5_machine.hexdigest()[9:25] if is_16_bit else md5_machine.hexdigest()


def get_func_code_by_name(file_path, class_name, method_name):
    logger_instance = LoggerUtils.get_instance()
    logger = logger_instance.logger
    with open(file_path, 'r') as file:
        file_code = file.read()
    try:
        tree = ast.parse(file_code)
        class_def_node = next((node for node in ast.walk(tree)
                               if isinstance(node, ast.ClassDef) and
                               node.name == class_name), None)
        if class_def_node:
            method_def_node = next((node for node in class_def_node.body if
                                    isinstance(node, ast.FunctionDef) and
                                    node.name == method_name), None)

            if method_def_node:
                # 生成去掉缩进的源代码
                source_without_indent = astunparse.unparse(method_def_node)
                return source_without_indent
    except Exception as e:
        logger.error(f"fail to get function {method_name} from class {class_name} in file {file_path}")
        logger.error(str(e))
    return f"fail to get function {method_name} from class {class_name} in file {file_path}"


def parse_coverage(coverage_obj):
    # get the values of key "files"
    files = coverage_obj["files"]
    file_names = list(files.keys())
    file_names.sort()
    coverage_dict = dict()
    vector = []
    # for each file, get the coverage
    for file_name in file_names:
        # get the executed lines
        executed_lines = files[file_name]["executed_lines_frequency"]
        executed_lines = list(executed_lines.keys())
        # get the keys of executed_lines_frequency, and sort the numbers by their integer value
        executed_lines.sort(key=lambda x: int(x))
        coverage_dict[file_name] = executed_lines
        vector.extend([1] * len(executed_lines))
    return np.array(vector), coverage_dict


def coverage_path(coverage_obj, fail_cover_info):
    vector = []
    # covert the coverage to a vector
    files = coverage_obj["files"]

    file_names = list(fail_cover_info.keys())
    file_names.sort()
    for file_name in file_names:

        if file_name in files.keys():
            # get the all lines
            fail_cover_lines = fail_cover_info[file_name]
            fail_cover_lines.sort(key=lambda x: int(x))
            # get the executed lines
            executed_lines_frequency = files[file_name]["executed_lines_frequency"]
            executed_lines = list(executed_lines_frequency.keys())

            for lineno in fail_cover_lines:
                if lineno in executed_lines:
                    vector.append('1')
                else:
                    vector.append('0')
        else:
            vector.extend(['0'] * len(fail_cover_info[file_name]))

    # return np.array(vector)
    return "".join(vector)


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = norm(A)
    norm_B = norm(B)

    similarity = dot_product / (norm_A * norm_B)
    return similarity


# def jaccard_similarity(binary_vector_a, binary_vector_b):
#     intersection = sum(a * b for a, b in zip(binary_vector_a, binary_vector_b))
#     union = sum(a or b for a, b in zip(binary_vector_a, binary_vector_b))
#     similarity = intersection / union if union != 0 else 0.0

#     return similarity
def coverage_jaccard_similarity(binary_vector_a: str, binary_vector_b: str):
    assert len(binary_vector_a) == len(binary_vector_b), "The length of two vectors should be the same"
    # intersection: a covers the line and b covers the line. if a == 0 and b == 0, we do not count it.
    intersection = sum(a == "1" and b == "1" for a, b in zip(binary_vector_a, binary_vector_b))
    # the union set of covered lines
    union = sum(a == '1' or b == '1' for a, b in zip(binary_vector_a, binary_vector_b))
    similarity = intersection / union if union != 0 else 0.0

    return similarity


# def checkout():
#     tf_python_crash = ['37599', '37798', '37919', '39123', '39131', '39134', '40636', '41426', '41502', '41603',
#      '39159', '39825', '46375', '47012', '48207', '44780', '45015', '45298', '45613', '46063',
#      '46349', '47128', '48434', '48962', '49609', '29987', '30258', '31145', '31409', '34420',
#      '35821', '36037']
#     tf_python_assert = ['30018', '31812', '32220', '33921', '36316', '37018', '37916', '38142', '38647', '38717',
#      '38808', '38899', '39481', '40807', '48315', '48707', '48887', '48900']
#     results = []
#     results += tf_python_crash
#     results += tf_python_assert
#     quoted_characters = [f"cd ../../tf-{word}/tensorflow/ && git branch" for word in results]
#     for result in quoted_characters:
#         print(result)


if __name__ == "__main__":

    # get all bug id under the path
    c_list = ["tf-37915", "tf-39678", "tf-41514", "tf-42067", "tf-45841", "tf-47747", "tf-51050", "tf-52720",
              "tf-41790", "tf-39137"]
    test_case_path = "../data/tensorflow/testCase"
    target_bugs = [b for b in os.listdir(test_case_path) if b.startswith("tf-")]
    target_bugs.sort()
    # target_bugs = ["torch-46042"]

    total_api_calss = []

    with_assertion = set()
    for target_bug in target_bugs:
        raw_source_code = read_text(f"../data/tensorflow/testCase/{target_bug}/{target_bug}.py")
        code_lines = raw_source_code.split('\n')
        for code_line in code_lines:
            if CodeUtils.has_assertion(code_line):
                with_assertion.add(target_bug)

    # get the target bug without assertion
    ids = [i.split("-")[1] for i in target_bugs if i in with_assertion and i not in c_list]
    print(f"total {len(ids)} bugs without assertion")
    print(ids)
