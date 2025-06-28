import re
import ast
import astor
import random
import numpy as np
from src.tools.logger_utils import LoggerUtils
from src.mutation.base import TreeMutator


logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


def remove_tf_function(node):
    node.decorator_list = [
        decorator
        for decorator in node.decorator_list
        if not (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "function"
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "tf"
        )
    ]


class RemoveTFFunctionAnnotations(ast.NodeTransformer):
    def __init__(self):
        self.tf_function_node_list = []

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == "function" and isinstance(decorator.value,
                                                                                                    ast.Name) and decorator.value.id == "tf":
                self.tf_function_node_list.append(node)
        return self.generic_visit(node)


def add_tf_function(node):
    decorator = ast.Name(id='tf', ctx=ast.Load())
    decorator_attr = ast.Attribute(value=decorator, attr='function', ctx=ast.Load())
    node.decorator_list.insert(0, decorator_attr)


class AddTFFunctionAnnotations(ast.NodeTransformer):
    def __init__(self):
        self.tf_function_node_list = []

    def visit_FunctionDef(self, node):
        self.tf_function_node_list.append(node)
        return self.generic_visit(node)


class FunctionCallVisitor():
    def __init__(self, targets):
        self.targets = targets
        self.function_calls = []
        self.mutation_points = []

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

    def visit(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for target in self.targets:
                    if isinstance(node.func, ast.Name) and node.func.id in target.keys():
                        call_code = ast.unparse(node).strip()
                        self.function_calls.append(call_code)
                        # append the start and end lineno and the offset
                        self.mutation_points.append({"target": {node.func.id: target[node.func.id]}, "point": (
                            node.lineno, node.end_lineno, node.col_offset, node.end_col_offset), "api_call_dict":target})

                    if isinstance(node.func, ast.Attribute):
                        library_name = self._get_library_name(node.func)
                        if library_name and library_name.startswith(tuple(list(target.keys()))):
                            api_name = self._get_api_name(node.func)
                            if api_name == target[library_name]:
                                call_code = ast.unparse(node).strip()
                                self.function_calls.append(call_code)
                                self.mutation_points.append({"target": {library_name: api_name}, "point": (
                                    node.lineno, node.end_lineno, node.col_offset, node.end_col_offset), "api_call_dict":target})


class InsertNoGrad(ast.NodeTransformer, TreeMutator):
    def scan(self, tree):
        self.mode = "scan"
        self.mutation_points.append(0)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        source_code = astor.to_source(tree)
        code_list = source_code.split("\n")
        code_list = ["    " + code_item for code_item in code_list]
        code_list.insert(0, "import tensorflow as tf")
        code_list.insert(1, "with tf.GradientTape(watch_accessed_variables=False) as tape:")
        new_code = ""
        for code_item in code_list:
            new_code += code_item + "\n"
        return ast.parse(new_code, mode='exec')


class StaticGraph(ast.NodeTransformer, TreeMutator):
    def scan(self, tree):
        self.mode = "scan"
        # 0: "disable_eager_execution", 1: "run_functions_eagerly_False", 2: "add_tf_function"
        self.mutation_points.append(0)
        self.mutation_points.append(1)

        visitor = AddTFFunctionAnnotations()
        visitor.visit(tree)
        if len(visitor.tf_function_node_list) > 0:
            self.mutation_points.append(2)

    def mutate(self, tree,target_point_item):

        self.mode = "mutate"
        self.target_point = target_point_item
        # 0: "disable_eager_execution", 1: "run_functions_eagerly_False", 2: "add_tf_function"
        if self.target_point == 0:
            # 添加tf.compat.v1.disable_eager_execution()
            new_node_1 = ast.parse("import tensorflow as tf").body[0]
            new_node_2 = ast.parse("tf.compat.v1.disable_eager_execution()").body[0]
            tree.body.insert(0, new_node_1)
            tree.body.insert(1, new_node_2)
            return tree
        elif self.target_point == 1:
            # tf.config.run_functions_eagerly(False)
            new_node_1 = ast.parse("import tensorflow as tf").body[0]
            new_node_2 = ast.parse("tf.config.run_functions_eagerly(False)").body[0]
            tree.body.insert(0, new_node_1)
            tree.body.insert(1, new_node_2)
            return tree
        elif self.target_point == 2:
            visitor = AddTFFunctionAnnotations()
            visitor.visit(tree)
            func_index = np.random.randint(0, len(visitor.tf_function_node_list))
            add_tf_function(visitor.tf_function_node_list[func_index])
            # insert the import statement
            new_node_1 = ast.parse("import tensorflow as tf").body[0]
            tree.body.insert(0, new_node_1)
            return tree
        else:
            raise ValueError(f"Invalid target point:{self.target_point}")


class DynamicGraph(ast.NodeTransformer, TreeMutator):
    def scan(self, tree):
        # 0: "remove_disable_eager_execution", 1: "add run_functions_eagerly_True", 2: "remove_tf_function"
        self.mode = "scan"
        # check if tf.compat.v1.disable_eager_execution() exists
        code = astor.to_source(tree)
        pattern = r'tf.compat.v1.disable_eager_execution()'
        if len(re.findall(pattern, code)) > 0:
            self.mutation_points.append(0)

        # # add mutation points for adding tf.config.run_functions_eagerly(True)
        # self.mutation_points.append(1)

        visitor = RemoveTFFunctionAnnotations()
        visitor.visit(tree)
        if len(visitor.tf_function_node_list) > 0:
            self.mutation_points.append(2)

    def mutate(self, tree,target_point_item):
        self.target_point = target_point_item
        self.mode = "mutate"
        # 0: "remove_disable_eager_execution", 1: "run_functions_eagerly_True", 2: "remove_tf_function"

        if self.target_point == 0:
            logger.debug("remove_disable_eager_execution")
            source_code = astor.to_source(tree)
            new_source_code = source_code.replace("tf.compat.v1.disable_eager_execution()", "")
            return ast.parse(new_source_code, mode='exec')
        elif self.target_point == 1:
            logger.debug("add run_functions_eagerly_True")
            # add tf.config.run_functions_eagerly(False)
            new_node_1 = ast.parse("import tensorflow as tf").body[0]
            new_node_2 = ast.parse("tf.config.run_functions_eagerly(True)").body[0]
            tree.body.insert(0, new_node_1)
            tree.body.insert(1, new_node_2)
            return tree
        elif self.target_point == 2:
            logger.debug("remove_tf_function")
            visitor = RemoveTFFunctionAnnotations()
            visitor.visit(tree)
            func_index = np.random.randint(0, len(visitor.tf_function_node_list))
            remove_tf_function(visitor.tf_function_node_list[func_index])
            return tree
        else:
            raise ValueError(f"Invalid target point:{self.target_point}")



