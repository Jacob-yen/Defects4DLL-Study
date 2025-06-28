import ast
import astor
import astunparse
from src.mutation.base import TreeMutator
from src.mutation.base.mutation_utils import CodeUtils
from src.tools.logger_utils import LoggerUtils

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class InsertNoGrad(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.mutation_points.append(0)
        # for node in ast.walk(tree):
        #     if isinstance(node, ast.Import):
        #         for idx, alias in enumerate(node.names):
        #             if alias.name == 'torch':
        #                 insert_position = tree.body.index(node)
        #                 self.mutation_points.append(insert_position)
        #                 break


    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        # self.target_point = mutation_point
        new_node = ast.parse("torch.set_grad_enabled(False)").body[0]
        tree.body.insert(0, new_node)
        new_node = ast.parse("import torch").body[0]
        tree.body.insert(0, new_node)

class RemoveTorchJitCalls(ast.NodeTransformer,TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Call(self, node):
        # if the call node is a torch.jit.script or torch.jit.trace call, then we record the mutation point
        if isinstance(node.func, ast.Attribute) \
                and (node.func.attr == 'script' or node.func.attr == 'trace') \
                and hasattr(node.func, "value") \
                and isinstance(node.func.value, ast.Attribute) \
                and node.func.value.attr == 'jit' \
                and hasattr(node.func.value, "value") \
                and hasattr(node.func.value.value, "id") \
                and node.func.value.value.id == "torch":
            code_line = astunparse.unparse(node)
            if not CodeUtils.has_assertion(code_line):
                if self.mode == "scan":
                    self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                elif self.mode == "mutate":
                    if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                        # remove the torch.jit.script or torch.jit.trace call
                        logger.debug(f"Removing torch.jit call at line {ast.unparse(node)}")
                        node = node.args[0]
                        return node
                else:
                    raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


class AddTorchJitCall(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()
        self.module_based_classes = set()
        
    def scan(self, tree):
        self.mode = "scan"
        # we first get the module based classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute):
                        # if the base class is a module based class, then we record it
                        if base.attr == "Module" and isinstance(base.value, ast.Attribute) and \
                                base.value.attr == "nn" and hasattr(base.value, "value") and \
                                hasattr(base.value.value, "id") and base.value.value.id == "torch":
                            self.module_based_classes.add(node.name)
        # then we scan the tree and visit Call Node.
        # If the Call Node is a function call of the module based classes, then we record the mutation point
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node, "func") and hasattr(node.func, "id") and node.func.id in self.module_based_classes:
                    code_line = astunparse.unparse(node)
                    if not CodeUtils.has_assertion(code_line):
                        self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Call(self, node):
        # if the call node is at the target point, then we add the torch.jit.script call
        if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
            if self.mode == "mutate":
                # add the torch.jit.script call
                node = ast.Call(func=ast.Attribute(value=ast.Name(id='torch.jit', ctx=ast.Load()), 
                                attr='script',ctx=ast.Load()), args=[node], keywords=[])
                ast.fix_missing_locations(node)
                return node
            else:
                raise ValueError("mode for AddTorchJitCall be mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


class RemoveTorchJitDecorator(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    @staticmethod
    def is_jit_decorator(decorator):
        # if the decorator is a jit decorator, then return True
        if isinstance(decorator, ast.Attribute) and decorator.attr in {"script", "_script_if_tracing"} \
                and isinstance(decorator.value, ast.Attribute) and decorator.value.attr == "jit" \
                and hasattr(decorator.value, "value") and hasattr(decorator.value.value, "id") \
                and decorator.value.value.id == "torch":
            return True
        return False
    
    def visit_FunctionDef(self, node):
        decorator_list = node.decorator_list
        if self.mode == "scan":
            # if any of the decorator is jit decorator, then we record the mutation point
            for decorator in decorator_list:
                if self.is_jit_decorator(decorator):
                    self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                    break
        elif self.mode == "mutate":
            # if the mutation point is the target point, then we remove the jit decorator
            if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                node.decorator_list = [decorator for decorator in decorator_list if not self.is_jit_decorator(decorator)]
                return node
        else:
            raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


class RemoveCudaToDevice(ast.NodeTransformer,TreeMutator):
    def __init__(self) -> None:
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if (isinstance(func, ast.Attribute) and
                    func.attr == 'to_device' and
                    isinstance(func.value, ast.Name) and
                    func.value.id == 'cuda'):
                if self.mode == "scan":
                    self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                elif self.mode == "mutate":
                    if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                        node.value = node.value.args[0]
                        return node
                else:
                    raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)

        
class RemoveDeviceCall(ast.NodeTransformer, TreeMutator):
    def __init__(self) -> None:
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit(self, node):
        # if the node contains the .cuda() or .cpu() call, then we record the mutation point
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'cuda' or node.func.attr == 'cpu':
                    if self.mode == "scan":
                        self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                    elif self.mode == "mutate":
                        if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                            # remove the .cuda() or .cpu() call
                            node = node.func.value
                            return node
                    else:
                        raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
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
                        else:
                            if "torch" in target.keys() and f"Tensor.{node.func.attr}" == target["torch"]:
                                call_code = ast.unparse(node).strip()
                                library_name = call_code.split(f".{node.func.attr}")[0]
                                self.function_calls.append(call_code)
                                self.mutation_points.append(
                                    {"target": {library_name: f"torch.Tensor.{node.func.attr}"}, "point": (
                                        node.lineno, node.end_lineno, node.col_offset, node.end_col_offset), "api_call_dict":target})


