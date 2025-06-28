import numpy as np
import ast
from src.tools.enum_class import Framework

class TreeMutator:
    def __init__(self) -> None:
        self.mutation_points = []
        self.mode = None
        self.target_point = None

class ModifyGraphForward(ast.NodeTransformer, TreeMutator):
    def __init__(self,framework) -> None:
        super().__init__()
        self.framework = framework
    
    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, target_point_item):
        self.mode = "mutate"
        self.target_point = target_point_item
        self.visit(tree)
        return tree

    def is_forward_function_def(self, node):
        if self.framework == Framework.PYTORCH:
            return node.name == "forward"
        elif self.framework == Framework.TENSORFLOW:
            return node.name == "call" or node.name == "__call__"
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def visit_FunctionDef(self, node):        
        if self.is_forward_function_def(node):
            if self.mode == "scan":
                self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                return node
            elif self.mode == "mutate":
                mutate_name, point = self.target_point["mode"], self.target_point["point"]
                if point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                    if mutate_name == "return_none":
                        node = self._return_none(node)
                    elif mutate_name == "return_input":
                        node = self._return_input(node)
                    elif mutate_name == "remove_layer":
                        node = self._remove_layer(node)
                    else:
                        raise ValueError(f"Unknown mutation name: {mutate_name}")
                return node
            else:
                raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)

    @staticmethod
    def _return_none(node):
        # remove all node from node.body and insert one return None node
        node.body.clear()
        new_node = ast.Return(value=ast.Name(id="None", ctx=ast.Load()))
        node.body.insert(0, new_node)
        return node

    @staticmethod
    def _return_input(node):
        input_arg = "None"
        for arg in node.args.args:
            if arg.arg != "self":
                input_arg = arg.arg
                break
        node.body = [ast.Return(value=ast.Name(id=input_arg, ctx=ast.Load()))]
        return node

    @staticmethod
    def _remove_layer(node):
        layer_num = np.random.randint(1, len(node.body) + 1)
        node.body = node.body[:-layer_num]
        input_arg = "None"
        for arg in node.args.args:
            if arg.arg != "self":
                input_arg = arg.arg
                break  
        # we need to return the last defined variable
        body_len = len(node.body)
        if body_len > 0 and hasattr(node.body[body_len - 1], "targets") and hasattr(
                node.body[body_len - 1].targets[0], "id"):
            new_node = ast.Return(value=ast.Name(id=node.body[body_len - 1].targets[0].id, ctx=ast.Load()))
            node.body.insert(len(node.body), new_node)
        else:
            new_node = ast.Return(value=ast.Name(id=input_arg, ctx=ast.Load()))
            node.body.insert(len(node.body), new_node)
        return node
    