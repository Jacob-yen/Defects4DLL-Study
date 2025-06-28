import os
import sys
sys.path.append(os.getcwd())
import ast
import astor
from src.mutation.base import mutation_utils
from src.tools.logger_utils import LoggerUtils

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class VariableMutator:

    def __init__(self,**kwargs):
        self.framework = kwargs.get('framework', None)
        self.supported_apis = kwargs.get('supported_apis', None)

    @staticmethod
    def _conduct_boolean_mutation(tree, mutation_point):
        for node in ast.walk(tree):
            if (isinstance(node, (ast.Constant,ast.NameConstant))
                    and mutation_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)):
                if isinstance(node.value, bool):
                    node.value = False if node.value else True
        return astor.to_source(tree)

    @staticmethod
    def mutation_point_scan(source_code,mode=None):
        # we have three kinds of mutation points
        mutation_points = []
        # boolean mutation points, subscript removing mutation points, general variable mutation points
        # boolean mutation points
        tree = ast.parse(source=source_code, mode='exec')
        if mode == "boolean":
            values = ["True", "False"]
            mutation_points1 = mutation_utils.target_value_scan(tree, targets=values)
            for point in mutation_points1:
                mutation_points.append({'mode': 'boolean', 'point': point})
            # subscript removing mutation points
        if mode == "subscript_removing":
            visitor = mutation_utils.RemoveSubscript()
            visitor.scan(tree)
            mutation_points2 = visitor.mutation_points
            for point in mutation_points2:
                mutation_points.append({'mode': 'subscript_removing', 'point': point})

        if mode == "modify_variable":
            # general variable mutation points
            visitor = mutation_utils.ModifyVariable()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': 'modify_variable', 'point': point})

        return mutation_points

    def mutate(self,source_code, mutation_point_item):
        tree = ast.parse(source=source_code, mode='exec')
        mode = mutation_point_item['mode']
        point = mutation_point_item['point']
        if mode == 'boolean':
            return self._conduct_boolean_mutation(tree, point)
        elif mode == 'subscript_removing':
            visitor = mutation_utils.RemoveSubscript()
            visitor.mutate(tree, point)
            return astor.to_source(tree)
        elif mode == 'modify_variable':
            visitor = mutation_utils.ModifyVariable()
            visitor.mutate(tree, mutation_point_item)
            return astor.to_source(tree)
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")
        




        


