import os
import sys
sys.path.append(os.getcwd())
import ast
import astor
from src.mutation.base.graph_mutator import GraphMutator
from src.mutation.base import ModifyGraphForward
from src.mutation.tensorflow import tensorflow_utils
from src.tools.logger_utils import LoggerUtils
from src.tools.enum_class import Framework

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class TFGraphMutator(GraphMutator):
    
    def __init__(self,**kwargs):
        super().__init__()

    @staticmethod
    def mutation_point_scan(source_code, mode=None):
        mutation_points = []
        tree = ast.parse(source_code, mode='exec')
        if mode == 'insert_no_grad':
            visitor = tensorflow_utils.InsertNoGrad()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': mode, 'point': point})
        elif mode == 'static_graph':
            visitor = tensorflow_utils.StaticGraph()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': mode, 'point': point})
        elif mode == 'dynamic_graph':
            visitor = tensorflow_utils.DynamicGraph()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': mode, 'point': point})

        elif mode in ['return_none','return_input','remove_layer']:
            tree = ast.parse(source_code, mode='exec')
            # add the forward graph mutation points
            visitor = ModifyGraphForward(framework=Framework.TENSORFLOW)
            visitor.scan(tree)
            for point in visitor.mutation_points:
                # we add three kinds of mutation points
                # 1. return_none 2. return_input 2. return_input
                mutation_points.append({'mode': mode, 'point': point})
        else:
            raise ValueError(f"Invalid mode {mode} for TFGraphMutator")
        return mutation_points

    @staticmethod
    def mutate(source_code, mutation_point_item):

        mode = mutation_point_item['mode']
        point = mutation_point_item['point']
        tree = ast.parse(source_code, mode='exec')
        if mode in ['return_none', 'return_input', 'remove_layer']:
            tree = ast.parse(source=source_code, mode='exec')
            visitor = ModifyGraphForward(framework=Framework.TENSORFLOW)
            visitor.mutate(tree, mutation_point_item)
        elif mode == 'insert_no_grad':
            visitor = tensorflow_utils.InsertNoGrad()
            tree = visitor.mutate(tree,point)
        elif mode == 'static_graph':
            visitor = tensorflow_utils.StaticGraph()
            tree = visitor.mutate(tree,point)
        elif mode == 'dynamic_graph':
            visitor = tensorflow_utils.DynamicGraph()
            tree = visitor.mutate(tree,point)
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")
        return astor.to_source(tree)
