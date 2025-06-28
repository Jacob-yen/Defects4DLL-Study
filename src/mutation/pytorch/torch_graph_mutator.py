import astor
import ast
from src.tools.logger_utils import LoggerUtils
from src.mutation.base.graph_mutator import GraphMutator
from src.mutation.pytorch import pytorch_utils
from src.mutation.base import ModifyGraphForward
from src.tools.enum_class import Framework

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class TorchGraphMutator(GraphMutator):

    def __init__(self,**kwargs):
        super().__init__()

    @staticmethod
    def mutation_point_scan(source_code,mode=None):
        # we have two kinds of graph mutation
        # get the graph and forward mutation points
        # get the forward mutation points
        mutation_points = []
        tree = ast.parse(source=source_code, mode='exec')
        if mode in ['return_none', 'return_input', 'remove_layer']:
            visitor = ModifyGraphForward(framework=Framework.PYTORCH)
            visitor.scan(tree)
            for point in visitor.mutation_points:
                # we add three kinds of mutation points
                # 1. return_none 2. return_input 2. return_input
                mutation_points.append({'mode': mode, 'point': point})

        if mode == "insert_no_grad":
            # get the insert no grad mutation points
            visitor = pytorch_utils.InsertNoGrad()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': 'insert_no_grad', 'point': point})

        if mode == "remove_jitCalls":
            # get the graph mutation points
            visitor = pytorch_utils.RemoveTorchJitCalls()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': 'remove_jitCalls', 'point': point})

        if mode == "add_jitCalls":
            visitor = pytorch_utils.AddTorchJitCall()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': 'add_jitCalls', 'point': point})

        if mode == "remove_jitDecorators":
            visitor = pytorch_utils.RemoveTorchJitDecorator()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({'mode': 'remove_jitDecorators', 'point': point})
        return mutation_points
    

    @staticmethod
    def mutate(source_code, mutation_point_item):
        tree = ast.parse(source=source_code, mode='exec')
        mode = mutation_point_item['mode']
        point = mutation_point_item['point']
        if mode in ['return_none', 'return_input', 'remove_layer']:
            visitor = ModifyGraphForward(framework=Framework.PYTORCH)
            visitor.mutate(tree, mutation_point_item)
        elif mode == 'insert_no_grad':
            visitor = pytorch_utils.InsertNoGrad()
            visitor.mutate(tree, point)
        elif mode == 'remove_jitCalls':
            visitor = pytorch_utils.RemoveTorchJitCalls()
            visitor.mutate(tree, point)
        elif mode == 'add_jitCalls':
            visitor = pytorch_utils.AddTorchJitCall()
            visitor.mutate(tree, point)
        elif mode == 'remove_jitDecorators':
            visitor = pytorch_utils.RemoveTorchJitDecorator()
            visitor.mutate(tree, point)
        else:
            raise ValueError(f"Unknown mutation mode: {mode}")
        return astor.to_source(tree)

    
        

        
