import ast
import copy
import astor
from src.mutation.base import mutation_utils
from src.mutation.base.device_mutator import DeviceMutator
from src.mutation.pytorch import pytorch_utils


class TorchDeviceMutator(DeviceMutator):
    def __init__(self,**kwargs):
        super().__init__()
    

    @staticmethod
    def mutation_point_scan(source_code, mode=None):
        mutation_points = []
        tree = ast.parse(source=source_code, mode='exec')
        if mode == "flip":
            # get the mutation points of flip mutation
            values = ["cpu", "cuda"]
            mutation_points1 = mutation_utils.target_value_scan(tree, targets=values)
            for point in mutation_points1:
                mutation_points.append({'mode': 'flip', 'point': point})
        if mode == "remove_cuda_to_device":
            # get the mutation points of remove cuda to device
            visitor = pytorch_utils.RemoveCudaToDevice()
            visitor.scan(tree)
            mutation_points2 = visitor.mutation_points
            for point in mutation_points2:
                mutation_points.append({'mode': 'remove_cuda_to_device', 'point': point})
        if mode == "remove_device_call":
            # get the mutation points of device call
            visitor = pytorch_utils.RemoveDeviceCall()
            visitor.scan(tree)
            mutation_points3 = visitor.mutation_points
            for point in mutation_points3:
                mutation_points.append({'mode': 'remove_device_call', 'point': point})
        return mutation_points

    @staticmethod
    def mutate(source_code, mutation_point_item):
        """
        mutate_mode: 
        - flip: change parameter of 'cuda' to 'cpu'
        - remove_device_call: remove the .cuda() call
        - remove_cuda_to_device: change cuda.to_device(x) to x

        """
        mutate_mode, mutation_point = mutation_point_item['mode'], mutation_point_item['point']
        tree = ast.parse(source=source_code, mode='exec')
        if mutate_mode == "flip":
            values = ["cpu", "cuda"]
            copy_tree = copy.deepcopy(tree)
            copy_tree = mutation_utils.flip_mutation(copy_tree, mutation_point, values)
            mutated_code = astor.to_source(copy_tree)
        elif mutate_mode == "remove_cuda_to_device":
            visitor = pytorch_utils.RemoveCudaToDevice()
            visitor.mutate(tree, mutation_point)
            mutated_code = astor.to_source(tree)
        elif mutate_mode == "remove_device_call":
            visitor = pytorch_utils.RemoveDeviceCall()
            visitor.mutate(tree, mutation_point)
            mutated_code = astor.to_source(tree)
        else:
            raise NotImplementedError
        return mutated_code

