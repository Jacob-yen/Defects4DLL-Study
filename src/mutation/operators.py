import random
import ast
import astor
from functools import partial
from collections import defaultdict
from src.tools.enum_class import Framework

class Mutator:
    def __init__(self, framework, mutator_level, sub_mutator=None):
        assert framework in [Framework.PYTORCH, Framework.TENSORFLOW, Framework.JITTOR, Framework.PADDLE, Framework.MXNET, Framework.MINDSPORE], f"Unsupported framework: {framework}"
        self.framework = framework
        from src.mutation.base.variable_mutator import VariableMutator as variable_mutation
        from src.mutation.base.control_flow_mutator import ControlFlowMutator as control_flow_mutation
        from src.mutation.base.llm_mutator import LLMMutator as llm_mutation
        if framework == Framework.PYTORCH:
            from src.mutation.pytorch.torch_api_mutator import TorchAPIMutator as api_mutation
            from src.mutation.pytorch.torch_device_mutator import TorchDeviceMutator as device_mutation
            from src.mutation.pytorch.torch_graph_mutator import TorchGraphMutator as graph_mutation
        elif framework == Framework.TENSORFLOW:
            from src.mutation.tensorflow.tf_api_mutator import TFAPIMutator as api_mutation
            from src.mutation.tensorflow.tf_device_mutator import TFDeviceMutator as device_mutation
            from src.mutation.tensorflow.tf_graph_mutator import TFGraphMutator as graph_mutation
        elif framework == Framework.PADDLE:
            from src.mutation.paddle.paddle_api_mutator import PaddleAPIMutator as api_mutation
            from src.mutation.paddle.paddle_device_mutator import PaddleDeviceMutator as device_mutation
            from src.mutation.paddle.paddle_graph_mutator import PaddleGraphMutator as graph_mutation
        elif framework == Framework.JITTOR:
            from src.mutation.jittor.jittor_api_mutator import JittorAPIMutator as api_mutation
            from src.mutation.jittor.jittor_device_mutator import JittorDeviceMutator as device_mutation
            from src.mutation.jittor.jittor_graph_mutator import JittorGraphMutator as graph_mutation
        elif framework == Framework.MINDSPORE:
            from src.mutation.mindspore.mindspore_api_mutator import MindSporeAPIMutator as api_mutation
            from src.mutation.mindspore.mindspore_device_mutator import MindSporeDeviceMutator as device_mutation
            from src.mutation.mindspore.mindspore_graph_mutator import MindSporeGraphMutator as graph_mutation
        elif framework == Framework.MXNET:
            from src.mutation.mxnet.mxnet_api_mutator import MxnetAPIMutator as api_mutation
            from src.mutation.mxnet.mxnet_device_mutator import MxnetDeviceMutator as device_mutation
            from src.mutation.mxnet.mxnet_graph_mutator import MxnetGraphMutator as graph_mutation
        else:
            raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")
        self.mutator_level = mutator_level
        self.sub_mutator = sub_mutator
        self.selected = 0
        self.success_pass = 0
        self.success_mutation = 0
        self.score = 0
        self.mutation_mapping = {
            "api": api_mutation,
            "device": device_mutation,
            "graph": graph_mutation,
            "control_flow": control_flow_mutation,
            "variable": variable_mutation,
        }

    @property
    def mutator_name(self):
        if self.sub_mutator is None:
            return self.mutator_level
        else:
            return self.sub_mutator

    @property
    def rate(self):
        if self.selected == 0:
            raise ValueError("No mutation has been conducted yet.")
        else:
            return (self.success_pass + self.success_mutation + 1) / (self.selected + 1e-6)

    def update_score(self, score):
        self.score = score

    def update_selected(self):
        self.selected += 2

    def update_success_pass(self):
        self.success_pass += 1

    def update_success_mutation(self):
        self.success_mutation += 1

    def print_score(self):
        return f"[Success_Mutation({self.success_mutation}) + Success_Pass({self.success_pass})] / Selected({self.selected} + 1e-6) Score: {round(self.score, 4)}"

    def mutation_point_scan(self, **kwargs):
        source_code = kwargs["source_code"]
        supported_apis = kwargs["supported_apis"]
        traceback = kwargs["traceback"]
        order = kwargs["order"]
        tree = ast.parse(source_code, mode='exec')
        code_wo_comments = astor.to_source(tree)

        mutation_class = self.mutation_mapping[self.mutator_level]
        mutator = mutation_class(framework=self.framework, supported_apis=supported_apis)
        fail_test = source_code if self.mutator_level == "control_flow" else code_wo_comments
        if self.mutator_level == "api":
            mutations = mutator.mutation_point_scan(source_code=fail_test, mode=self.sub_mutator)

        elif self.mutator_level == "control_flow":
            if order == 0:
                mutations = mutator.mutation_point_scan(fail_test, traceback, mode=self.sub_mutator)
            else:
                return []
        else:
            mutations = mutator.mutation_point_scan(source_code=fail_test, mode=self.sub_mutator)
        return mutations

    def mutation(self, **kwargs):
        source_code = kwargs["source_code"]
        supported_apis = kwargs["supported_apis"]
        order = kwargs["order"]
        mutation_item = kwargs["mutation_point"]

        tree = ast.parse(source_code, mode='exec')
        code_wo_comments = astor.to_source(tree)
        mutation_class = self.mutation_mapping[self.mutator_level]
        fail_test = source_code if self.mutator_level == "control_flow" else code_wo_comments

        mutator = mutation_class(framework=self.framework,supported_apis=supported_apis)
        if self.mutator_level == "api":
            return mutator.mutate(source_code=fail_test, mutation_point_item=mutation_item)
        elif self.mutator_level == "control_flow":
            if order > 0:
                return None
            else:
                return mutator.mutate(source_code=fail_test, mutation_point_item=mutation_item)
        else:
            return mutator.mutate(source_code=fail_test, mutation_point_item=mutation_item)


def construct_mutator(framework, mutator_levels):
    mutator_list = []
    if framework == Framework.PYTORCH:
        from src.mutation.pytorch import sub_mutators_mapping
    elif framework == Framework.TENSORFLOW:
        from src.mutation.tensorflow import sub_mutators_mapping
    elif framework == Framework.JITTOR:
        from src.mutation.jittor import sub_mutators_mapping
    elif framework == Framework.MXNET:
        from src.mutation.mxnet import sub_mutators_mapping
    elif framework == Framework.MINDSPORE:
        from src.mutation.mindspore import sub_mutators_mapping
    elif framework == Framework.PADDLE:
        from src.mutation.paddle import sub_mutators_mapping
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    for mutator_level in mutator_levels:
        sub_mutators = sub_mutators_mapping[mutator_level]
        for sub_mutator in sub_mutators:
            # print(1)
            mutator_list.append(Mutator(framework, mutator_level, sub_mutator))
    return mutator_list
