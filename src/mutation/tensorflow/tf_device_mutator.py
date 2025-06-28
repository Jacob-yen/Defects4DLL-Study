from src.tools.logger_utils import LoggerUtils
from src.mutation.base.device_mutator import DeviceMutator

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class TFDeviceMutator(DeviceMutator):

    def __init__(self,**kwargs):
        super().__init__()

    @staticmethod
    def mutation_point_scan(**kwargs):
        """
        Since the Device setting in TF is global, we perform the mutation on the whole source code
        """
        source_code = kwargs['source_code']
        mutation_points = [{"mode": "visible_device","point":0}]
        if "self.cached_session(use_gpu=" in source_code:
            mutation_points.append({"mode": "switch_device","point":1})
        return mutation_points

    @staticmethod
    def mutate(source_code, mutation_point_item):
        mutate_mode, _ = mutation_point_item['mode'], mutation_point_item['point']
        if mutate_mode == "visible_device":
            code_list = source_code.split("\n")
            code_list.insert(0, "import tensorflow as tf")
            code_list.insert(1, "tf.config.set_visible_devices([], 'GPU')")
            new_code = ""
            for code_item in code_list:
                new_code += code_item + "\n"
            return new_code
        elif mutate_mode == "switch_device":
            code_list = source_code.split("\n")
            new_code = ""
            for code_item in code_list:
                if "self.cached_session(use_gpu=True)" in code_item:
                    new_code += code_item.replace("self.cached_session(use_gpu=True)",
                                                  "self.cached_session(use_gpu=False)") + "\n"
                elif "self.cached_session(use_gpu=False)" in code_item:
                    new_code += code_item.replace("self.cached_session(use_gpu=False)",
                                                  "self.cached_session(use_gpu=True)") + "\n"
                else:
                    new_code += code_item + "\n"
            return new_code
        else:
            raise ValueError(f"Invalid mutate_mode: {mutate_mode}")
