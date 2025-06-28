import sys
import os
# control the random seed
import random
import numpy as np
# random.seed(0)
# np.random.seed(0)

sys.path.append(os.getcwd())
from src.tools import utils
from src import root_path,DOCTER_PATH,FUNC_SIG_PATH
from src.mutation.operators import construct_mutator
from tqdm import tqdm
from src.tools.logger_utils import LoggerUtils
logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


# pytorch 40
cuda10_torch_bug_list = ['36111', '39344', '39791', '40572', '41013', '46100', '47735', '48215', '50504', '51222', '51396', '51938', '52021', '52646', '53690', '54033', '56277', '56857', '58141', '58651', '59272', '59316', '59979', '60554']
cuda11_torch_bug_list = ['62257', '63607', '64230', '64289', '65001', '65926', '65940', '72128', '72697', '81952', '89928', '95661', '96452', '99092', '99740','111694']
# tf 40
cuda11_tf_bug_list = ['29987', '30781', '32220', '32511', '33921', '34420', '35270', '35973', '36037','35821', '37115', '37599', '37798', '37916', '37919', '38808', '39123', '39131', '39134', '39159',  '40626', '40636', '40807', '41426', '41502', '41603', '44780', '45015', '45613', '46063', '46321', '46349', '46375', '47128', '47135', '48207', '48434', '48707', '48962', '49609',] # remove [39825,55350,53676]

if __name__ == "__main__":
    # test the mutators
    # root_path = "/data/yanming/jiangtianjie/ym_projects/gdefects4dll"
    framework = "pytorch"
    # framework = "tensorflow"
    lib_abbr = "torch" if framework == "pytorch" else "tf"
    bug_list = []
    if framework == "pytorch":
        bug_list = list(set(cuda11_torch_bug_list)|set(cuda10_torch_bug_list))
    elif framework == "tensorflow":
        bug_list = list(set(cuda11_tf_bug_list))
    else:
        raise NotImplemented()


    bug_list.sort(key=lambda x: int(x))

    docter_cons_names = [f[:-5] for f in os.listdir(os.path.join(root_path, DOCTER_PATH[framework]))]
    supported_func_sig = [f[:-5].lower() for f in os.listdir(os.path.join(root_path, FUNC_SIG_PATH[framework]))]
    supported_apis = list(set(docter_cons_names) & set(supported_func_sig))

    bug_info_dict = dict()
    for bug_id in bug_list:
    # for bug_id in ["96452"]:
        # load the text from the file
        target_bug = f"{lib_abbr}-{bug_id}"

        orig_code = utils.read_text(
        os.path.join(root_path, f"./data/{framework}/Result/{target_bug}/{target_bug}-original.py"))
        stack_trace = utils.read_text(
        os.path.join(root_path, f"./data/{framework}/Result/{target_bug}/stack_trace.txt"))

        bug_info_dict[bug_id] = {
            "bug_id": bug_id,
            "orig_code": orig_code,
            "stack_trace": stack_trace,
        }
    #
    # for mutator_level in ["api"]:
    #     mutator = Mutator(framework, mutator_level)
    mutator_list = construct_mutator(framework=framework,mutator_levels=["apis"])
    for mutator in mutator_list:
        for bug_id, bug_info in tqdm(bug_info_dict.items()):
            logger.debug(f"Mutating {bug_id} with {mutator.mutator_level} level mutator")
            # logger.debug(bug_info["orig_code"])
            mutation_points = mutator.mutation_point_scan(source_code=bug_info["orig_code"],
                                                          traceback=bug_info["stack_trace"],
                                                          supported_apis=supported_apis,
                                                          order=0)
            logger.debug(f"{len(mutation_points)} Mutation points.")
            for mutation_point in mutation_points:
                mutated_code = mutator.mutation(source_code=bug_info["orig_code"], traceback=bug_info["stack_trace"],
                                                supported_apis=supported_apis,
                                                order=0,mutation_point=mutation_point)
                # logger.debug(bug_info["orig_code"])
                # logger.debug("===========")
                # logger.debug(mutated_code)




