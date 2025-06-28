
import os
import json

# 24 + 2
cuda10_torch_crash = ['36111', '40572', '41013', '47391', '50504', '51222', '51396', '51938', '52021', '52646',
                      '53053', '54033', '55259', '56277', '56857', '58141', '58651', '58854', '59316', '54312',
                      '60554', '39791', '53690', '59979', ]   # '39108', '39153' can only run on GPUs before 30XX series

# 15
cuda11_torch_crash = ['63607', '64230', '64289', '65001', '65940', '72697', '81952', '89928', '95661', '96452',
                      '99092', '99740', '111694', '62257', '72128']

# 5
cuda10_torch_assert = ['39344', '40919', '47735', '48215', '59272']


# 14
cuda11_torch_assert = ['64190', '66671', '67080', '68202', '78762', '79490', '79611', '89931', '90044', '96705',
                       '97885', '98017', '99666', '98523']

# 40
cuda11_tf_crash = ['29987', '30258', '30781', '31409', '32511', '34420', '35270', '35821', '35973', '36037',
                   '37115', '37599', '37798', '37919', '39123', '39131', '39134', '39159', '39825', '40626',
                   '40636', '41426', '41502', '41603', '44780', '45015', '45298', '45613', '46321', '46349',
                   '46375', '47012', '47128', '47135', '48207', '48434', '48962', '49609', '52172', '57506']



# 20
cuda11_tf_assert = ['30018', '31495', '32220', '33921', '36316', '37018', '37813', '37916', '38142', '38549',
                    '38647', '38717', '38808', '38899', '39481', '40807', '48315', '48887', '48900', '48707']


cuda10_paddlepaddle_crash = ['29540', '32501']

cuda11_paddlepaddle_crash = ['38850', '49986', '53800']

cuda11_paddlepaddle_assert = ['53152', '53624', '53713', '53779', '55568']


bug_dict = {
    "pytorch": {
        "cuda10": {
            "crash": cuda10_torch_crash,
            "assert": cuda10_torch_assert
        },
        "cuda11": {
            "crash": cuda11_torch_crash,
            "assert": cuda11_torch_assert
        }
    },
    "tensorflow": {
        "cuda11": {
            "crash": cuda11_tf_crash,
            "assert": cuda11_tf_assert
        }
    },
    "paddlepaddle": {
        "cuda10": {
            "crash": cuda10_paddlepaddle_crash,
            "assert": []
        },
        "cuda11": {
            "crash": cuda11_paddlepaddle_crash,
            "assert": cuda11_paddlepaddle_assert
        }
    },
}


def parse_bug_info():
    # result = {"torch": {}, "tf": {}}
    # for bug_type, bug_list in bug_lists.items():
    #     framework_nick_name, lang, symptom = bug_type.split("_")
    #     for bug_id in bug_list:
    #         # print(bug_id)
    #         result[framework_nick_name][bug_id] = (lang, symptom)
    # return result
    result = {"torch": {}, "tf": {}, "paddle": {}}
    lang = "python"
    for framework, framework_bugs in bug_dict.items():
        framework_nick_name = framework_nicknames[framework]
        for cuda_version, bug_types in framework_bugs.items():
            for bug_type, bug_list in bug_types.items():
                for bug_id in bug_list:
                    result[framework_nick_name][bug_id] = (lang, bug_type)
    return result


framework_nicknames = {"tensorflow": "tf", "pytorch": "torch", "paddlepaddle": "paddle",}
DOCTER_PATH = {"pytorch": "constraints/pytorch/d2c_result_v2", "tensorflow": "constraints/tensorflow/d2c_result_v2",
               "paddlepaddle": "constraints/paddlepaddle/d2c_result_v2",}
FUNC_SIG_PATH = {"pytorch": "constraints/pytorch/func_sigs", "tensorflow": "constraints/tensorflow/func_sigs",
                 "paddlepaddle": "constraints/paddlepaddle/func_sigs",}

src_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(src_path)

# define the SPFL TOP Level
TOP_K = [1, 5, 10, 20]

regression_version = {"pytorch": {}, "tensorflow": {}}

if __name__ == "__main__":
    # print the total bug numbers and the distribution
    for framework in ["pytorch", "tensorflow"]:
        bugs = []
        for bug_type in ["crash", "assert"]:
            cuda11_bugs = bug_dict[framework]["cuda11"][bug_type]

            if 'cuda10' in bug_dict[framework]:
                cuda10_bugs = bug_dict[framework]["cuda10"][bug_type]
            else:
                cuda10_bugs = []
            bugs.extend(cuda10_bugs)
            bugs.extend(cuda11_bugs)

            print(f"{framework} {bug_type}: {len(cuda10_bugs) + len(cuda11_bugs)}")
        print(f"{framework} total bugs: {len(bugs)}")
