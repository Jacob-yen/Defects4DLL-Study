import os
import numpy as np
import openpyxl
import json
import configparser
import pandas as pd
from copy import deepcopy
from src.tools.enum_class import Granularity
from src.tools import utils
from src import root_path
from functools import partial


def mean_first_rank(data_result, fault_ground_truth_lists):
    is_find = False
    fault_ground_truth_value = 0
    index = 1
    for res in data_result:
        if not is_find:
            for fault_ground_truth_list in fault_ground_truth_lists:
                if fault_ground_truth_list in res[0]:
                    fault_ground_truth_value = res[1]
                    is_find = True
        else:
            if res[1] != fault_ground_truth_value:
                return index - 1, fault_ground_truth_value
        index = index + 1
    return index - 1, fault_ground_truth_value


def mean_average_rank(data_result, fault_ground_truth_lists):
    index_list = []
    fault_ground_truth_value_list = []
    for fault_ground_truth_list in fault_ground_truth_lists:
        num = len(index_list)
        is_find = False
        fault_ground_truth_value = 0
        index = 1
        for res in data_result:
            if not is_find:
                if fault_ground_truth_list in res[0]:
                    fault_ground_truth_value = res[1]
                    is_find = True
            else:
                if res[1] != fault_ground_truth_value:
                    index_list.append(index - 1)
                    fault_ground_truth_value_list.append(fault_ground_truth_value)
                    is_find = False
            index = index + 1
        if len(index_list) == num:
            index_list.append(index - 1)
            fault_ground_truth_value_list.append(fault_ground_truth_value)

    return np.mean(index_list), np.mean(fault_ground_truth_value_list)


rank_metrics = {
    "mean_first_rank": mean_first_rank,
    "mean_average_rank": mean_average_rank
}


def ochiai(num_cf, num_uf, num_cs, num_us):
    if num_cf == 0 and num_cs == 0:
        return 0.0
    return round(float(num_cf) / ((num_cf + num_uf) * (num_cf + num_cs)) ** (.5), 6)


def ochiai2(num_cf, num_uf, num_cs, num_us):
    divisor = np.sqrt((num_cf + num_cs) * (num_uf + num_us) * (num_cf + num_uf) * (num_cs + num_us))
    if divisor == 0:
        return 0.0
    return round((num_cf * num_us)/divisor, 6)


def dstar(num_cf, num_uf, num_cs, num_us):
    # print(f"num_cf={num_cf},num_uf={num_uf},num_cs={num_cs},num_us={num_us}")
    # EPSILON = 1.0
    if num_cs + num_uf == 0:
        # slightly higher than the
        return 1.1
    star = 2
    return round(float(num_cf ** star) / (num_cs + num_uf), 6)


def tarantula(num_cf, num_uf, num_cs, num_us):
    if num_cf + num_uf == 0 or num_cf + num_uf == 0 or num_cs + num_us == 0:
        return 0.0
    return round(float(float(num_cf) / (num_cf + num_uf)) / (
            float(num_cf) / (num_cf + num_uf) + float(num_cs) / (num_cs + num_us)), 6)


def jaccard(num_cf, num_uf, num_cs, num_us):
    return round(num_cf/(num_cf + num_uf + num_cs), 6)


def hamann(num_cf, num_uf, num_cs, num_us):
    return round((num_cf + num_us - num_cs - num_uf)/(num_cf + num_uf + num_cs + num_us),6)


def overlap(num_cf, num_uf, num_cs, num_us):
    return round(num_cf/min(num_cf,num_uf,num_cs), 6)



formula_function = {
    "ochiai": ochiai,
    "ochiai2": ochiai2,
    "dstar": dstar,
    "tarantula": tarantula,
    "jaccard": jaccard,
    "hamann": hamann,
    "overlap": overlap

}


def block_aggregation(pass_data, fail_data, formula, mode):
    block_dict = {}
    for file in fail_data['files'].keys():
        for block in fail_data['files'][file]["executed_blocks_frequency"].keys():
            block_start = int(block.split("-")[0])
            block_end = int(block.split("-")[1])
            if pass_data:
                lines = list(fail_data['files'][file]["executed_lines_frequency"].keys())
                score_list = []
                for line in lines:
                    line_num = int(line)
                    if line_num >= block_start and line_num <= block_end:
                        num_cf = fail_data['files'][file]["executed_lines_frequency"][line]
                        num_uf = fail_data['totals']["all_frequency"] - num_cf
                        if file not in pass_data['files'].keys() or line not in pass_data['files'][file][
                            "executed_lines_frequency"].keys():
                            num_cs = 0
                            num_us = pass_data['totals']["all_frequency"]
                        else:
                            num_cs = pass_data['files'][file]["executed_lines_frequency"][line]
                            num_us = pass_data['totals']["all_frequency"] - num_cs
                        score = formula_function[formula](num_cf, num_uf, num_cs, num_us)
                        score_list.append(score)
                if len(score_list) == 0:
                    # print("empty score list")
                    block_dict[file + "::" + block] = -1
                else:
                    if mode == "average":
                        # get the mean of the scores
                        block_dict[file + "::" + block] = float('{:.10f}'.format(sum(score_list) / len(score_list)))
                        # print(f"score list: len={len(score_list)},sum={sum(score_list)}")
                    else:
                        block_dict[file + "::" + block] = float('{:.10f}'.format(max(score_list)))
                        # print(f"score list: len={len(score_list)},sum={sum(score_list)}")
            else:
                block_dict[file + "::" + block] = -1
                # print("empty data pass")
    return block_dict


def function_aggregation(pass_data, fail_data, formula, mode):
    function_dict = {}
    for file in fail_data['files'].keys():
        lines = list(fail_data['files'][file]["executed_lines_frequency"].keys())
        for function in fail_data['files'][file]["executed_functions_frequency"].keys():
            function_start = int(function.split("-")[0])
            function_end = int(function.split("-")[1])
            if pass_data:
                score_list = []
                if function in pass_data['files'][file]['missing_functions']:
                    function_dict[file + "::" + function.split("-")[0]] = 1.0
                    continue
                for line in lines:
                    line_num = int(line)
                    if line_num >= function_start and line_num <= function_end:
                        num_cf = fail_data['files'][file]["executed_lines_frequency"][line]
                        num_uf = fail_data['totals']["all_frequency"] - num_cf
                        if file not in pass_data['files'].keys() or line not in pass_data['files'][file][
                            "executed_lines_frequency"].keys():
                            num_cs = 0
                            num_us = pass_data['totals']["all_frequency"]
                        else:
                            num_cs = pass_data['files'][file]["executed_lines_frequency"][line]
                            num_us = pass_data['totals']["all_frequency"] - num_cs

                        score = formula_function[formula](num_cf, num_uf, num_cs, num_us)
                        score_list.append(score)
                if len(score_list) == 0 and sum(score_list) == 0:
                    function_dict[file + "::" + function.split("-")[0]] = -1
                else:
                    if mode == "average":
                        # get the mean of the scores
                        function_dict[file + "::" + function.split("-")[0]] = \
                            float('{:.10f}'.format(sum(score_list) / len(score_list)))
                    else:
                        function_dict[file + "::" + function.split("-")[0]] = \
                            float('{:.10f}'.format(max(score_list)))
            else:
                function_dict[file + "::" + function.split("-")[0]] = -1
    return function_dict


def file_aggregation(pass_data, fail_data, formula, mode):
    file_dict = {}
    for file in fail_data['files'].keys():
        if pass_data:
            score_list = []
            for line in fail_data['files'][file]["executed_lines_frequency"].keys():
                num_cf = fail_data['files'][file]["executed_lines_frequency"][line]
                num_uf = fail_data['totals']["all_frequency"] - num_cf
                if file not in pass_data['files'].keys() or \
                        line not in pass_data['files'][file]["executed_lines_frequency"].keys():
                    num_cs = 0
                    num_us = pass_data['totals']["all_frequency"]
                else:
                    num_cs = pass_data['files'][file]["executed_lines_frequency"][line]
                    num_us = pass_data['totals']["all_frequency"] - num_cs

                score = formula_function[formula](num_cf, num_uf, num_cs, num_us)
                score_list.append(score)
            if len(score_list) == 0 and len(score_list) == 0:
                file_dict[file] = -1
            else:
                if mode == "average":
                    # get the mean of the scores
                    file_dict[file] = float('{:.10f}'.format(sum(score_list) / len(score_list)))
                else:
                    file_dict[file] = float('{:.10f}'.format(max(score_list)))

        else:
            file_dict[file] = -1
    return file_dict


average_aggregation = {
    # generate the partial function
    Granularity.BLOCK: partial(block_aggregation, mode="average"),
    Granularity.FUNCTION: partial(function_aggregation, mode="average"),
    Granularity.FILE: partial(file_aggregation, mode="average"),
}

max_aggregation = {
    # generate the partial function
    Granularity.BLOCK: partial(block_aggregation, mode="max"),
    Granularity.FUNCTION: partial(function_aggregation, mode="max"),
    Granularity.FILE: partial(file_aggregation, mode="max"),
}


def combine_data(select_files, deduplicate=False):
    all_data = {}
    unique_files = []
    unique_pool = set()
    for select_file in select_files:
        # print(select_file)
        with open(select_file, 'r', encoding='utf8') as fp:
            text = fp.read()
            if deduplicate:
                text_md5 = utils.gen_md5_id(text)
                if text_md5 not in unique_pool:
                    unique_pool.add(text_md5)
                    unique_files.append(select_file)
                else:
                    continue
            else:
                unique_files.append(select_file)
            select_data = json.loads(text)
        if not select_data:
            continue
        if all_data == {}:
            all_data = select_data
        else:
            for file in select_data["files"].keys():
                if file not in all_data["files"].keys():
                    all_data["files"][file] = select_data["files"][file]
                else:
                    for line in select_data["files"][file]["executed_lines_frequency"].keys():
                        if line in all_data["files"][file]["executed_lines_frequency"].keys():
                            all_data["files"][file]["executed_lines_frequency"][line] += \
                                select_data["files"][file]["executed_lines_frequency"][line]
                        else:
                            all_data["files"][file]["executed_lines_frequency"][line] = \
                                select_data["files"][file]["executed_lines_frequency"][line]
                    for function in select_data["files"][file]["executed_functions_frequency"].keys():
                        if function in all_data["files"][file]["executed_functions_frequency"].keys():
                            all_data["files"][file]["executed_functions_frequency"][function] += \
                                select_data["files"][file]["executed_functions_frequency"][function]
                        else:
                            all_data["files"][file]["executed_functions_frequency"][function] = \
                                select_data["files"][file]["executed_functions_frequency"][function]
                    for block in select_data["files"][file]["executed_blocks_frequency"].keys():
                        if block in all_data["files"][file]["executed_blocks_frequency"].keys():
                            all_data["files"][file]["executed_blocks_frequency"][block] += \
                                select_data["files"][file]["executed_blocks_frequency"][block]
                        else:
                            all_data["files"][file]["executed_blocks_frequency"][block] = \
                                select_data["files"][file]["executed_blocks_frequency"][block]
    if len(all_data) != 0:
        for file in all_data["files"].keys():
            mlines = deepcopy(all_data["files"][file]["missing_lines"])
            elines = all_data["files"][file]["executed_lines_frequency"].keys()
            mblocks = deepcopy(all_data["files"][file]["missing_blocks"])
            eblocks = all_data["files"][file]["executed_blocks_frequency"].keys()
            mfunctions = deepcopy(all_data["files"][file]["missing_functions"])
            efunctions = all_data["files"][file]["executed_functions_frequency"].keys()
            for eline in elines:
                if eline in mlines:
                    mlines.remove(eline)
            for eblock in eblocks:
                if eblock in mblocks:
                    mblocks.remove(eblock)
            for efunction in efunctions:
                if efunction in mfunctions:
                    mfunctions.remove(efunction)
            all_data['files'][file]["missing_lines"] = mlines
            all_data['files'][file]["missing_blocks"] = mblocks
            all_data['files'][file]["missing_functions"] = mfunctions

        all_data['totals']['all_frequency'] = len(select_files)

    if len(all_data) != 0:
        for file in all_data["files"].keys():
            lines_dict = all_data["files"][file]["executed_lines_frequency"]
            all_data["files"][file]["executed_lines_frequency"] = {k: lines_dict[k] for k in
                                                                   sorted(lines_dict, key=lambda x: int(x))}
            functions_dict = all_data["files"][file]["executed_functions_frequency"]
            all_data["files"][file]["executed_functions_frequency"] = {k: functions_dict[k] for k in
                                                                       sorted(functions_dict,
                                                                              key=lambda x: int(x.split("-")[0]))}
            blocks_dict = all_data["files"][file]["executed_blocks_frequency"]
            all_data["files"][file]["executed_blocks_frequency"] = {k: blocks_dict[k] for k in sorted(blocks_dict,
                                                                                                      key=lambda
                                                                                                          x: int(
                                                                                                          x.split(
                                                                                                              "-")[
                                                                                                              0]))}
    return all_data


def combine_intersection_data(select_files):
    all_data = {"files":{},'totals':{}}
    unique_files = []
    unique_pool = set()
    # get the intersection file keys
    fail_data_jsons = []
    for select_file in select_files:
        with open(select_file, 'r', encoding='utf8') as fp:
            text = fp.read()
            select_data = json.loads(text)
            fail_data_jsons.append(select_data)
    # get the intersection keys
    intersection_keys = set(fail_data_jsons[0]["files"].keys())
    for fail_data_json in fail_data_jsons:
        intersection_keys = intersection_keys & set(fail_data_json["files"].keys())

    for select_data in fail_data_jsons:
        for file in intersection_keys:
            if file not in all_data["files"].keys():
                all_data["files"][file] = select_data["files"][file]
            else:
                for line in select_data["files"][file]["executed_lines_frequency"].keys():
                    if line in all_data["files"][file]["executed_lines_frequency"].keys():
                        all_data["files"][file]["executed_lines_frequency"][line] += \
                            select_data["files"][file]["executed_lines_frequency"][line]
                    else:
                        all_data["files"][file]["executed_lines_frequency"][line] = \
                            select_data["files"][file]["executed_lines_frequency"][line]
                for function in select_data["files"][file]["executed_functions_frequency"].keys():
                    if function in all_data["files"][file]["executed_functions_frequency"].keys():
                        all_data["files"][file]["executed_functions_frequency"][function] += \
                            select_data["files"][file]["executed_functions_frequency"][function]
                    else:
                        all_data["files"][file]["executed_functions_frequency"][function] = \
                            select_data["files"][file]["executed_functions_frequency"][function]
                for block in select_data["files"][file]["executed_blocks_frequency"].keys():
                    if block in all_data["files"][file]["executed_blocks_frequency"].keys():
                        all_data["files"][file]["executed_blocks_frequency"][block] += \
                            select_data["files"][file]["executed_blocks_frequency"][block]
                    else:
                        all_data["files"][file]["executed_blocks_frequency"][block] = \
                            select_data["files"][file]["executed_blocks_frequency"][block]
    if len(all_data) != 0:
        for file in all_data["files"].keys():
            mlines = deepcopy(all_data["files"][file]["missing_lines"])
            elines = all_data["files"][file]["executed_lines_frequency"].keys()
            mblocks = deepcopy(all_data["files"][file]["missing_blocks"])
            eblocks = all_data["files"][file]["executed_blocks_frequency"].keys()
            mfunctions = deepcopy(all_data["files"][file]["missing_functions"])
            efunctions = all_data["files"][file]["executed_functions_frequency"].keys()
            for eline in elines:
                if eline in mlines:
                    mlines.remove(eline)
            for eblock in eblocks:
                if eblock in mblocks:
                    mblocks.remove(eblock)
            for efunction in efunctions:
                if efunction in mfunctions:
                    mfunctions.remove(efunction)
            all_data['files'][file]["missing_lines"] = mlines
            all_data['files'][file]["missing_blocks"] = mblocks
            all_data['files'][file]["missing_functions"] = mfunctions

        all_data['totals']['all_frequency'] = len(select_files)

    if len(all_data) != 0:
        for file in all_data["files"].keys():
            lines_dict = all_data["files"][file]["executed_lines_frequency"]
            all_data["files"][file]["executed_lines_frequency"] = {k: lines_dict[k] for k in
                                                                   sorted(lines_dict, key=lambda x: int(x))}
            functions_dict = all_data["files"][file]["executed_functions_frequency"]
            all_data["files"][file]["executed_functions_frequency"] = {k: functions_dict[k] for k in
                                                                       sorted(functions_dict,
                                                                              key=lambda x: int(x.split("-")[0]))}
            blocks_dict = all_data["files"][file]["executed_blocks_frequency"]
            all_data["files"][file]["executed_blocks_frequency"] = {k: blocks_dict[k] for k in sorted(blocks_dict,
                                                                                                      key=lambda
                                                                                                          x: int(
                                                                                                          x.split(
                                                                                                              "-")[
                                                                                                              0]))}
    return all_data


def isolation(pass_json, fail_json, granularity, formula, fault_ground_truths, metric="mean_first_rank"):
    with open(pass_json, 'r', encoding='utf8') as fp:
        pass_data = json.load(fp)
    with open(fail_json, 'r', encoding='utf8') as fp:
        fail_data = json.load(fp)
    data_dict = average_aggregation[granularity](pass_data, fail_data, formula)
    # import pdb
    # pdb.set_trace()
    data_result = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)

    data_result_name = f"{granularity}_data_result.json"
    with open(os.path.join(os.path.dirname(pass_json), data_result_name), 'w', encoding='utf8') as f2:
        json.dump(data_result, f2, ensure_ascii=False, indent=2)

    fault_ground_truth = fault_ground_truths[granularity - 1]
    fault_ground_truth_arr = fault_ground_truth.split("+")
    fault_ground_truth_lists = []
    for fault_ground_truth_simple in fault_ground_truth_arr:
        fault_ground_truth_file = fault_ground_truth_simple.split("::")[0]
        if granularity == 3:
            fault_ground_truth_lists.append(fault_ground_truth_file)
        else:
            fault_ground_truth_others = fault_ground_truth_simple.split("::")[1]
            for fault_ground_truth_other in fault_ground_truth_others.split(","):
                fault_ground_truth_lists.append(fault_ground_truth_file + "::" + fault_ground_truth_other)

    return rank_metrics[metric](data_result, fault_ground_truth_lists)


def get_all_rank(fail_data, granularity):
    all_rank = 0
    # with open(file_json_path, 'r', encoding='utf8') as fp:
    #     fail_data = json.load(fp)
    if granularity == 3:
        all_rank = len(fail_data['files'].keys())
    elif granularity == 2:
        for file in fail_data['files'].keys():
            functions_num = len(fail_data['files'][file]["executed_functions_frequency"].keys()) + len(
                fail_data['files'][file]["missing_functions"])
            all_rank += functions_num
    elif granularity == 1:
        for file in fail_data['files'].keys():
            blocks_num = len(fail_data['files'][file]["executed_blocks_frequency"].keys()) + len(
                fail_data['files'][file]["missing_blocks"])
            all_rank += blocks_num
    return all_rank


class LocalizationAnalyzer():
    def __init__(self, framework, method_name, coverage_json_path) -> None:
        self.framework = framework
        self.method_name = method_name
        self.coverage_json_path = coverage_json_path
        self.coverage_json = os.path.join(self.coverage_json_path, f"{self.framework}-{self.method_name}")
        if self.framework == "tensorflow":
            self.framework_prefix = "tf"
        elif self.framework == "pytorch":
            self.framework_prefix = "torch"
        elif self.framework == "jittor":
            self.framework_prefix = "jittor"
        elif self.framework == "mxnet":
            self.framework_prefix = "mxnet"
        elif self.framework == "mindspore":
            self.framework_prefix = "ms"
        elif self.framework == "paddlepaddle":
            self.framework_prefix = "paddle"
        self.bug_info = pd.read_excel(os.path.join(root_path, "data", f'{framework}-release.xlsx'), sheet_name='Sheet1')


    def bug_localization(self, fail_bug, pass_file_paths, fail_json_paths, formula, granularity_levels,aggregate_mode="average",
                         metric="mean_first_rank"):

        localization = dict()
        selected_row = self.bug_info[self.bug_info['pr_id'] == int(fail_bug)]
        if not selected_row.empty:
            fault_ground_truth_file = selected_row['fault_ground_truth_file'].values[0]
            fault_ground_truth_function = selected_row['fault_ground_truth_function'].values[0]
            fault_ground_truth_block = selected_row['fault_ground_truth_block'].values[0]
            fault_ground_truths = [fault_ground_truth_block, fault_ground_truth_function, fault_ground_truth_file]
        else:
            raise ValueError(f"Bug {fail_bug} not found in {self.framework}-release.xlsx")

        fail_data = combine_data(fail_json_paths)
        pass_data = combine_data(pass_file_paths)

        data_result_total = {}
        for granularity in granularity_levels:
            (rank, score),data_result = self.isolation(pass_data, fail_data, fail_bug, granularity,
                                         formula, fault_ground_truths, metric, aggregate_mode)
            data_result_total[granularity] = data_result
            all_rank = get_all_rank(fail_data, granularity)
            rank_ratio = 0
            if all_rank != 0:
                rank_ratio = rank / all_rank
            localization[granularity] = (rank, score, rank_ratio)
        return localization, pass_data,data_result_total

    def isolation(self, pass_data, fail_data, fail_bug, granularity, formula, fault_ground_truths,
                  metric="mean_first_rank", mode="average"):

        if mode == "average":
            data_dict = average_aggregation[granularity](pass_data, fail_data, formula)
        else:
            data_dict = max_aggregation[granularity](pass_data, fail_data, formula)
        data_result = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)

        data_result_name = f"{formula}_{mode}_{granularity}_data_result.json"
        with open(os.path.join(self.coverage_json, fail_bug, data_result_name), 'w', encoding='utf8') as f2:
            json.dump(data_result, f2, ensure_ascii=False, indent=2)

        fault_ground_truth = fault_ground_truths[granularity - 1]
        fault_ground_truth_arr = fault_ground_truth.split("+")
        fault_ground_truth_lists = []
        try:
            for fault_ground_truth_simple in fault_ground_truth_arr:
                fault_ground_truth_file = fault_ground_truth_simple.split("::")[0]
                if granularity == 3:
                    fault_ground_truth_lists.append(fault_ground_truth_file)
                else:
                    fault_ground_truth_others = fault_ground_truth_simple.split("::")[1]
                    for fault_ground_truth_other in fault_ground_truth_others.split(","):
                        fault_ground_truth_lists.append(fault_ground_truth_file + "::" + fault_ground_truth_other)
        except:
            print(f"fault_ground_truth_simple: {fault_ground_truth_simple}")
            print(f"fault_ground_truth: {fault_ground_truth}")
            print(f"fault_ground_truth_arr: {fault_ground_truth_arr}")
            print(f"fault_ground_truth_lists: {fault_ground_truth_lists}")
            raise ValueError("fault_ground_truth_lists error")

        return rank_metrics[metric](data_result, fault_ground_truth_lists), data_result



    def filtered_bug_localization(self, fail_bug, pass_file_paths, formula, deduplicate_choice, granularity_levels, filtered_files,
                            aggregate_mode="average", metric="mean_first_rank"):

            localization = dict()
            selected_row = self.bug_info[self.bug_info['pr_id'] == int(fail_bug)]
            if not selected_row.empty:
                fault_ground_truth_file = selected_row['fault_ground_truth_file'].values[0]
                fault_ground_truth_function = selected_row['fault_ground_truth_function'].values[0]
                fault_ground_truth_block = selected_row['fault_ground_truth_block'].values[0]
                fault_ground_truths = [fault_ground_truth_block, fault_ground_truth_function, fault_ground_truth_file]
            else:
                raise ValueError(f"Bug {fail_bug} not found in {self.framework}-release.xlsx")

            # first we get the data_fail
            fail_json_path = os.path.join(self.coverage_json, fail_bug, f"data_fail.json")
            # get the parent path

            # then we combine and get the data_pass
            with open(fail_json_path, 'r', encoding='utf8') as fp:
                fail_data = json.load(fp)
            pass_data = combine_data(pass_file_paths, deduplicate_choice)

            #
            for granularity in granularity_levels:
                rank, score = self.filtered_isolation(pass_data, fail_data, fail_bug, granularity,
                                            formula, fault_ground_truths, filtered_files, metric, aggregate_mode)
                all_rank = get_all_rank(fail_json_path, granularity)
                rank_ratio = 0
                if all_rank != 0:
                    rank_ratio = rank / all_rank
                localization[granularity] = (rank, score, rank_ratio)
            return localization, pass_data


    def filtered_isolation(self, pass_data, fail_data, fail_bug, granularity, formula, fault_ground_truths,filtered_files,
                           metric="mean_first_rank", mode="average"):

        if mode == "average":
            data_dict = average_aggregation[granularity](pass_data, fail_data, formula)
        else:
            data_dict = max_aggregation[granularity](pass_data, fail_data, formula)
        # print(f"Before filter ({granularity}). I have {len(data_dict)} elements to rank")
        # filtered_data_dict
        filtered_data_dict = {}
        for element_key in data_dict.keys():
            if element_key.split("::")[0] in filtered_files:
                filtered_data_dict[element_key] = data_dict[element_key]
        # print(f"After filter ({granularity}). I have {len(filtered_data_dict)} elements to rank")
        data_result = sorted(filtered_data_dict.items(), key=lambda x: x[1], reverse=True)

        data_result_name = f"{mode}_{granularity}_data_result.json"
        with open(os.path.join(self.coverage_json, fail_bug, data_result_name), 'w', encoding='utf8') as f2:
            json.dump(data_result, f2, ensure_ascii=False, indent=2)

        fault_ground_truth = fault_ground_truths[granularity - 1]
        fault_ground_truth_arr = fault_ground_truth.split("+")
        fault_ground_truth_lists = []
        try:
            for fault_ground_truth_simple in fault_ground_truth_arr:
                fault_ground_truth_file = fault_ground_truth_simple.split("::")[0]
                if granularity == 3:
                    fault_ground_truth_lists.append(fault_ground_truth_file)
                else:
                    fault_ground_truth_others = fault_ground_truth_simple.split("::")[1]
                    for fault_ground_truth_other in fault_ground_truth_others.split(","):
                        fault_ground_truth_lists.append(fault_ground_truth_file + "::" + fault_ground_truth_other)
        except:
            print(f"fault_ground_truth_simple: {fault_ground_truth_simple}")
            print(f"fault_ground_truth: {fault_ground_truth}")
            print(f"fault_ground_truth_arr: {fault_ground_truth_arr}")
            print(f"fault_ground_truth_lists: {fault_ground_truth_lists}")
            raise ValueError("fault_ground_truth_lists error")

        return rank_metrics[metric](data_result, fault_ground_truth_lists)


