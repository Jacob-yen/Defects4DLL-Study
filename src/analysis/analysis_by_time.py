import datetime
import os
import sys

sys.path.append(os.getcwd())
from src import root_path, parse_bug_info, TOP_K
from collections import defaultdict
from src.tools.enum_class import Granularity
import pandas as pd
from src.localization.spectrum_analysis import LocalizationAnalyzer
import argparse



def localization_dataframe(baseline_name, baseline_dict):
    header = ["bug_id", "symptom", f"{baseline_name}_block_rank", f"{baseline_name}_block_score",
              f"{baseline_name}_block_rank_ratio", f"{baseline_name}_function_rank", f"{baseline_name}_function_score",
              f"{baseline_name}_function_rank_ratio", f"{baseline_name}_file_rank", f"{baseline_name}_file_score",
              f"{baseline_name}_file_rank_ratio"]
    granularity_dict = Granularity.DICT
    bug_ids = sorted(list(baseline_dict.keys()))
    rows = []
    for bug_id in bug_ids:
        row = dict()
        row["bug_id"] = f"{args.framework}-{bug_id}"
        row["symptom"] = total_bug_info[bug_id][1]
        for granularity in granularity_dict.keys():
            row[f"{baseline_name}_{granularity}_rank"], row[f"{baseline_name}_{granularity}_score"], \
                row[f"{baseline_name}_{granularity}_rank_ratio"] = baseline_dict[bug_id][granularity_dict[granularity]]
        rows.append(row)
    return pd.DataFrame(rows, columns=header)


def get_bug_list(baseline):
    res = []
    # for lang in ["python","c"]:
    save_path = os.path.join(coverage_json_path, f"{args.framework}-{baseline}")
    res.extend([f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))])
    return res


def statistic(counter_dict, score, metric):
    header = ["granularity", "method"]
    for k in TOP_K:
        header.append(f"Top{k}")
    header.append(f"{metric}")
    rows = []
    granularity_dict = Granularity.DICT
    for granularity in granularity_dict.keys():
        # append baseline2 row
        row2 = dict()
        row2["granularity"] = granularity
        row2["method"] = args.method
        for k in TOP_K:
            row2[f"Top{k}"] = counter_dict[granularity][f"Top{k}"]
        row2[f"{metric}"] = score[granularity]
        rows.append(row2)
    return pd.DataFrame(rows, columns=header)


def top_counter(df, baseline_name):
    results = dict()
    granularity_dict = Granularity.DICT
    for granularity in granularity_dict.keys():
        results[granularity] = {}
        for k in TOP_K:
            count = len(df[df[f"{baseline_name}_{granularity}_rank"] <= k])
            results[granularity][f"Top{k}"] = count
    return results


def metric_calculator(df, baseline_name):
    results = {}
    granularity_dict = Granularity.DICT
    for granularity in granularity_dict.keys():
        average_value = df[f"{baseline_name}_{granularity}_rank"].mean()
        results[granularity] = average_value
    return results



formulas = ["ochiai", "ochiai2", "dstar", "tarantula", "jaccard", "hamann", "overlap"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiments Script For Rebug")
    parser.add_argument("--framework", type=str, default="tensorflow",
                        choices=["pytorch", "tensorflow", "paddlepaddle"])

    parser.add_argument("--root_result_dir", type=str,
                        default='/data/all-rebug-test/result_20250626-demo-run-900s-r1.bk')
    parser.add_argument("--aggregate_mode", type=str, default="average")
    parser.add_argument("--method", type=str, default="rule", )
    parser.add_argument("--specificMethod", type=str, default="our", )
    parser.add_argument("--formulas", type=str, default=["ochiai"])
    parser.add_argument("--bug_types", type=str, nargs="+", default=["crash"])
    parser.add_argument("--time_slots", type=int, nargs="+", default=[900])
    parser.add_argument("--metrics", type=str, nargs="+", default=["mean_first_rank"])
    parser.add_argument("--save_dir", type=str, default="results/pytorch/result_20250626-demo-run-900s-r1")
    args = parser.parse_args()
    print(args)

    s0 = datetime.datetime.now()
    # metrics_mapping = {"mean_first_rank": "MFR", "mean_average_rank": "MAR"}
    # metrics = {"mean_first_rank": "MFR"}
    formulas = args.formulas
    aggregate_mode = args.aggregate_mode
    granularity_levels = Granularity.TOTAL
    framework = args.framework
    target_bug_types = args.bug_types
    # get the result_dir name of root_result_dir
    exp_idntfr = os.path.basename(args.root_result_dir.rstrip("/"))
    result_dir = os.path.join(root_path, args.save_dir)
    os.makedirs(result_dir, exist_ok=True)

    specificMethod = args.specificMethod
    # specificMethod =
    method = args.method
    top_k_choices = [1, 3, 5, 10]
    # time_slots = [900, 1800, 2700, 3600]
    time_slots = args.time_slots

    # load bug info
    if framework == "tensorflow":
        framework_prefix = "tf"
    elif framework == "pytorch":
        framework_prefix = "torch"
    else:
        framework_prefix = "paddle"
    total_bug_info = parse_bug_info()[framework_prefix]

    for time_slot in time_slots:

        for metric_name in args.metrics:
            for formula in formulas:
                print(
                    f"result in {exp_idntfr}-{framework}-{method}-{specificMethod}-{time_slot}s-{formula}-{aggregate_mode}-{'+'.join(target_bug_types)}-{metric_name}.xlsx")
                method_dict = {}
                bug_save_dir = os.path.join(args.root_result_dir, method)
                method_test_path = os.path.join(args.root_result_dir, method)
                coverage_json_path = os.path.join(args.root_result_dir, "coverage_json")
                method_coverage_json_path = os.path.join(args.root_result_dir, "coverage_json", f"{framework}-{method}")
                method_analyzer = LocalizationAnalyzer(framework=framework, method_name=method,
                                                       coverage_json_path=coverage_json_path)

                target_bug_ids = [b for b in os.listdir(method_coverage_json_path) if b in total_bug_info.keys()]
                target_bug_ids = [t for t in target_bug_ids if total_bug_info[t][1] in target_bug_types]
                target_bug_ids.sort(key=lambda x: int(x))

                # we only analyze the bugs that in bug_dict
                print(f"{len(target_bug_ids)} {target_bug_types} bugs for {framework} with method {method}")

                for idx, target_bug_id in enumerate(target_bug_ids):
                    fail_json_paths = [
                        os.path.join(coverage_json_path, f"{args.framework}-{args.method}", target_bug_id,
                                        f"{framework_prefix}-{target_bug_id}-original_data_fail.json")]
                    pass_file_paths = [
                        os.path.join(coverage_json_path, f"{args.framework}-{args.method}", target_bug_id, f)
                        for f in
                        os.listdir(os.path.join(coverage_json_path, f"{args.framework}-{args.method}",
                                                target_bug_id))
                        if "_data_pass.json" in f]
                    try:
                        localization_res, _, _ = method_analyzer.bug_localization(fail_bug=target_bug_id,
                                                                                  pass_file_paths=pass_file_paths,
                                                                                  fail_json_paths=fail_json_paths,
                                                                                  formula=formula,
                                                                                  granularity_levels=granularity_levels,
                                                                                  aggregate_mode=aggregate_mode,
                                                                                  metric=metric_name)
                    except Exception as e:
                        raise Exception(f"method: {method}, bug:{target_bug_id}, time_slot:{time_slot}")
                    print(f"{idx + 1} Bug {target_bug_id} localization result: {localization_res}")
                    method_dict[target_bug_id] = localization_res

                method_df = localization_dataframe(method, method_dict)
                # calculate the improvements
                method_counter = top_counter(method_df, args.method)
                print(method_counter)
                metric_results1 = metric_calculator(method_df, args.method)
                print(metric_results1)

                statistic_df = statistic(method_counter, metric_results1, metric_name)
                print(statistic_df)
                # save the three dataframes into one excel file. distinguish it with different sheet_name

                report_path = os.path.join(result_dir,
                                           f"{exp_idntfr}-{framework}-{method}-{specificMethod}-{time_slot}s-{formula}-{aggregate_mode}-{'+'.join(target_bug_types)}-{metric_name}.xlsx")

                with pd.ExcelWriter(report_path) as writer:
                    method_df.to_excel(writer, sheet_name=args.method)
                    statistic_df.to_excel(writer, sheet_name="statistic")
