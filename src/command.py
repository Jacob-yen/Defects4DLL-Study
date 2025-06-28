import os
import sys

sys.path.append(os.getcwd())
from src import bug_dict

# kill all python: ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
if __name__ == '__main__':
    # framework = "tensorflow"

    import sys


    # cuda_version = "cuda10"
    # framework = "pytorch"
    framework = sys.argv[1]
    cuda_version = sys.argv[2]
    random_seed = 20240223



    config_names = {"hybrid_config-900": random_seed}

    methods = ["hybrid"]
    log_time_idntfr = "20250626"


    cmd_cnt = 0
    extra_time_buffer = 300
    last_cmd_sleep = False
    if framework == "tensorflow":
        sleep_interval = 4
    elif framework == "pytorch":
        sleep_interval = 2
        print("echo 'pytorch bugs 39108 and 39153 can only be reproduced with GPUs before 30xx series. '")
        print("echo 'Modify the bug list in src/__init__.py if you want to reproduce them.'")
    else:
        sleep_interval = 2
        # run_time = 900
    clean_command = "ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9"
    for run_idx, (config_name, random_seed) in enumerate(config_names.items()):

        log_dir = f"/data/defects4dll-log/{config_name}-{'+'.join(methods)}_{random_seed}-{log_time_idntfr}"
        print(f"# {config_name}")
        run_time = int(config_name.split("-")[-1])
        for idx, method in enumerate(methods):
            bug_types = ["crash"]

            bug_list = []
            for bug_type in bug_types:
                bug_list += bug_dict[framework][cuda_version][bug_type]

            bug_list.sort(key=lambda x: int(x))
            print(f"## {method}: {len(bug_list)} bugs. Type:{bug_types}")

            for index, bug in enumerate(bug_list):
                command = f'nohup python -u src/fuzzer.py --framework {framework} --bug_list "{bug}" --method_name {method} --config_name {config_name} --reuse_dir --log_name "{cuda_version}-{method}-{framework}-{bug}-{index + 1}" --log_dir {log_dir} --random_seed {random_seed} > /dev/null 2>&1 &'
                # command = f'python -u src/fuzzer.py --framework {framework} --bug_list "{bug}" --method_name {method} --config_name {config_name} --reuse_dir --log_name "{cuda_version}-{method}-{framework}-{bug}-{index + 1}" --log_dir {log_dir} --random_seed {random_seed}'
                print(command)
                cmd_cnt += 1
                last_cmd_sleep = False
                print("\nsleep 2\n")

                if cmd_cnt % sleep_interval == 0 and cmd_cnt < len(config_names) * len(methods) * len(bug_list) - 1:
                    last_cmd_sleep = True
                    print(f"sleep {run_time + extra_time_buffer}")
