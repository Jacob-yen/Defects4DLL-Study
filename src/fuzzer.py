import argparse
import os
import sys
sys.path.append(os.getcwd())
import configparser
import shutil
import argparse
from src.tools.enum_class import Framework,Approach
from src.tools.logger_utils import LoggerUtils
from datetime import datetime

if __name__ == '__main__':

    # console parameters
    parser = argparse.ArgumentParser(description="Experiments Script For Rebug")
    parser.add_argument("--framework", type=str, default=Framework.PYTORCH, choices=Framework.TOTAL)
    parser.add_argument("--bug_list", nargs='+', type=str, default=["66170"])
    parser.add_argument("--all_bug", action='store_true', help='run all the bug')
    parser.add_argument("--method_name", type=str, choices=Approach.TOTAL)
    parser.add_argument("--config_name", type=str, default="tdd_config")
    parser.add_argument("--log_name", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--reuse_dir", action='store_true', help='reuse the dir')
    parser.add_argument("--random_seed", type=int)

    args = parser.parse_args()
    # set the random seed for math, random, numpy
    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    s = datetime.now()
    # define the global logger.
    logger_instance = LoggerUtils.get_instance(log_name=args.log_name, log_dir=args.log_dir)
    logger = logger_instance.logger
    # import the packages after the logger is initialized
    from src import parse_bug_info, framework_nicknames

    # load config
    config = configparser.ConfigParser()
    config.read(f'src/config/{args.config_name}.ini')
    # load root result dir
    root_result_dir = config.get('general', 'root_result_dir')
    os.makedirs(root_result_dir, exist_ok=True)
    # copy the config file to the result dir
    shutil.copy(f'src/config/{args.config_name}.ini', root_result_dir)
    result_dir = os.path.join(root_result_dir,args.method_name)
    # empty result folder
    if os.path.exists(result_dir) and not args.reuse_dir:
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    framework_prefix = framework_nicknames[args.framework]
    bug_info = parse_bug_info()
    total_bug_list = bug_info[framework_prefix]
    run_bug = total_bug_list if args.all_bug else args.bug_list
    for fail_bug in run_bug:
        assert fail_bug in total_bug_list, f"{fail_bug} is not in the bug list for {args.framework}"

    # import worker after the logger is initialized
    from src.worker import Worker
    # print process bar
    for fail_bug in run_bug:
        lang, symptom = bug_info[framework_prefix][fail_bug]
        logger.info(f"fail_bug: {fail_bug}, Library: {args.framework}, LANG: {lang}, Symptom: {symptom}")
        worker = Worker(framework=args.framework, fail_bug=fail_bug, symptom=symptom, method_name=args.method_name,
                        config_name=args.config_name)
        # generate the test case and collect the spectrum
        worker.run()

    logger.info(f"Total time: {datetime.now() - s}")
