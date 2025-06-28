import os
import configparser
import shutil
from src.schedule.hybrid_schedule import HybridScheduler
from src.schedule.baseline_schedule import BaselineScheduler
from src.localization.coverage import CollectorWrapper
from src import framework_nicknames, root_path
from src.tools.logger_utils import LoggerUtils
from src.tools.enum_class import Approach, Framework
from src.tools.enum_class import TestType
import subprocess

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class Worker:
    def __init__(self, fail_bug, framework, symptom, method_name, config_name) -> None:
        self.fail_bug = fail_bug
        self.framework = framework
        # self.lang = lang
        self.symptom = symptom
        self.framework_prefix = framework_nicknames[framework]
        self.method_name = method_name
        self.config = configparser.ConfigParser()
        self.config_name = config_name
        self.config.read(f'src/config/{config_name}.ini')
        self.source_compile_dir = self.config.get("compile", 'source_compile_dir')
        self.interpreter_path = self.config.get('interpreter', 'conda_path')
        self.root_result_dir = self.config.get("general", 'root_result_dir')
        self.time_limit_seconds = float(self.config.get("general", 'time_limit_seconds'))
        self.max_test_case_count = float(self.config.get("general", 'max_test_case_count'))
        self.target_bug = self.framework_prefix + '-' + self.fail_bug
        self.fail_test_path = None

    def initialize_bug_folder(self, bug_save_dir):
        self.fail_test_path = os.path.join(bug_save_dir, self.target_bug, f"{self.target_bug}-original.py")
        bug_run_dir = os.path.join(bug_save_dir, self.target_bug)
        orig_bug_script_dir = os.path.join(root_path, f"data/{self.framework}", "Result", self.target_bug)
        if not os.path.exists(orig_bug_script_dir):
            raise FileExistsError(f"Original bug info dir: {orig_bug_script_dir} not exists")
        if os.path.exists(bug_run_dir):
            shutil.rmtree(bug_run_dir)
        shutil.copytree(orig_bug_script_dir, bug_run_dir)

    def run(self):
        if not self.check_env():
            logger.error(f"Runtime environment is not ready for bug {self.fail_bug} on {self.framework}!")
        else:
            if self.method_name in [Approach.HYBRID, Approach.RULE]:
                self.run_hybrid()
            elif self.method_name in Approach.BASELINE:
                self.run_baseline()
            else:
                raise ValueError(f"method_name: {self.method_name} is not supported!")

    def run_hybrid(self):
        # initialize the configuration
        hybrid_techniques = self.config.get(self.method_name, "hybrid_techniques").split(",")
        hybrid_techniques = [t.strip() for t in hybrid_techniques]
        logger.info(f"hybrid_techniques: {hybrid_techniques}")
        mutation_levels = self.config.get(self.method_name, "mutation_levels").split(",")
        mutation_levels = [t.strip() for t in mutation_levels if t.strip() != ""]
        method_save_name = "+".join(hybrid_techniques)
        bug_save_dir = os.path.join(self.root_result_dir, method_save_name)

        self.initialize_bug_folder(bug_save_dir)

        cov_collector = CollectorWrapper(framework=self.framework, method_save_name=method_save_name,
                                         config_name=self.config_name, fail_bug=self.fail_bug)
        fail_spectrum = cov_collector.process_single_file(self.fail_test_path, is_developer_file=False, test_type=TestType.FAIL)

        # generate the mutation
        max_mutation_order = self.config.getfloat(self.method_name, 'max_mutation_order')
        mutator_selection = self.config.get(self.method_name, "mutator_selection")

        if self.method_name == Approach.HYBRID:
            api_url = self.config.get(Approach.HYBRID, "api_url")
            api_key = self.config.get(Approach.HYBRID, "api_key")
            model_name = self.config.get(Approach.HYBRID, "model_name")
            temperature = self.config.getfloat(Approach.HYBRID, "temperature")
        else:
            api_url = None
            api_key = None
            model_name = None
            temperature = None
        scheduler = HybridScheduler(root_result_dir=self.root_result_dir, bug_save_dir=bug_save_dir,
                                    method_save_name=method_save_name,framework=self.framework,
                                    fail_bug=self.fail_bug, max_test_case_count=self.max_test_case_count,
                                    time_limit_seconds=self.time_limit_seconds, config=self.config,
                                    cov_collector=cov_collector, interpreter_path=self.interpreter_path,
                                    source_compile_path=self.source_compile_dir,
                                    symptom=self.symptom, max_mutation_order=max_mutation_order,
                                    mutator_selection=mutator_selection, hybrid_techniques=hybrid_techniques,
                                    api_url=api_url,api_key=api_key, model_name=model_name, temperature=temperature,
                                    fail_spectrum=fail_spectrum, mutation_levels=mutation_levels)

        scheduler.mutation_loop()

        scheduler.mutation_status()

    def run_baseline(self):
        bug_save_dir = os.path.join(self.root_result_dir, self.method_name)
        self.initialize_bug_folder(bug_save_dir)

        is_developer_file = self.method_name == Approach.DEVELOPER
        cov_collector = CollectorWrapper(framework=self.framework, method_save_name=self.method_name,
                                         config_name=self.config_name, fail_bug=self.fail_bug)
        cov_collector.process_single_file(self.fail_test_path, is_developer_file=False, test_type=TestType.FAIL)

        scheduler = BaselineScheduler(root_result_dir=self.root_result_dir, bug_save_dir=bug_save_dir,
                                      method_save_name=self.method_name, framework=self.framework,
                                      fail_bug=self.fail_bug, max_test_case_count=self.max_test_case_count,
                                      time_limit_seconds=self.time_limit_seconds, config=self.config,
                                      cov_collector=cov_collector, interpreter_path=self.interpreter_path,
                                      source_compile_path=self.source_compile_dir)

        scheduler.process_files(is_developer_file,test_type=TestType.PASS)
        scheduler.status()

    def check_env(self):
        """
        check and install the environment.
        """
        is_conde_env_exists = False
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"{self.target_bug}-buggy/bin/python")
        if self.framework == Framework.PYTORCH:
            framework_env = "torch"
        elif self.framework == "tensorflow":
            framework_env = "tensorflow"
        elif self.framework == "jittor":
            framework_env = "jittor"
        elif self.framework == "mxnet":
            framework_env = "mxnet"
        elif self.framework == "mindspore":
            framework_env = "mindspore"
        elif self.framework == "paddlepaddle":
            framework_env = "paddle"
        env_check_command = f"{python_interpreter} -m pip list | grep {framework_env}"
        process = subprocess.Popen(env_check_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            line = stdout.decode().strip()
            line_list = line.splitlines()
            for line_str in line_list:
                if framework_env in line_str.split(" ")[0]:
                    is_conde_env_exists = True

        if not is_conde_env_exists:
            logger.error(f"{self.target_bug} 's conde_env is not exists!")

        return is_conde_env_exists
