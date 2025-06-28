import json
import os
import sys
from collections import defaultdict

import pandas as pd
import subprocess
import random
import time
import shutil
from src import framework_nicknames, root_path
from src.tools import utils
from src.tools.logger_utils import LoggerUtils
from src.schedule import AbstractScheduler
from src.baseline import extract_lib_api, extract_lib_api_with_test_file, \
    find_class_and_function, get_pytest_path
from src.tools.enum_class import Approach, Framework

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class BaselineScheduler(AbstractScheduler):
    def __init__(self, root_result_dir, bug_save_dir, framework, method_save_name, fail_bug, max_test_case_count,
                 time_limit_seconds, config, cov_collector, interpreter_path, source_compile_path) -> None:

        super().__init__(root_result_dir=root_result_dir, bug_save_dir=bug_save_dir, method_save_name=method_save_name,
                         framework=framework, fail_bug=fail_bug, max_test_case_count=max_test_case_count,
                         time_limit_seconds=time_limit_seconds, config=config, cov_collector=cov_collector,
                         interpreter_path=interpreter_path, source_compile_path=source_compile_path)

        self.framework_prefix = framework_nicknames[framework]
        self.target_bug = self.framework_prefix + '-' + self.fail_bug
        if self.method_save_name in [Approach.DEEPREL, Approach.DOCTER]:
            self.code_file_dir = self.config.get(self.method_save_name, 'code_file_dir')
            self.all_file_dir = self.config.get(self.method_save_name, 'all_file_dir')
            self.all_file_root_path = f"{self.all_file_dir}/{self.framework}/conform_constr"

            self.docter_source_dir = f"{self.all_file_dir}/{self.framework}/conform_constr{self.fail_bug}"
            self.docter_backup_dir = f"{self.all_file_dir}/{self.framework}_copy/conform_constr{self.fail_bug}"
            self.deeprel_source_dir = f"{self.all_file_dir}/{self.framework}/expr/{self.target_bug}"
            self.deeprel_backup_dir = f"{self.all_file_dir}/{self.framework}_copy/expr/{self.target_bug}"

            if os.path.exists(self.docter_source_dir):
                shutil.rmtree(self.docter_source_dir)
            if os.path.exists(self.docter_backup_dir):
                shutil.rmtree(self.docter_backup_dir)
            if os.path.exists(self.deeprel_source_dir):
                shutil.rmtree(self.deeprel_source_dir)
            if os.path.exists(self.deeprel_backup_dir):
                shutil.rmtree(self.deeprel_backup_dir)

        self.python_interpreter = os.path.join(self.interpreter_path, "envs", f"{self.target_bug}-buggy/bin/python")
        self.bug_info = pd.read_excel(os.path.join(root_path, "data", f'{self.framework}-V6.xlsx'), sheet_name='Sheet1')

        # selected_row = self.bug_info[self.bug_info['pr_id'] == int(self.fail_bug)]
        # if not selected_row.empty:
        #     self.test_patch = selected_row['test_patch'].values[0]
        # else:
        #     logger.error("test_patch is None or defect_api is None")
        #     raise ValueError(f"Bug {self.fail_bug} not found in {self.framework}-V6.xlsx")

        self.fail_test_path = os.path.join(self.bug_save_dir, self.target_bug, f"{self.target_bug}-original.py")
        self.run_tmp_path = os.path.join(self.bug_save_dir, self.target_bug, "tmp")
        self.bug_run_dir = os.path.join(self.bug_save_dir, self.target_bug)
        os.makedirs(self.run_tmp_path, exist_ok=True)
        self.pass_file_path_txt = os.path.join(self.bug_save_dir, self.target_bug, "history_pass_file_list.txt")

        self.tmp_exec = os.path.join(self.root_result_dir, "coverage_intermedia",
                                     f"tmp-exec-{framework}-{self.method_save_name}")
        if not os.path.exists(self.tmp_exec):
            os.mkdir(self.tmp_exec)
        with open(self.pass_file_path_txt, "w") as file:
            file.write(f"{time.time()},start\n")

        self.pass_pool = []

        self.api_all = extract_lib_api(self.framework, self.target_bug)
        self.all_history = defaultdict(int)
        self.total_generate_duration = 0  # 累计生成时间
        self.total_execute_duration = 0  # 累计执行时间
        # logger.info(f"self.api_all: {self.api_all}")

        if self.method_save_name == Approach.DEVELOPER:
            # get all test cases provided by developer
            self.test_files = []

            test_files_path = os.path.join(self.source_compile_path, f"{self.framework_prefix}-project",
                                           self.target_bug,
                                           self.framework)

            if framework == Framework.PYTORCH:
                process = subprocess.Popen(f"find {test_files_path} -name 'test_*.py' | grep -v 'third_party'",
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
            else:
                process = subprocess.Popen(f"find {test_files_path} -name '*_test.py'", shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                line = stdout.decode().strip()
                self.test_files = line.splitlines()

            self.api_test_files_dict = {}

    def record_pass_file(self, file_path):
        with open(self.pass_file_path_txt, "a+") as file:
            file.write(f"{time.time()},{file_path}\n")

    def rename_baseline_file(self, file_path):
        logger.debug(f"change file name for method {self.method_save_name}: {file_path}")
        if self.method_save_name == Approach.DEEPREL:
            # deeprel pass file name is like:
            # /xxx/xxx/expr/torch-65926/torch.randperm+torch.blackman_window+rel+0/success/1.py
            api_name = file_path.split("/")[-3]
            new_file_name = f"pass_file_{api_name}_{utils.gen_md5_id(str(int(time.time())))}.py"
            return new_file_name
        elif self.method_save_name == Approach.DOCTER:
            # docter pass file name is like:
            # /home/workdir/pytorch/conform_constr65926/torch.randperm.yaml_workdir/{md5_code}.py
            api_name = file_path.split("/")[-2]
            new_file_name = f"pass_file_{api_name}_{utils.gen_md5_id(str(int(time.time())))}.py"
            return new_file_name
        else:
            raise ValueError("Currently we only rename the test cases by deeprel and docter")

    def process_files(self,is_developer_file,test_type):
        start_time = time.time()
        file_num = 1
        tmp_cov_path = os.path.join(self.tmp_exec, f"{self.fail_bug}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        while True:
            if time.time() - start_time >= self.time_limit_seconds:
                logger.info("Time limit reached. Mutation process terminated.")
                break
            file_num = file_num + 1
            self.all_history["usage"] += 1
            start_generate = time.time() 
            pass_file_path = self.get_pass_file(start_time)
            self.total_generate_duration += time.time() - start_generate
            start_execute = time.time()  
            logger.info(pass_file_path)

            if pass_file_path is None:
                self.total_execute_duration += time.time() - start_execute
                break
            self.all_history["generate"] += 1
            if self.method_save_name != Approach.DEVELOPER:
                command = f"{self.python_interpreter} {pass_file_path}"
            else:
                command = f"{self.python_interpreter} -m pytest {pass_file_path}"

            process = subprocess.Popen(f"cd {tmp_cov_path} && {command}", shell=True)
            try:
                process.wait(timeout=5 * 60)
                if process.returncode == 0:
                    logger.info(f"SUCCESS: {self.target_bug} {pass_file_path} PASS")
                    self.all_history["pass"] += 1
                else:
                    process.terminate()
                    logger.error(f"Command '{command}' returned non-zero exit status {process.returncode}:")
                    self.all_history["fail"] += 1
                    self.total_execute_duration += time.time() - start_execute
                    continue
            except subprocess.TimeoutExpired:
                logger.error("Command timed out after 5 minutes")
                process.terminate()
                kill_command = f"pkill -f '{command}'"
                logger.info(f"{kill_command} is running!")
                process = subprocess.Popen(kill_command, shell=True)
                process.wait()
                self.all_history["fail"] += 1
                self.total_execute_duration += time.time() - start_execute
                continue
            except Exception as e:
                process.terminate()
                logger.error(f"Exception raised when executing {command}")
                logger.error(str(e))
                self.all_history["fail"] += 1
                self.total_execute_duration += time.time() - start_execute
                continue

            if self.method_save_name == Approach.DEVELOPER:
                # save the pass file
                logger.debug(f"pass_file_path:{pass_file_path}")
                # the pass_file_path is like test_xxx_file.py::test_class::test_name
                file_path, test_class_name, test_func_name = pass_file_path.split("::")

                # get the file_name from the file_path
                file_name = os.path.basename(file_path)
                # get the function code
                func_code = utils.get_func_code_by_name(file_path, test_class_name, test_func_name)
                # save the function
                save_name = f"pass_file_{utils.gen_md5_id(pass_file_path)}.py"
                save_path = os.path.join(self.bug_save_dir, self.target_bug, save_name)
                utils.write_text(save_path, func_code)

            elif self.method_save_name in [Approach.DEEPREL, Approach.DOCTER]:
                target_file_name = self.rename_baseline_file(pass_file_path)
                target_dir = os.path.join(self.bug_save_dir, self.target_bug)
                save_path = os.path.join(target_dir, target_file_name)
                shutil.copy(pass_file_path, save_path)
                pass_file_path = save_path

            else:
                self.total_execute_duration += time.time() - start_execute
                raise ValueError(f"No such baseline: {self.method_save_name}")

            self.cov_collector.process_single_file(pass_file_path,is_developer_file,test_type=test_type)
            self.record_pass_file(save_path)
            self.total_execute_duration += time.time() - start_execute
        # we should remove the tmp_cov_path
        shutil.rmtree(tmp_cov_path)

    def get_pass_file(self, start_time):
        pass_file_path = None
        if self.method_save_name == Approach.DEVELOPER:
            pass_file_path = self.get_developer_file(start_time)
        elif self.method_save_name == Approach.DEEPREL:
            pass_file_path = self.get_deeprel_file(start_time)
        elif self.method_save_name == Approach.DOCTER:
            pass_file_path = self.get_docter_file(start_time)
        else:
            logger.error("We do not support the current method!")
        return pass_file_path

    def get_developer_file(self, start_time):

        # random developer
        if self.api_test_files_dict == {}:
            for test_file in self.test_files:
                try:
                    test_file_api_dict = extract_lib_api_with_test_file(self.framework, test_file)
                except Exception as e:
                    continue
                for api in test_file_api_dict.keys():
                    if api not in self.api_test_files_dict.keys():
                        self.api_test_files_dict[api] = {}
                    self.api_test_files_dict[api][test_file] = test_file_api_dict[api]

        while True:
            if time.time() - start_time >= self.time_limit_seconds:
                logger.info("Time limit reached. Mutation process terminated.")
                return None

            if len(self.test_files) == 0:
                return None

            api = random.choice(list(self.api_test_files_dict.keys()))

            test_files_dict = self.api_test_files_dict[api]
            source_test_file = random.choice(list(test_files_dict.keys()))
            line_dict = random.choice(test_files_dict[source_test_file])

            folder_path = os.path.join(self.bug_save_dir, self.target_bug)
            os.makedirs(folder_path, exist_ok=True)
            shutil.copy(source_test_file, folder_path)
            if self.framework == Framework.TENSORFLOW:
                test_file = os.path.join(folder_path, os.path.basename(source_test_file))
            else:
                test_file = source_test_file
            logger.debug(source_test_file)

            self.api_test_files_dict[api][source_test_file].remove(line_dict)
            if len(self.api_test_files_dict[api][source_test_file]) == 0:
                del self.api_test_files_dict[api][source_test_file]
            if len(self.api_test_files_dict[api]) == 0:
                del self.api_test_files_dict[api]

            class_name, function_name = find_class_and_function(test_file, line_dict[0], line_dict[1])
            if class_name is None or function_name is None:
                continue

            pass_file_path = get_pytest_path(test_file, class_name, function_name, self.python_interpreter)
            if pass_file_path is None or pass_file_path == "":
                continue

            if utils.gen_md5_id(pass_file_path) not in self.pass_pool:
                self.pass_pool.append(utils.gen_md5_id(pass_file_path))
                logger.info(f"api: pass_file_path - {api}: {pass_file_path}")
                return pass_file_path
            else:
                continue

    def get_deeprel_file(self, start_time):

        while True:
            if time.time() - start_time >= self.time_limit_seconds:
                logger.info("Time limit reached. Mutation process terminated.")
                return None

            # flush the save dir
            if os.path.exists(self.deeprel_source_dir):
                os.makedirs(os.path.dirname(self.deeprel_backup_dir), exist_ok=True)
                os.system(f"cp -ru {self.deeprel_source_dir}/ {self.deeprel_backup_dir}")
                shutil.rmtree(self.deeprel_source_dir)

            # invoke deeprel
            command = f"{os.path.join(self.interpreter_path, 'envs/DeepREL/bin/python')} " \
                      f"{self.code_file_dir}/{self.framework}/src/baseline_run.py " \
                      f"{self.target_bug} "

            logger.debug(command)
            try:
                result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True,
                                        check=True)
                logger.debug(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}:")

            # if the test cases exist, return the first one
            pytest_paths = []
            process = subprocess.Popen(
                f"find {self.all_file_dir}/{self.framework}/expr/{self.target_bug} -name '*py' | grep success",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                line = stdout.decode().strip()
                pytest_paths = line.splitlines()
            if len(pytest_paths) != 0:
                return pytest_paths[0]

    def get_docter_file(self, start_time):

        if self.framework == Framework.PYTORCH:
            covered_file_dir = f"{self.code_file_dir}/all_constr/pt1.5"
        else:
            covered_file_dir = f"{self.code_file_dir}/all_constr/tf2.1"

        while True:
            if time.time() - start_time >= self.time_limit_seconds:
                logger.info("Time limit reached. Mutation process terminated.")
                return None

            if os.path.exists(self.docter_source_dir):
                os.makedirs(self.docter_backup_dir, exist_ok=True)
                os.system(f"cp -ru {self.docter_source_dir}/ {self.docter_backup_dir}")
                shutil.rmtree(self.docter_source_dir)

            command = f"cd {self.run_tmp_path} && bash {self.code_file_dir}/run_fuzzer.sh " \
                      f"{self.framework} {covered_file_dir} " \
                      f"{self.code_file_dir}/configs/ci.config {self.fail_bug}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

            pytest_paths = []
            search_command = f"find {self.all_file_dir}/{self.framework}/conform_constr{self.fail_bug} -name '*py'"
            process = subprocess.Popen(
                search_command, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            logger.debug(search_command)
            if process.returncode == 0:
                line = stdout.decode().strip()
                pytest_paths = line.splitlines()

            if len(pytest_paths) != 0:
                return pytest_paths[0]

    def status(self):
        # Save the history information to a file
        output_data = {
            # 'mutation_history': self.mutation_history,
            'all_history': self.all_history,
            'total_generate_duration': self.total_generate_duration,
            'total_execute_duration': self.total_execute_duration
        }
        # Construct the file path with the target bug name
        filename = f"{self.target_bug}_{self.method_save_name}_status.json"
        filepath = os.path.join(self.bug_run_dir, filename)

        # Save data to the file as JSON
        try:
            with open(filepath, 'w') as file:
                json.dump(output_data, file, indent=4)
            logger.info(f"Mutation history and all history saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save mutation history: {e}")


if __name__ == '__main__':
    pass
