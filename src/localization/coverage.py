import sys
import os
import configparser
import json
from src import framework_nicknames
from src.tools import utils
from src.tools.logger_utils import LoggerUtils
from src.localization import python_function_and_block_spectrum, python_file_spectrum
from src.tools.enum_class import Approach, Framework, TestType
from copy import deepcopy
import subprocess
import shutil
import site
import time

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class BaseCollector:
    def __init__(self, framework, method_save_name, config_name, fail_bug="") -> None:
        self.framework = framework
        # NOTE: we currently do not consider coverage from c-level
        # assert lang in ["python", "c"], "language must be python or c"
        # self.lang = lang
        self.fail_bug = fail_bug
        self.config = configparser.ConfigParser()
        self.config.read(f'src/config/{config_name}.ini')
        self.root_result_dir = self.config.get('general', 'root_result_dir')
        self.coverage_intermedia = os.path.join(self.root_result_dir, "coverage_intermedia",
                                                f"{framework}-{method_save_name}")
        self.coverage_json = os.path.join(self.root_result_dir, 'coverage_json', f"{framework}-{method_save_name}")
        self.cov_json_path = os.path.join(self.coverage_json, self.fail_bug)
        os.makedirs(self.cov_json_path, exist_ok=True)
        self.interpreter_path = self.config.get('interpreter', 'conda_path')
        self.source_compile_dir = self.config.get("compile", 'source_compile_dir')
        self.spectrum_files = []
        os.makedirs(self.coverage_intermedia, exist_ok=True)
        os.makedirs(self.coverage_json, exist_ok=True)

    @staticmethod
    def delete_files(file_dir, pattern):
        for f in os.listdir(file_dir):
            if f.startswith(pattern):
                file_path = os.path.join(file_dir, f)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Fail to delete: {file_path}")
                    logger.error(e)
                    sys.exit(1)

    def extract_testfile_path(self, test_file,is_developer_file,test_type):
        # get file identifier of test file under different approaches
        test_file_parent_dir, test_file_name = os.path.split(test_file)
        execute_file = test_file
        logger.debug("Before: " + test_file_name)

        if is_developer_file and test_type == TestType.PASS:
            file_idntfr = utils.gen_md5_id(test_file)
        else:
            file_idntfr = test_file_name
            if file_idntfr.startswith("pass_file_"):
                file_idntfr = file_idntfr[10:]
            if file_idntfr.endswith(".py"):
                file_idntfr = file_idntfr[:-3]
        logger.debug("After: " + file_idntfr)
        # if self.test_type == TestType.FAIL:
        #     test_cov_res_name = f"data_{self.test_type}.json"
        # else:
        if is_developer_file:
            execute_file = " -m pytest " + test_file
        test_cov_res_name = f"{file_idntfr}_data_{test_type}.json"
        test_cov_res_file = os.path.join(self.coverage_json, self.fail_bug, test_cov_res_name)
        return execute_file, test_file_parent_dir, file_idntfr, test_cov_res_file

    def dump_spectrum_files(self, test_cov_res_file, data):
        # if self.test_type != TestType.FAIL:
        #     self.spectrum_files.append(test_cov_res_file)
        with open(test_cov_res_file, 'w', encoding='utf8') as f2:
            json.dump(data, f2, ensure_ascii=False, indent=2)

    def process_single_file(self, test_file,is_developer_file,test_type):
        execute_file, test_file_parent_dir, file_idntfr, test_cov_res_file = self.extract_testfile_path(test_file,
                                                                                                        is_developer_file,
                                                                                                        test_type)
        data, success_flag = self.collect_single_python_file(execute_file, test_file_parent_dir, self.fail_bug,
                                                             file_idntfr, test_type, test_cov_res_file,is_developer_file)
        if not success_flag:
            logger.error(f"{test_file} collect_single_file is failed!!")
            return None
        else:
            self.dump_spectrum_files(test_cov_res_file, data)
            return data

    def collect_single_file(self, **kwargs):
        pass

    def collect_single_python_file(self, **kwargs):
        return None, False

    def collect_single_c_file(self, **kwargs):
        pass

    def dist_file_similarity(self, target_file, fail_file):
        with open(target_file, 'r', encoding='utf8') as fp:
            target_data = json.load(fp)
        with open(fail_file, 'r', encoding='utf8') as fp:
            fail_data = json.load(fp)
        intersection_line_num = 0
        union_line_num = 0
        file_set = fail_data["files"].keys()
        for file in file_set:
            fail_set = set(fail_data["files"][file]["executed_lines_frequency"].keys())
            union_line_num += len(fail_set)
            if file in target_data["files"].keys():
                for line in fail_set:
                    if line in set(target_data["files"][file]["executed_lines_frequency"].keys()):
                        intersection_line_num += 1

        return round(intersection_line_num / union_line_num, 6)


class CollectorWrapper(BaseCollector):
    def __init__(self, framework, method_save_name, config_name='exp_config', fail_bug="") -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug)
        if framework == Framework.PYTORCH:
            self.collector = TorchCollector(framework, method_save_name, config_name, fail_bug)
        elif framework == Framework.TENSORFLOW:
            self.collector = TFCollector(framework, method_save_name, config_name, fail_bug)
        elif framework == Framework.JITTOR:
            self.collector = JITTORCollector(framework, method_save_name, config_name, fail_bug)
        elif framework == Framework.MXNET:
            self.collector = MXNETCollector(framework, method_save_name, config_name, fail_bug)
        elif framework == Framework.MINDSPORE:
            self.collector = MINDSPORECollector(framework, method_save_name, config_name, fail_bug)
        elif framework == Framework.PADDLE:
            self.collector = PADDLECollector(framework, method_save_name, config_name, fail_bug)
        else:
            raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")

    # def run(self, file_list, fail_bug, test_type):
    #     return self.collector.run(file_list, fail_bug, test_type)

    def process_single_file(self, test_file,is_developer_file,test_type):
        """
        Run a test file and collect the spectrum
        """
        return self.collector.process_single_file(test_file,is_developer_file=is_developer_file,test_type=test_type)

    def dist_file_similarity(self, target_file, fail_file):
        return self.collector.dist_file_similarity(target_file, fail_file)

    def get_spectrum_files_and_cov_json_file(self):
        return self.collector.spectrum_files, self.collector.cov_json_file


class TorchCollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TorchCollector.delete_files(tmp_cov_path, ".coverage")
        TorchCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"torch-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=torch {test_file_path}"
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=5 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            process.wait()
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f" {python_interpreter} -m  coverage combine &&  {python_interpreter} -m  coverage json -o {coverage_file_path}"
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        return results_data, True


class TFCollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug,) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug,)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TFCollector.delete_files(tmp_cov_path, ".coverage")
        TFCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"tf-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=tensorflow,tensorflow_core {test_file_path}"
        logger.info(f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}")
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=10 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f"{python_interpreter} -m coverage combine && {python_interpreter} -m coverage json -o {coverage_file_path}"
        logger.info(f"cd {tmp_cov_path} && {python_command2}")
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        # For tensorflow, we need to remove the bazel-out folder
        # shutil.rmtree(os.path.join(tmp_cov_path, "bazel-out"))
        return results_data, True

class JITTORCollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug,) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug,)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TFCollector.delete_files(tmp_cov_path, ".coverage")
        TFCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"jittor-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=jittor {test_file_path}"
        logger.info(f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}")
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=10 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f"{python_interpreter} -m coverage combine && {python_interpreter} -m coverage json -o {coverage_file_path}"
        logger.info(f"cd {tmp_cov_path} && {python_command2}")
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        # For tensorflow, we need to remove the bazel-out folder
        # shutil.rmtree(os.path.join(tmp_cov_path, "bazel-out"))
        return results_data, True

class MXNETCollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug,) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug,)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TFCollector.delete_files(tmp_cov_path, ".coverage")
        TFCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"mxnet-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=mxnet {test_file_path}"
        logger.info(f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}")
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=10 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f"{python_interpreter} -m coverage combine && {python_interpreter} -m coverage json -o {coverage_file_path}"
        logger.info(f"cd {tmp_cov_path} && {python_command2}")
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        # For tensorflow, we need to remove the bazel-out folder
        # shutil.rmtree(os.path.join(tmp_cov_path, "bazel-out"))
        return results_data, True

class MINDSPORECollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug,) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug,)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TFCollector.delete_files(tmp_cov_path, ".coverage")
        TFCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"ms-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=mindspore {test_file_path}"
        logger.info(f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}")
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=10 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f"{python_interpreter} -m coverage combine && {python_interpreter} -m coverage json -o {coverage_file_path}"
        logger.info(f"cd {tmp_cov_path} && {python_command2}")
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        # For tensorflow, we need to remove the bazel-out folder
        # shutil.rmtree(os.path.join(tmp_cov_path, "bazel-out"))
        return results_data, True

class PADDLECollector(BaseCollector):
    def __init__(self, framework, method_save_name, config_name, fail_bug,) -> None:
        super().__init__(framework, method_save_name, config_name, fail_bug,)

    def collect_single_python_file(self, test_file_path, test_file_parent_dir, fail_bug, file_prefix, test_type,
                                   file_name,is_developer_file):

        if is_developer_file:
            file_prefix = utils.gen_md5_id(file_prefix)
        tmp_cov_path = os.path.join(self.coverage_intermedia, f"{fail_bug}-{test_type}-{file_prefix}")
        if not os.path.exists(tmp_cov_path):
            os.mkdir(tmp_cov_path)
        TFCollector.delete_files(tmp_cov_path, ".coverage")
        TFCollector.delete_files(tmp_cov_path, "coverage")

        # 使用 subprocess 执行命令
        # get project path
        python_interpreter = os.path.join(self.interpreter_path, "envs", f"paddle-{fail_bug}-buggy/bin/python")
        python_command1 = f"{python_interpreter} -m coverage run -p --source=paddle {test_file_path}"
        logger.info(f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}")
        process = subprocess.Popen(
            f"cd {tmp_cov_path} && export PYTHONPATH={test_file_parent_dir}:$PYTHONPATH && {python_command1}",
            shell=True)
        try:
            process.wait(timeout=10 * 60)
        except subprocess.TimeoutExpired:
            logger.error(f"Command {python_command1} timed out after 5 minutes")
            process.terminate()
            kill_command = f"pkill -f '{python_command1}'"
            logger.info(f"{kill_command} is running!")
            process = subprocess.Popen(kill_command, shell=True)
            return None, False
        except Exception as e:
            process.terminate()
            logger.error(f"Exception raised when executing {python_command1}")
            logger.error(str(e))
            return None, False

        coverage_file_path = os.path.join(tmp_cov_path, "coverage.json")
        python_command2 = f"{python_interpreter} -m coverage combine && {python_interpreter} -m coverage json -o {coverage_file_path}"
        logger.info(f"cd {tmp_cov_path} && {python_command2}")
        process = subprocess.Popen(f"cd {tmp_cov_path} && {python_command2}", shell=True)
        process.wait()

        if not os.path.exists(coverage_file_path):
            return None, False

        results_data = {"files": {}}
        with open(coverage_file_path, 'r', encoding='utf8') as fp:
            coverage_json_data = json.load(fp)
        # get the file spectrum
        site_package_prefix, results_data = python_file_spectrum(results_data, coverage_json_data)
        # manually add the function and block spectrum
        results_data = python_function_and_block_spectrum(results_data, site_package_prefix)
        # For tensorflow, we need to remove the bazel-out folder
        # shutil.rmtree(os.path.join(tmp_cov_path, "bazel-out"))
        return results_data, True