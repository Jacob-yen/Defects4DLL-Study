import os
from glob import glob
import configparser
import shutil
import subprocess
import pandas as pd
from src import framework_nicknames, root_path
from src.tools.logger_utils import LoggerUtils

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger
MAX_JOBS = 60
select = "fix"

class EnvBuilder:
    def __init__(self, framework, fail_bug, config_name) -> None:
        self.fail_bug = fail_bug
        self.framework = framework
        self.framework_prefix = framework_nicknames[framework]
        self.config = configparser.ConfigParser()
        self.config_name = config_name
        self.config.read(f'src/config/{config_name}.ini')

        self.bug_info = pd.read_excel(os.path.join(root_path, "data", f'{self.framework}-V7.xlsx'), sheet_name='Sheet1')
        selected_row = self.bug_info[self.bug_info['pr_id'] == int(self.fail_bug)]
        if not selected_row.empty:
            self.py_version = str(selected_row['py_version'].values[0])
            self.commit_hash = selected_row[select].values[0]
            if self.framework == "tensorflow":
                self.bazel_version = selected_row['bazel_version'].values[0]
        else:
            logger.error("test_patch is None or defect_api is None")
            raise ValueError(f"Bug {self.fail_bug} not found in {self.framework}-V4.xlsx")

        self.iterpreter_path = self.config.get('interpreter', 'conda_path')
        self.target_bug = self.framework_prefix + '-' + self.fail_bug
        self.python_interpreter = os.path.join(self.iterpreter_path, "envs", f"{self.target_bug}-{select}/bin/python")
        self.orig_bug_script_dir = os.path.join(root_path, f"data/{self.framework}", "Result", self.target_bug)
        self.bazel_path = self.config.get("compile", 'bazel_path')
        self.source_compile_dir = os.path.join(self.config.get("compile", 'source_compile_dir'))
        self.bug_source_compile_dir = os.path.join(self.source_compile_dir,
                                                   f"{self.framework_prefix}-project/{self.target_bug}")
        os.makedirs(self.bug_source_compile_dir, exist_ok=True)

    def setup(self):
        # 构建conda环境
        # self.create_conda_environment()
        # 源码编译构建环境
        return self.build_source_environment()
        # return True

    def create_conda_environment(self):
        env_name = f"{self.target_bug}-{select}"
        conda_cmd = f"conda create -n {env_name} python={self.py_version} -y"
        print(conda_cmd)
        process = subprocess.Popen(conda_cmd, shell=True)
        process.wait()

        requirement_name = "requirements.txt"
        if self.framework == "tensorflow":
            requirement_name = f"{self.py_version.replace('.', '')}-{requirement_name}"
        requirement_path = os.path.join(self.orig_bug_script_dir, requirement_name)

        pip_cmd = f"{self.python_interpreter} -m pip install -r {requirement_path} --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple/"
        process = subprocess.Popen(pip_cmd, shell=True)
        process.wait()

        # requirement_path = os.path.join(self.source_compile_dir, "wheel_result", self.framework, self.target_bug)
        #
        # whl_files = glob(os.path.join(requirement_path, "**", "*.whl"), recursive=True)
        #
        # if len(whl_files) == 1:
        #     pip_cmd = f"{self.python_interpreter} -m pip install {whl_files[0]} --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple/"
        #     process = subprocess.Popen(pip_cmd, shell=True)
        #     process.wait()
        # #
        # fail_file_path = os.path.join(self.orig_bug_script_dir, f"{self.target_bug}-original.py")
        # print(f"Reproduce the results of the program: {self.python_interpreter} {fail_file_path}")
        # process = subprocess.Popen(
        #     f"{self.python_interpreter} {fail_file_path}", shell=True)
        # process.wait()
        #
        #     # fail_cov_collector = CollectorWrapper(framework=self.framework, method_name='our',
        #     #                                       config_name=self.config_name, fail_bug=self.fail_bug, test_type="fail")
        #     # fail_cov_collector.process_single_file(fail_file_path)
        #
        # else:
        #     print(f"No wheel files found or the number of wheel files exceeds one !")
        #     logger.info(f"No wheel files found or the number of wheel files exceeds one !")

    def build_source_environment(self):
        pass


class EnvBuildWrapper(EnvBuilder):
    def __init__(self, framework, fail_bug, config_name) -> None:
        super().__init__(framework, fail_bug, config_name)
        if framework == "pytorch":
            self.builder = Torch_EnvBuilder(framework, fail_bug, config_name)
        elif framework == "tensorflow":
            self.builder = TF_EnvBuilder(framework, fail_bug, config_name)
        else:
            raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")

    def setup(self):
        return self.builder.setup()


def switch_to_on(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    is_ON = False
    for i in range(len(lines)):
        if 'option(USE_CPP_CODE_COVERAGE' in lines[i]:
            lines[i] = 'option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" ON)\n'
            is_ON = True
            break
    if not is_ON:
        lines.insert(0, 'SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage") \n')
        lines.insert(0, 'SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage") \n')
    with open(file_path, 'w') as f:
        f.write(''.join(lines))


def switch_to_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if 'option(USE_CPP_CODE_COVERAGE' in lines[i]:
            lines[i] = 'option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" OFF)\n'
            break
    with open(file_path, 'w') as f:
        f.write(''.join(lines))


class Torch_EnvBuilder(EnvBuilder):
    def __init__(self, framework, method_name, config_name) -> None:
        super().__init__(framework, method_name, config_name)

    def build_source_environment(self):
        # 在每个bug的路径下，gitclone下来对应文件（可以用我们origin的文件夹来代替这一步）
        # 然后执行对应的浮现流程，继承两个类
        project_path = os.path.join(self.bug_source_compile_dir, "pytorch")
        # if os.path.exists(project_path):
        #     shutil.rmtree(project_path)
        # 下面这几步release的时候改为git clone
        if int(self.fail_bug) > 59978:
            orig_bug_script_dir = os.path.join(self.source_compile_dir, "origin_2", "pytorch")
        else:
            orig_bug_script_dir = os.path.join(self.source_compile_dir, "origin_1", "pytorch")
            self.python_interpreter = "TORCH_CUDA_ARCH_LIST='7.5' " + self.python_interpreter
        # orig_bug_script_dir = os.path.join(self.source_compile_dir, "origin", "pytorch")
        # orig_bug_script_dir = "/home/jiangtianjie/workspace/origin/pytorch"
        shutil.copytree(orig_bug_script_dir, project_path, symlinks=True)

        process = subprocess.Popen(f"cd {project_path} && {self.python_interpreter} setup.py clean", shell=True)
        process.wait()

        process = subprocess.Popen(f"{self.python_interpreter} -m pip uninstall torch -y", shell=True)
        process.wait()

        # release的时候这一行应该变成对分支的切换
        print(f"cd {project_path} && git add . && git commit -m 'Refresh'")
        logger.info(f"cd {project_path} && git add . && git commit -m 'Refresh'")
        process = subprocess.Popen(
            f"cd {project_path} && git add . && git commit -m 'Refresh'", shell=True)
        process.wait()
        print(f"cd {project_path} && git checkout {self.commit_hash}")
        logger.info(f"cd {project_path} && git checkout {self.commit_hash}")
        process = subprocess.Popen(
            f"cd {project_path} && git checkout {self.commit_hash}", shell=True)
        process.wait()

        # switch_to_on(os.path.join(project_path, "CMakeLists.txt"))

        process = subprocess.Popen(
            f"cd {project_path} && git submodule sync && git submodule update --init --recursive", shell=True)
        process.wait()

        max_attempts = 1
        attempts = 0
        success = False
        while attempts < max_attempts and not success:
            process = subprocess.Popen(
                f"cd {project_path} && git submodule sync && git submodule update --init --recursive", shell=True)
            process.wait()
            return_code = process.returncode

            if return_code == 0:
                success = True
            else:
                attempts += 1
                print(f"Attempt {attempts} failed with return code {return_code}. Retrying...")
                logger.info(f"Attempt {attempts} failed with return code {return_code}. Retrying...")

        subprocess.call("export USE_CUDA=0", shell=True)

        process = subprocess.Popen(
            f"cd {project_path} && MAX_JOBS=60 {self.python_interpreter} setup.py bdist_wheel", shell=True)
        process.wait()

        success = False

        find_command = f"find {project_path}/dist -type f -name 'torch-*.whl' -print -quit"
        print(f"find_wheel_command: {find_command}")
        logger.info(f"find_wheel_command: {find_command}")
        try:
            whl_file = subprocess.check_output(find_command, shell=True, universal_newlines=True).strip()
            logger.info(f"success find whl_file: {self.target_bug}")
            success = True
        except subprocess.CalledProcessError as e:
            whl_file = ""
            logger.info(f"fail find whl_file: {self.target_bug}")

        process = subprocess.Popen(
            f"cd {project_path} && {self.python_interpreter} -m pip install {whl_file} s",
            shell=True)
        process.wait()

        fail_file_path = os.path.join(self.orig_bug_script_dir, f"{self.target_bug}-original.py")
        print(f"Reproduce the results of the program: {self.python_interpreter} {fail_file_path}")
        logger.info(f"Reproduce the results of the program: {self.python_interpreter} {fail_file_path}")
        process = subprocess.Popen(
            f"{self.python_interpreter} {fail_file_path}", shell=True)
        process.wait()

        return success


class TF_EnvBuilder(EnvBuilder):
    def __init__(self, framework, method_name, config_name) -> None:
        super().__init__(framework, method_name, config_name)

    def build_source_environment(self):
        project_path = os.path.join(self.bug_source_compile_dir, "tensorflow")
        # if os.path.exists(project_path):
        #     shutil.rmtree(project_path)
        # # 下面这几步release的时候改为git clone
        # orig_bug_script_dir = os.path.join(self.source_compile_dir, "origin", "tensorflow")
        # shutil.copytree(orig_bug_script_dir, project_path)
        #
        # shutil.copy(f"{self.bazel_path}/bazel-{self.bazel_version}-linux-x86_64", "/usr/bin/bazel")
        # process = subprocess.Popen(f"chmod 777 /usr/bin/bazel", shell=True)
        # process.wait()
        #
        # # release的时候这一行应该变成对分支的切换
        # process = subprocess.Popen(
        #     f"cd {project_path} && git add . && git commit -m 'Refresh'", shell=True)
        # process.wait()
        #
        # process = subprocess.Popen(
        #     f"cd {project_path} && git checkout {self.commit_hash}", shell=True)
        # process.wait()
        #
        # process = subprocess.Popen(f"cd {project_path} && bazel clean --expunge", shell=True)
        # process.wait()
        #
        # process = subprocess.Popen(f"{self.python_interpreter} -m pip uninstall tensorflow -y", shell=True)
        # process.wait()
        #
        # process = subprocess.Popen(
        #     f"cd {project_path} && yes '' | {self.python_interpreter} configure.py", shell=True)
        # process.wait()
        #
        # max_attempts = 1
        # attempts = 0
        # success = False
        # while attempts < max_attempts and not success:
        #     process = subprocess.Popen(
        #         f"cd {project_path} && bazel build --jobs=60 //tensorflow/tools/pip_package:build_pip_package",
        #         shell=True)
        #     process.wait()
        #     return_code = process.returncode
        #
        #     if return_code == 0:
        #         success = True
        #     else:
        #         attempts += 1
        #         print(f"Attempt {attempts} failed with return code {return_code}. Retrying...")
        #         logger.info(f"Attempt {attempts} failed with return code {return_code}. Retrying...")
        #
        # process = subprocess.Popen(
        #     f"cd {project_path} && bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg",
        #     shell=True)
        # process.wait()

        success = False

        find_command = f"find {project_path}/tmp/tensorflow_pkg -type f -name 'tensorflow-*.whl' -print -quit"
        print(f"find_wheel_command: {find_command}")
        logger.info(f"find_wheel_command: {find_command}")
        try:
            whl_file = subprocess.check_output(find_command, shell=True, universal_newlines=True).strip()
            success = True
            logger.info(f"success find whl_file: {self.target_bug}")
        except subprocess.CalledProcessError as e:
            whl_file = ""
            logger.info(f"fail find whl_file: {self.target_bug}")

        process = subprocess.Popen(
            f"cd {project_path} && {self.python_interpreter} -m pip install {whl_file}",
            shell=True)
        process.wait()

        fail_file_path = os.path.join(self.orig_bug_script_dir, f"{self.target_bug}-original.py")
        print(f"Reproduce the results of the program: {self.python_interpreter} {fail_file_path}")
        logger.info(f"Reproduce the results of the program: {self.python_interpreter} {fail_file_path}")
        process = subprocess.Popen(
            f"{self.python_interpreter} {fail_file_path}", shell=True)
        process.wait()

        return success