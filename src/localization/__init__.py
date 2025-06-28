import os
import subprocess
import re
import json
import shutil
import ast
from scalpel.cfg import CFGBuilder
from src.tools.logger_utils import LoggerUtils
import fnmatch
from copy import deepcopy

class Infosave:
    def __init__(self, framework, split_name) -> None:
        self.info = {
            'files': {},
        }
        self.SF = None
        self.func_save = None
        self.line_save = None
        self.branch_save = None
        self.framework = framework
        self.split_name = split_name

    def create_file(self):
        self.func_save = {}
        self.line_save = {}
        self.branch_save = {}
        self.info['files'][self.SF] = {
            "executed_lines_frequency": {},
            "missing_lines": [],
            "executed_functions_frequency": {},
            "missing_functions": []
        }

    def add_function(self, function_name, executed_cnt=None, line_num=None):
        if function_name in self.func_save:
            if executed_cnt is not None:
                self.func_save[function_name]['exe_cnt'] = executed_cnt
            if line_num is not None:
                self.func_save[function_name]['line_num'] = line_num
        else:
            self.func_save[function_name] = {
                'exe_cnt': executed_cnt,
                'line_num': line_num,
            }

    def add_line(self, line_num: int, executed_cnt: int):
        if executed_cnt > 0:
            self.line_save[line_num] = executed_cnt
        else:
            if line_num not in self.line_save.keys():
                self.line_save[line_num] = executed_cnt

    def add_branch(self, branch_line_num: int, executed_cnt: int):
        self.branch_save[branch_line_num] = executed_cnt

    def solve_TN(self, line):
        pass

    def solve_SF(self, line):
        self.SF = None
        if self.framework == "pytorch":
            self.SF = "/pytorch" + line.split(':')[1].split("pytorch")[1]
        elif self.framework == "tensorflow":
            sf_str = line.split(':')[1]
            if self.split_name in sf_str:
                sf_str = sf_str.split(self.split_name)[1]
            if "/bazel-out/k8-opt/bin" in sf_str:
                sf_str = sf_str.split("/bazel-out/k8-opt/bin")[1]
            if "/bazel-out/k8-opt/bin" in sf_str:
                sf_str = sf_str.split("/bazel-out/k8-opt/bin")[1]
            sf_str = "/tensorflow" + sf_str.split("tensorflow")[-1]
            self.SF = sf_str
        self.create_file()
        

    def solve_FN(self, line):
        elements = re.split(':|,', line)
        line_num = str(elements[1].strip())
        function_name = elements[2].strip()
        self.add_function(function_name, line_num=line_num)

    def solve_FNDA(self, line):
        elements = re.split(':|,', line)
        executed_cnt = int(elements[1].strip())
        function_name = elements[2].strip()
        self.add_function(function_name, executed_cnt=executed_cnt)

    def solve_FNF(self, line):
        pass

    def solve_FNH(self, line):
        pass

    def solve_BRDA(self, line):
        pass

    def solve_BRF(self, line):
        pass

    def solve_BRH(self, line):
        pass

    def solve_DA(self, line):
        elements = re.split(':|,', line)
        line_num = str(elements[1].strip())
        executed_cnt = int(elements[2].strip())
        self.add_line(line_num, executed_cnt)

    def solve_LF(self, line):
        pass

    def solve_LH(self, line):
        pass

    def solve_end_of_record(self, line):
        for key, value in self.func_save.items():
            line_num_str = str(value['line_num'])
            if line_num_str in self.info['files'][self.SF]['executed_functions_frequency'].keys():
                continue
            if value['exe_cnt'] > 0:
                self.info['files'][self.SF]['executed_functions_frequency'][line_num_str] = 1
                if line_num_str in self.info['files'][self.SF]['missing_functions']:
                    self.info['files'][self.SF]['missing_functions'].remove(line_num_str)
            else:
                self.info['files'][self.SF]['missing_functions'].append(line_num_str)
        for key, value in self.line_save.items():
            if value > 0:
                self.info['files'][self.SF]['executed_lines_frequency'][key] = 1
            else:
                self.info['files'][self.SF]['missing_lines'].append(key)

        del self.SF
        del self.func_save
        del self.line_save
        del self.branch_save
        self.SF = None
        self.func_save = None
        self.line_save = None
        self.branch_save = None

    def read_info_file(self, file_path):
        solve_functions = {
            "TN": self.solve_TN,
            "SF": self.solve_SF,
            "FN": self.solve_FN,
            "FNDA": self.solve_FNDA,
            "FNF": self.solve_FNF,
            "FNH": self.solve_FNH,
            "BRDA": self.solve_BRDA,
            "BRF": self.solve_BRF,
            "BRH": self.solve_BRH,
            "DA": self.solve_DA,
            "LF": self.solve_LF,
            "LH": self.solve_LH,
            "end_of_record": self.solve_end_of_record,
        }
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if len(line) == 0:
                continue
            operator = line.split(':')[0]
            if operator in solve_functions.keys():
                solve_functions[operator](line)

    def export(self):
        return self.info
        # component = json.dumps(self.info, sort_keys=False, indent=4, separators=(', ', ': '))
        # with open(file_path, "w") as f:
        #     f.write(component)


def get_ignored_pattern():
    return ['*.h', '/usr/*', '*anaconda3/*', '*third_party/*', '*tmp*']


def info_oriented_report(build_path, info_path) -> None:
    ignored_patterns = get_ignored_pattern()
    cmd = f"lcov --capture --directory {build_path} --output-file {info_path} "
    for pattern in ignored_patterns:
        cmd = cmd + f" --exclude '{pattern}' "
    print(f"info_oriented_report: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def handle_lcov(build_path, info_path, json_path, framework, split_name):
    info_save_module = Infosave(framework=framework, split_name=split_name)
    info_oriented_report(build_path, info_path)
    print("read_info_file and export")
    info_save_module.read_info_file(info_path)
    return info_save_module.export()

def pytorch_c_block_spectrum(json_info, file_dir, cov_file, gcda_list):
    gcov_file_dir = os.path.join(file_dir, "gcov_intermedia")
    # with open(cov_file, 'r', encoding='utf8') as fp:
    #     json_data = json.load(fp)
    json_data = json_info
    files = json_data['files'].keys()
    file_list = list(files)
    for gcdafile in gcda_list:

        if os.path.exists(gcov_file_dir):
            shutil.rmtree(gcov_file_dir)
        os.mkdir(gcov_file_dir)

        file_name = os.path.basename(gcdafile).replace('.gcda', '')

        python_command = f'cd {gcov_file_dir} && gcov -a ' + gcdafile + " > branfile"
        process = subprocess.Popen(python_command, shell=True)
        process.wait()

        if 'torch_cuda_generated_' in file_name:
            file_name = file_name.replace('torch_cuda_generated_', '')

        gcov_file_name = os.path.join(gcov_file_dir, f"{file_name}.gcov")

        if not os.path.exists(gcov_file_name):
            continue

        with open(gcov_file_name) as f:
            line = f.readline()
            source_file = "/pytorch" + line.replace('\n', '').split('Source:')[-1].strip().split("pytorch")[-1].strip()
            if source_file not in files:
                continue
            if source_file in file_list:
                file_list.remove(source_file)
            json_data['files'][source_file]["missing_blocks"] = []
            json_data['files'][source_file]["executed_blocks_frequency"] = {}

            block_dict = {}
            # %%%%%:   24-block   1
            for line in f:
                line = line.replace('\n', '')
                if "-block" in line:
                    lines = line.split(":")
                    block_key = lines[1].strip().split(' ')[0]
                    if block_key in block_dict.keys():
                        if block_dict[block_key] == 1:
                            continue
                        else:
                            if lines[0].strip() == "%%%%%" or lines[0].strip() == "$$$$$" or lines[0].strip() == "0":
                                block_dict[block_key] = 0
                            else:
                                block_dict[block_key] = 1
                    else:
                        if lines[0].strip() == "%%%%%" or lines[0].strip() == "$$$$$" or lines[0].strip() == "0":
                            block_dict[block_key] = 0
                        else:
                            block_dict[block_key] = 1
            pre_block = 0
            for block_key in block_dict.keys():
                line_num = int(block_key.split("-")[0])
                block_name = str(pre_block) + '-' + str(line_num)
                pre_block = line_num + 1
                if block_name in json_data['files'][source_file]["executed_blocks_frequency"].keys():
                    continue
                if block_dict[block_key] == 1:
                    json_data['files'][source_file]["executed_blocks_frequency"][block_name] = 1
                else:
                    json_data['files'][source_file]["missing_blocks"].append(block_name)


    file_name = cov_file
    for file in file_list:
        json_data['files'][file]["missing_blocks"] = []
        json_data['files'][file]["executed_blocks_frequency"] = {}
    json_data['totals'] = {}
    json_data['totals']['all_frequency'] = 1
    return json_data
    # with open(file_name, 'w', encoding='utf8') as f2:
    #     json.dump(json_data, f2, ensure_ascii=False, indent=2)


def tensorflow_c_block_spectrum(json_info, file_dir, cov_file, gcda_list):
    gcov_file_dir = os.path.join(file_dir, "gcov_intermedia")
    # with open(cov_file, 'r', encoding='utf8') as fp:
    #     json_data = json.load(fp)
    json_data = json_info
    files = json_data['files'].keys()
    file_list = list(files)
    for gcdafile in gcda_list:

        if os.path.exists(gcov_file_dir):
            shutil.rmtree(gcov_file_dir)
        os.mkdir(gcov_file_dir)

        file_name = os.path.basename(gcdafile).replace('.pic.gcda', '')

        python_command = f'cd {gcov_file_dir} && gcov -a ' + gcdafile + " > branfile"
        process = subprocess.Popen(python_command, shell=True)
        process.wait()

        gcov_list = find_files_with_suffix(gcov_file_dir, "gcov")

        gcov_files = []
        for gcov_line in gcov_list:
            gcov_file = os.path.basename(gcov_line)
            if file_name == gcov_file.split(".")[0]:
                gcov_files.append(gcov_line)
            else:
                continue

        for gcov_file in gcov_files:
            with open(gcov_file) as f:
                line = f.readline()
                source_file = "/" + line.replace('\n', '').split('Source:')[-1].strip()
                if source_file not in files:
                    continue
                if source_file in file_list:
                    file_list.remove(source_file)
                json_data['files'][source_file]["missing_blocks"] = []
                json_data['files'][source_file]["executed_blocks_frequency"] = {}

                block_dict = {}
                # %%%%%:   24-block   1
                for line in f:
                    line = line.replace('\n', '')
                    if "-block" in line:
                        lines = line.split(":")
                        block_key = lines[1].strip().split(' ')[0]
                        if block_key in block_dict.keys():
                            if block_dict[block_key] == 1:
                                continue
                            else:
                                if lines[0].strip() == "%%%%%" or lines[0].strip() == "$$$$$" or lines[
                                    0].strip() == "0":
                                    block_dict[block_key] = 0
                                else:
                                    block_dict[block_key] = 1
                        else:
                            if lines[0].strip() == "%%%%%" or lines[0].strip() == "$$$$$" or lines[0].strip() == "0":
                                block_dict[block_key] = 0
                            else:
                                block_dict[block_key] = 1
                pre_block = 0
                for block_key in block_dict.keys():
                    line_num = int(block_key.split("-")[0])
                    block_name = str(pre_block) + '-' + str(line_num)
                    pre_block = line_num + 1
                    if block_name in json_data['files'][source_file]["executed_blocks_frequency"].keys():
                        continue
                    if block_dict[block_key] == 1:
                        json_data['files'][source_file]["executed_blocks_frequency"][block_name] = 1
                    else:
                        json_data['files'][source_file]["missing_blocks"].append(block_name)

    file_name = cov_file
    for file in file_list:
        json_data['files'][file]["missing_blocks"] = []
        json_data['files'][file]["executed_blocks_frequency"] = {}
    json_data['totals'] = {}
    json_data['totals']['all_frequency'] = 1
    # with open(file_name, 'w', encoding='utf8') as f2:
    #     json.dump(json_data, f2, ensure_ascii=False, indent=2)
    return json_data


def find_files_with_suffix(directory_path, suffix):
    file_paths = []
    exclude_patterns = get_ignored_pattern()
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(suffix) and not any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                file_paths.append(os.path.join(root, file))

    return file_paths


def get_function_lines(function_node):
    function_lines = set(range(function_node.lineno, function_node.end_lineno + 1))

    for node in function_node.body:
        if isinstance(node, ast.FunctionDef):
            function_lines.difference_update(set(range(node.lineno, node.end_lineno + 1)))

    return sorted(list(function_lines))


def get_all_function_lines(node):
    function_lines_dict = {}
    if hasattr(node, "body"):
        for child_node in node.body:
            if isinstance(child_node, ast.FunctionDef):
                if hasattr(node, "name"):
                    function_lines_dict[node.name + "::" + child_node.name] = get_function_lines(child_node)
                else:
                    function_lines_dict[child_node.name] = get_function_lines(child_node)
            if isinstance(child_node, ast.AST):
                function_lines_dict.update(get_all_function_lines(child_node))
    return function_lines_dict


def python_file_spectrum(data, json_data):
    enum_data_origin = {"executed_lines_frequency": {},
                        "all_lines": [],
                        "missing_lines": []}
    site_package_prefix = None
    for file in json_data['files'].keys():
        # print(file)
        if site_package_prefix is None:
            site_package_prefix = file.split("site-packages")[0] + "site-packages"
        data_file = file.split("site-packages")[1]
        enum_data = deepcopy(enum_data_origin)
        for line in json_data['files'][file]['executed_lines']:
            strLine = str(line)
            enum_data["executed_lines_frequency"][strLine] = 1
            enum_data["all_lines"].append(strLine)
        for line in json_data['files'][file]['missing_lines']:
            strLine = str(line)
            enum_data["missing_lines"].append(strLine)
            enum_data["all_lines"].append(strLine)
        data["files"][data_file] = enum_data
    data['totals'] = {}
    data['totals']['all_frequency'] = 1
    return site_package_prefix, data


def python_function_and_block_spectrum(data, prefix,):
    for file in data["files"].keys():

        data["files"][file]["executed_functions_frequency"] = {}
        data["files"][file]["missing_functions"] = []
        file_path = prefix + file
        with open(file_path, encoding='utf-8') as f:
            try:
                module = ast.parse(f.read())
            except SyntaxError as e:
                continue
            function_lines = get_all_function_lines(module)
            for function_name, lines in function_lines.items():
                function_name_key = str(lines[0]) + "-" + str(lines[-1])
                frequency = 0
                for line in lines[1:]:
                    strLine = str(line)
                    if strLine in data["files"][file]["executed_lines_frequency"].keys():
                        frequency = max(data["files"][file]["executed_lines_frequency"].get(strLine), frequency)
                if frequency != 0:
                    data["files"][file]["executed_functions_frequency"][function_name_key] = frequency
                else:
                    data["files"][file]["missing_functions"].append(function_name_key)

        data["files"][file]["executed_blocks_frequency"] = {}
        data["files"][file]["missing_blocks"] = []

        try:
            cfg = CFGBuilder().build_from_file(file, prefix + file)
        except Exception as e:
            print(f"{prefix + file} CFGBuilder Error find : {e}")
            continue

        for class_name in cfg.class_cfgs.keys():
            for block in cfg.class_cfgs[class_name]:
                if block.statements == []:
                    continue
                block_key = str(block.statements[0].lineno) + "-" + str(block.statements[-1].end_lineno)
                frequency = 0
                for statement in block.statements:
                    if hasattr(statement, "lineno") and str(statement.lineno) in data["files"][file][
                        "executed_lines_frequency"].keys():
                        frequency = max(data["files"][file]["executed_lines_frequency"].get(str(statement.lineno)),
                                        frequency)
                if frequency != 0:
                    data["files"][file]["executed_blocks_frequency"][block_key] = frequency
                else:
                    data["files"][file]["missing_blocks"].append(block_key)

        for block in cfg:
            if block.statements == []:
                continue
            block_key = str(block.statements[0].lineno) + "-" + str(block.statements[-1].end_lineno)
            frequency = 0
            for statement in block.statements:
                if hasattr(statement, "lineno") and str(statement.lineno) in data["files"][file][
                    "executed_lines_frequency"].keys():
                    frequency = max(data["files"][file]["executed_lines_frequency"].get(str(statement.lineno)),
                                    frequency)
            if frequency != 0:
                data["files"][file]["executed_blocks_frequency"][block_key] = frequency
            else:
                data["files"][file]["missing_blocks"].append(block_key)

    # with open(json_name, 'w', encoding='utf8') as f2:
    #     json.dump(data, f2, ensure_ascii=False, indent=2)
    return data
