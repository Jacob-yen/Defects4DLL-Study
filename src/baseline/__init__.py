import astor

from src.tools.utils import AnalysisUtils, read_text
from src.localization import get_all_function_lines
from xml.etree import ElementTree as ET
import ast
import os
import subprocess
from src import root_path
import random
from src.tools.logger_utils import LoggerUtils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import jaccard_distance

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        if node.name.startswith("test_"):
            function_info = {
                "name": node.name,
                "text": astor.to_source(node),
                "start_lineno": node.lineno,
                "end_lineno": node.end_lineno
            }
            self.functions.append(function_info)
        self.generic_visit(node)


def init_test_file_func(framework, fail_test_path, test_files):
    test_file_func_similarity = {}
    fail_code = read_text(fail_test_path)
    # print(f"fail_code: {fail_code}")
    # api相似度
    fail_api_all = extract_lib_api_v2(framework, fail_code)
    fail_api_string = ' '.join(fail_api_all)
    print(f"fail_api_string: {fail_api_string}")

    for test_file in test_files:
        # print(test_file)
        source_code = read_text(test_file)
        try:
            tree = ast.parse(source=source_code, mode='exec')
        except SyntaxError as e:
            # 处理语法错误，例如输出错误信息
            print(f"SyntaxError: {e}")
            continue
        # except Exception as e:
        #     # 处理其他异常
        #     print(f"An unexpected error occurred: {e}")
        test_file_api_dict = extract_lib_api_with_test_file(framework, test_file)
        print(f"test_file: {test_file}")
        print(f"test_file_api_dict: {test_file_api_dict}")
        if len(test_file_api_dict) == 0:
            continue

        function_extractor = FunctionExtractor()
        function_extractor.visit(tree)

        for function_info in function_extractor.functions:
            print(f"function_info: {function_info}")
            # api相似度
            function_api_all = set()
            # TODO 遍历test_file_api_dict，抽取所有api在当前函数中的，构造function_api_all
            for api in test_file_api_dict.keys():
                lines_list = test_file_api_dict[api]
                for lines in lines_list:
                    if function_info["start_lineno"] <= lines[0] and lines[1] <= function_info["end_lineno"]:
                        print(f"api: {api}")
                        function_api_all.add(api)
                        break
            # function_api_all = extract_lib_api_v2(framework, function_info["text"])
            function_api_string = ' '.join(list(function_api_all))
            print(f"function_api_string: {function_api_string}")
            cosine_smi = calculate_similarity(fail_api_string, function_api_string, 'cosine')

            # 测试用例相似度
            # cosine_smi = calculate_similarity(fail_code, function_info["text"], 'cosine')
            # print(f"function_info: {function_info['text']}")
            test_file_func_key = f"{test_file}${function_info['name']}${function_info['start_lineno']}${function_info['end_lineno']}"
            # if cosine_smi > 0.3:
            test_file_func_similarity[test_file_func_key] = cosine_smi
    return sorted(test_file_func_similarity.items(), key=lambda x: x[1], reverse=True)


def calculate_similarity(text1, text2, method='cosine'):
    if method == 'cosine':
        tfidf_vectorized = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorized.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return cosine_sim[0][0]
    elif method == 'jaccard':
        set1 = set(text1.split())
        set2 = set(text2.split())
        jaccard_sim = 1 - jaccard_distance(set1, set2)
        return jaccard_sim
    elif method == 'levenshtein':
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        max_len = max(len(text1), len(text2))
        levenshtein_sim = 1 - levenshtein_distance(text1, text2) / max_len
        return levenshtein_sim
    else:
        raise ValueError("Unsupported similarity method")


# 获得pytest运行出来的xml中的所有函数路径
def collect_functions(source_compile_dir, test_file, function_xml_path, defect_api, pass_file_path, framework,
                      framework_prefix, target_bug, root_result_dir):
    with open(test_file, encoding='utf-8') as f:
        module = ast.parse(f.read())
        function_lines = get_all_function_lines(module)
    line_num = 0
    fun_list = []
    with open(test_file, encoding='utf-8') as f:
        for line in f:
            if defect_api.lower() in line.lower():
                for key, value in function_lines.items():
                    if line_num in value:
                        fun_list.append(key)
            line_num = line_num + 1
    source_path = ""
    tree = ET.parse(function_xml_path)
    root = tree.getroot()
    fun_paths = {}
    for child in root[0]:
        if child.findall('*') != []:
            continue
        for fun in fun_list:
            class_name = ""
            if "::" in fun:
                class_name = fun.split("::")[0]
                fun = fun.split("::")[1]
            if (class_name == "" or class_name in child.attrib.get('classname')) and child.attrib.get(
                    'name').startswith(fun):
                function = ""
                paths = child.attrib.get('classname').split(".")

                for path in paths[0:-2]:
                    function = function + path + "/"
                function_key = function + paths[-2] + ".py::" + paths[-1] + "::" + fun
                function_value = function + paths[-2] + ".py::" + paths[-1] + "::" + child.attrib.get('name')
                if function_key in fun_paths.keys():
                    fun_paths[function_key].append(source_path + function_value)
                else:
                    fun_paths[function_key] = [source_path + function_value]
    pass_test_paths = set()
    for fun_path_key, fun_path_value in fun_paths.items():
        pass_test_paths.update(fun_path_value)
    pass_test_paths = list(pass_test_paths)
    if framework == "pytorch":
        root_path = os.path.join(source_compile_dir, f"{framework_prefix}-project/{target_bug}/{framework}")
    else:
        root_path = root_result_dir
    pass_test_paths = [os.path.join(root_path, item) for item in pass_test_paths]
    with open(pass_file_path, "w") as file:
        for item in pass_test_paths:
            file.write(f"{item}\n")

    return pass_test_paths


def extract_lib_api(framework, target_bug):
    def _extract_lib_api_from_file(source_code):
        if framework == "tensorflow":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["tensorflow"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        elif framework == "pytorch":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["torch"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        elif framework == "paddlepaddle":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["paddle"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        elif framework == "jittor":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["jittor"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        elif framework == "mindspore":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mindspore"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        elif framework == "mxnet":
            lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mxnet"])
            api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
        else:
            raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")
        # logger.info(api_calls)
        api_all = set()
        for api_call in api_calls:
            full_name = api_call["full_name"]
            if framework == "tensorflow":
                full_name = full_name.replace("tensorflow", "tf")
            api_all.add(full_name)
        return api_all

    total_apis = set()
    original_code = read_text(os.path.join(root_path, f"data/{framework}/Result/{target_bug}/{target_bug}-original.py"))
    total_apis.update(_extract_lib_api_from_file(original_code))
    # util_path = os.path.join(root_path, f"data/{framework}/Result/{target_bug}/util.py")
    # if os.path.exists(util_path):
    #     util_code = read_text(util_path)
    #     total_apis.update(_extract_lib_api_from_file(util_code))
    logger.debug(total_apis)
    return total_apis


def extract_lib_api_v2(framework, source_code):
    # source_code = read_text(os.path.join(root_path, f"data/{framework}/Result/{target_bug}/{target_bug}-original.py"))
    if framework == "tensorflow":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["tensorflow"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    elif framework == "pytorch":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["torch"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    elif framework == "paddlepaddle":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["paddle"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    elif framework == "jittor":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["jittor"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    elif framework == "mindspore":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mindspore"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    elif framework == "mxnet":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mxnet"])
        api_calls = AnalysisUtils.extract_api_calls_v2(source_code, targets=lib_alias_pairs)
    else:
        raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")
    # logger.info(api_calls)
    api_all = set()
    for api_call in api_calls:
        full_name = api_call["full_name"]
        if framework == "tensorflow":
            full_name = full_name.replace("tensorflow", "tf")
        api_all.add(full_name)
    return api_all


def extract_lib_api_with_test_file(framework, test_file):
    source_code = read_text(test_file)
    if framework == "tensorflow":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["tensorflow"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    elif framework == "pytorch":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["torch"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    elif framework == "paddlepaddle":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["paddle"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    elif framework == "jittor":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["jittor"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    elif framework == "mindspore":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mindspore"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    elif framework == "mxnet":
        lib_alias_pairs = AnalysisUtils.extract_lib_alias(source_code, targets=["mxnet"])
        api_calls = AnalysisUtils.extract_api_calls_v3(source_code, targets=lib_alias_pairs)
    else:
        raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")
    # logger.info(api_calls)
    api_all = dict()
    for api_call in api_calls:
        full_name = api_call["full_name"]
        if framework == "tensorflow":
            full_name = full_name.replace("tensorflow", "tf")
        if full_name not in api_all.keys():
            api_all[full_name] = [[api_call["lineno"], api_call["end_lineno"]]]
        else:
            api_all[full_name].append([api_call["lineno"], api_call["end_lineno"]])
    return api_all


def find_class_and_function(code_file, line_number, end_line_number):
    with open(code_file, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    # Find the class containing the line number
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.lineno <= line_number and end_line_number <= node.end_lineno:
            class_node = node
            break

    # Find the function containing the line number within the found class
    function_node = None
    if class_node is not None:
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef) and node.lineno <= line_number and end_line_number <= node.end_lineno:
                function_node = node
                break

    return class_node.name if class_node else None, function_node.name if function_node else None


def get_pytest_path(test_file, class_name, function_name, python_interpreter):
    # 分析树形结构，过滤class_num，function_name, 得到对应的路径列表,然后随机sample一个
    "pytest --collect-only test_file"
    pass_file_path = ""
    pass_file_paths = []
    pytest_paths = []

    command = f"{python_interpreter} -m pytest --collect-only {test_file}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = process.communicate(timeout=5 * 60)
        if process.returncode == 0:
            line = stdout.decode().strip()
            pytest_paths = line.splitlines()
        else:
            process.terminate()
            logger.error(f"Command '{command}' returned non-zero exit status {process.returncode}:")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"Command {command} timed out after 5 minutes")
        process.terminate()
        kill_command = f"pkill -f '{command}'"
        logger.info(f"{kill_command} is running!")
        process = subprocess.Popen(kill_command, shell=True)
        process.wait()
        return None
    except Exception as e:
        process.terminate()
        logger.error(f"Exception raised when executing {command}")
        logger.error(str(e))
        return None


    class_pytest_name = ""
    function_pytest_name = ""
    for pytest_path in pytest_paths:
        if "UnitTestCase" in pytest_path and class_name in pytest_path:
            class_pytest_name = pytest_path.strip().split(" ")[-1][:-1]
        if "TestCaseFunction" in pytest_path and function_name in pytest_path:
            function_pytest_name = pytest_path.strip().split(" ")[-1][:-1]
            if class_pytest_name != "":
                pass_file_paths.append(f"{test_file}::{class_pytest_name}::{function_pytest_name}")

    if len(pass_file_paths) != 0:
        random.shuffle(pass_file_paths)
        pass_file_path = random.choice(pass_file_paths)

    return pass_file_path
