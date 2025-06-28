import json
import os
import ast
import random
import time
import subprocess
import shutil
import numpy as np
from src.sampler.mcmc_sampler import MCMCSampler as Sampler
from src import DOCTER_PATH, FUNC_SIG_PATH, root_path, regression_version, framework_nicknames
from src.schedule import AbstractScheduler
from src.tools.utils import LLMBot
from src.mutation.instrument import CodeInstrumenter, parse_output, remove_oracle
from src.baseline import extract_lib_api_with_test_file, find_class_and_function, get_pytest_path
from src.mutation.operators import Mutator, construct_mutator
from src.mutation.llm import system_prompt_template, user_prompt_template, feedback_template
from src.tools.enum_class import Symptom, Approach, Framework, TestType
from src.tools import utils
from src.tools.logger_utils import LoggerUtils
from collections import defaultdict

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class Mutant:
    def __init__(self, code) -> None:
        self.code = code
        self.history = []

    @property
    def order(self):
        return len(self.history)


def text_last_line(text):
    text_splits = [l for l in text.split("\n") if l != ""]
    if len(text_splits) == 0:
        return ""
    else:
        return text_splits[-1]


def calculate_text_similarity(text1, text2):
    # calculate the similarity between two text with bleu
    # text1 is the original text
    # text2 is the mutated text
    # return the bleu score
    from nltk.translate.bleu_score import sentence_bleu
    text1 = text1.replace("\n", " ").split(" ")
    text2 = text2.replace("\n", " ").split(" ")
    # remove str like "0x7f21b1fdddf0"
    text1 = [l for l in text1 if l != "" and "0x" not in l]
    text2 = [l for l in text2 if l != "" and "0x" not in l]
    # text1 = [l for l in text1 if l != ""]
    # text2 = [l for l in text2 if l != ""]
    if len(text1) == 0 or len(text2) == 0:
        return 0
    else:
        return sentence_bleu([text1], text2, weights=(1, 0, 0, 0))


def print_mutator_scores(mutators):
    for mutator in mutators:
        logger.debug(f"{mutator.mutator_name}: {mutator.print_score()}")


def get_jaccard_distance(new_pass_sp, history_pass_pool: dict):
    jaccard_distance = []
    for history_pass_spectrum_path in history_pass_pool.values():
        similarity = utils.coverage_jaccard_similarity(history_pass_spectrum_path, new_pass_sp)
        jaccard_distance.append(1 - similarity)

    distance = np.mean(jaccard_distance) if len(jaccard_distance) > 0 else 0.0
    return distance

def cut_stack_trace(stack_trace):
    stack_trace_lines = [l for l in stack_trace.split("\n") if l != ""]
    stack_trace_last_line = stack_trace_lines[-1]
    if len(stack_trace_lines) > 30:
        stack_trace_lines = stack_trace_lines[:15] + ["# skip middle lines"] + stack_trace_lines[-15:]
    return stack_trace_last_line, "\n".join(stack_trace_lines)

class HybridScheduler(AbstractScheduler):

    def __init__(self, root_result_dir, bug_save_dir, method_save_name, framework, fail_bug, max_test_case_count,
                 time_limit_seconds,
                 config, cov_collector, interpreter_path, source_compile_path, symptom, max_mutation_order,
                 mutator_selection, hybrid_techniques, api_url,api_key, model_name, temperature, fail_spectrum, mutation_levels) -> None:

        super().__init__(root_result_dir=root_result_dir, bug_save_dir=bug_save_dir, method_save_name=method_save_name,
                         framework=framework, fail_bug=fail_bug, max_test_case_count=max_test_case_count,
                         time_limit_seconds=time_limit_seconds, config=config, cov_collector=cov_collector,
                         interpreter_path=interpreter_path, source_compile_path=source_compile_path)

        # common configuration
        self.framework_prefix = framework_nicknames[framework]
        self.framework = framework
        assert self.framework in Framework.TOTAL, f"Unsupported framework: {self.framework}"
        self.time_limit_seconds = time_limit_seconds
        self.max_test_case_count = max_test_case_count
        self.max_mutation_order = max_mutation_order
        self.hybrid_techniques = hybrid_techniques
        self.source_compile_path = source_compile_path

        docter_cons_names = [f[:-5] for f in os.listdir(DOCTER_PATH[framework])]
        supported_func_sig = [f[:-5] for f in os.listdir(FUNC_SIG_PATH[framework])]
        self.supported_apis = [api for api in supported_func_sig if api.lower() in docter_cons_names]
        self.target_bug = f"{self.framework_prefix}-{fail_bug}"
        self.orig_code = utils.read_text(
            os.path.join(root_path, f"./data/{framework}/Result/{self.target_bug}/{self.target_bug}-original.py"))
        self.stack_trace = utils.read_text(
            os.path.join(root_path, f"./data/{framework}/Result/{self.target_bug}/stack_trace.txt"))
        # if the stack_trace lines more than 30, we use the first 15 lines and the last 15 lines
        self.stack_trace_last_line, self.stack_trace = cut_stack_trace(self.stack_trace)
        self.bug_run_dir = os.path.join(self.bug_save_dir, self.target_bug)

        self.tmp_exec = os.path.join(self.root_result_dir, "coverage_intermedia",
                                     f"tmp-exec-{framework}-{self.method_save_name}")

        if not os.path.exists(self.tmp_exec):
            os.mkdir(self.tmp_exec)

        self.tmp_cov_path = os.path.join(self.tmp_exec, f"{self.fail_bug}")
        if not os.path.exists(self.tmp_cov_path):
            os.mkdir(self.tmp_cov_path)

        self.symptom = symptom
        if self.symptom == Symptom.ASSERT:
            self.bug_run_infor_dir = os.path.join(self.bug_run_dir, "infor")
            os.makedirs(self.bug_run_infor_dir, exist_ok=True)
        self.spectrum_set = set()
        self.mutation_history = defaultdict(lambda: defaultdict(int))
        self.all_history = defaultdict(int)
        self.total_generate_duration = 0  # 累计生成时间
        self.total_execute_duration = 0  # 累计执行时间
        self.md5_time = 0
        self.exe_time = 0
        self.cov_time = 0
        self.ana_time = 0
        self.else_time = 0
        self.mutators = []
        self.one_order_pool = {}
        self.high_order_pool = {}
        self.pass_pool = {}
        self.fail_pool = {}

        # Configuration for our mutation approach
        if Approach.RULE in hybrid_techniques and self.symptom != Symptom.ASSERT:
            self.mutators = construct_mutator(framework=framework, mutator_levels=mutation_levels)
            # self.api_replace_mutators = [m for m in self.mutators if m.sub_mutator == "api_replace"]

        # Configuration for LLM technique
        if Approach.LLM in hybrid_techniques:

            system_prompt = system_prompt_template.format(self.framework)
            self.user_prompt = user_prompt_template.format(self.framework, self.orig_code, self.stack_trace)
            # replace the prompt pattern
            # define a chatbot
            self.chat_bot = LLMBot(api_base=api_url,api_key=api_key, model=model_name, system_prompt=system_prompt, temperature=temperature)
            self.mutators.append(Mutator(framework=framework, mutator_level=Approach.LLM))

        # Configuration for Developer technique
        if Approach.DEVELOPER in hybrid_techniques:
            self.python_interpreter = os.path.join(self.interpreter_path, "envs", f"{self.target_bug}-buggy/bin/python")
            self.mutators.append(Mutator(framework=framework, mutator_level=Approach.DEVELOPER))
            # get all test cases provided by developer
            self.test_files = []
            if framework == Framework.PADDLE:
                # test_files_path = os.path.join(self.source_compile_path, f"{self.framework_prefix}-project",
                #                                self.target_bug,
                #                                "Paddle")
                test_files_path = "/data/review/Paddle"
            else:
                test_files_path = os.path.join(self.source_compile_path, f"{self.framework_prefix}-project",
                                           self.target_bug,
                                           self.framework)

            if framework == Framework.PYTORCH or framework == Framework.PADDLE:
                process = subprocess.Popen(f"find {test_files_path} -name 'test_*.py' | grep -v 'third_party'",
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
            elif framework == Framework.TENSORFLOW:
                process = subprocess.Popen(f"find {test_files_path} -name '*_test.py'", shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
            else:
                raise NotImplementedError(f"Wrapper for {framework} not implemented yet.")
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                line = stdout.decode().strip()
                self.test_files = line.splitlines()

            self.api_test_files_dict = {}
        self.sampler = Sampler(mutators=self.mutators,mode=mutator_selection)
        self.last_mutator = None
        self.monitors = self.monitor_types = self.input_params = self.assert_type = self.target_assertion = None

        # get the spectrum vector of fail test case
        self.fail_spectrum_path, self.fail_cover_info = utils.parse_coverage(fail_spectrum)

        self.pass_file_path_txt = os.path.join(self.bug_save_dir, self.target_bug, "history_pass_file_list.txt")
        with open(self.pass_file_path_txt, "w") as file:
            file.write(f"{time.time()},start\n")

    def record_pass_file(self, file_path):
        with open(self.pass_file_path_txt, "a+") as file:
            file.write(f"{time.time()},{file_path}\n")

    def perform_mutation(self, mutator, mutant, start_time):
        if mutator.mutator_name == Approach.LLM:
            self.llm_mutation_execution(mutator)
        elif mutator.mutator_name == Approach.DEVELOPER:
            self.developer_selection_execution(mutator, start_time)
        else:
            self.our_mutation(mutator, mutant)

    def mutation_loop(self):
        start_time = time.time()
        # # we temporarily do not support the assertion bug for rule
        initial_code = self.orig_code

        # add the original code into one_order_pool
        original_seed = Mutant(initial_code)
        logger.info("Start one-order mutation")
        self.one_order_pool[utils.gen_md5_id(initial_code)] = original_seed

        while True:
            print_mutator_scores(self.mutators)
            # print the pool size
            logger.info(
                f"Pass Pool {len(self.pass_pool)}, One-order Pool {len(self.one_order_pool)}, High-order Pool {len(self.high_order_pool)}")
            logger.info("Start High-order mutation")
            if time.time() - start_time >= self.time_limit_seconds:
                logger.info("Time limit reached. Mutation process terminated.")
                break

            selected_mutator = self.sampler.selected_mutator(self.last_mutator)
            # choose one mutator
            logger.info("High-order: selected_mutator: " + selected_mutator.mutator_name)
            pool = self.high_order_pool if random.random() < 0.6 and len(self.high_order_pool) > 0 else self.one_order_pool
            _, selected_mutant = random.choice(list(pool.items()))
            self.perform_mutation(mutator=selected_mutator, mutant=selected_mutant, start_time=start_time)
            self.last_mutator = selected_mutator.mutator_name

            logger.info(f"{len(self.pass_pool)} mutants generated.")
            left_seconds = self.time_limit_seconds - (time.time() - start_time)
            left_seconds = int(left_seconds)
            logger.info(f"Normal Mutation Time left: {left_seconds} seconds")

    def generate_oracle(self, new_code, new_code_md5):
        if self.symptom == Symptom.ASSERT:
            file_dir = os.path.join(self.bug_run_infor_dir, new_code_md5)

            monitor_code = CodeInstrumenter.variable_monitor_gain_code(self.input_params, self.target_assertion,
                                                                       self.monitors, self.monitor_types,
                                                                       self.assert_type, self.framework,
                                                                       file_dir)

            logger.debug("==========Monitor Code==========")
            logger.debug(monitor_code)

            code_with_monitor = CodeInstrumenter.instrument_code(new_code, monitor_code)
            logger.debug("==========New Code==========")
            logger.debug(code_with_monitor)
            if self.fail_bug in regression_version[self.framework].keys():
                # run the reference library
                regress_ver = regression_version[self.framework][self.fail_bug][0]
                regress_ver = f"{framework_nicknames[self.framework]}-{regress_ver}"

                status, output = self.execute_code(code_with_monitor, regress_ver)

                if status:
                    stmts, is_success = CodeInstrumenter.variable_monitor_detection_code(self.monitors, output,
                                                                                         self.monitor_types,
                                                                                         self.input_params,
                                                                                         file_dir,
                                                                                         self.framework,
                                                                                         self.assert_type)
                    if is_success:
                        # generate oracle
                        logger.debug(stmts)
                        new_code = CodeInstrumenter.insert_oracle(new_code, stmts)
                        logger.debug("==========New Code with oracle==========")
                        logger.debug(new_code)
                        return new_code
                    else:
                        logger.error("The mutated code has runtime-error. Skip it")
                        return None
                else:
                    logger.error("The mutated code has runtime-error. Skip it")
                    return None
            else:
                raise NotImplemented("Cross-library oracle generation is not supported yet.")
        return new_code



    def our_mutation(self, mutator, mutant):
        mutator.update_selected()
        self.mutation_history[mutator.mutator_name]["usage"] += 1
        self.all_history["usage"] += 1
        try:
            start_generate = time.time()  
            all_mutation_points = mutator.mutation_point_scan(source_code=mutant.code,
                                                          traceback=self.stack_trace,
                                                          supported_apis=self.supported_apis,
                                                          order=mutant.order)
            self.total_generate_duration += time.time() - start_generate
            if len(all_mutation_points) == 0:
                return
            if self.framework == Framework.TENSORFLOW:
                # For TensorFlow, do not explore all mutations to avoid too many failed seeds
                mutation_points = [random.choice(all_mutation_points)]
            else:
                mutation_points = all_mutation_points
            for mutation_point in mutation_points:
                start_time = time.time()  
                new_codes = mutator.mutation(source_code=mutant.code, supported_apis=self.supported_apis,
                                            order=mutant.order, mutation_point=mutation_point)
                self.total_generate_duration += time.time() - start_time
                start_execute = time.time()  
                try:
                    if isinstance(new_codes, list):
                        # print(f"api replace new codes number is {len(new_codes)}")
                        for new_code in new_codes:
                            if new_code is not None:
                                self.mutation_history[mutator.mutator_name]["generate"] += 1
                                self.all_history["generate"] += 1
                                self.our_execution(new_code, mutator, mutant)
                    else:
                        new_code = new_codes
                        if new_code is not None:
                            self.mutation_history[mutator.mutator_name]["generate"] += 1
                            self.all_history["generate"] += 1
                            self.our_execution(new_code, mutator, mutant)
                except Exception as e:
                    self.total_execute_duration += time.time() - start_execute
                    self.mutation_history[mutator.mutator_name]["error"] += 1
                    self.all_history["error"] += 1
                    logger.error(f"Mutation error: {e}")
                    return
                self.total_execute_duration += time.time() - start_execute
        except Exception as e:
            self.mutation_history[mutator.mutator_name]["error"] += 1
            self.all_history["error"] += 1
            logger.error(f"Mutation error: {e}")
            return


    def llm_mutation_execution(self, mutator):
        self.mutation_history[mutator.mutator_name]["usage"] += 1
        self.all_history["usage"] += 1
        mutator.update_selected()
        start_generate = time.time()
        try:
            _s1 = time.time()
            raw_response = self.chat_bot.chat_completion(api_key="EMPTY", prompt=self.user_prompt)
            logger.info(f"LLM Invocation time cost: {time.time() - _s1} seconds")
            self.total_generate_duration += time.time() - start_generate
        except Exception as e:
            self.total_generate_duration += time.time() - start_generate
            self.mutation_history[mutator.mutator_name]["error"] += 1
            self.all_history["error"] += 1
            logger.error(f"Mutation error: {e}")
            return
        if raw_response is not None and (response := LLMBot.parse_response(raw_response)) is not None:
            start_execute = time.time()
            try:
                self.mutation_history[mutator.mutator_name]["generate"] += 1
                self.all_history["generate"] += 1
                mutator.update_success_mutation()
                md5 = time.time()
                md5_identity = utils.gen_md5_id(response)
                self.md5_time += time.time() - md5
                if md5_identity not in self.pass_pool:
                    exe = time.time()
                    pass_status, console_output = self.execute_code(response, f"{self.target_bug}-buggy")
                    self.exe_time += time.time() - exe
                    if pass_status:
                        self.chat_bot.clear_history()
                        mutator.update_success_pass()
                        self.mutation_history[mutator.mutator_name]["pass"] += 1
                        mutant_path = self.save_mutation_by_code(response, md5_identity)
                        cov = time.time()
                        pass_spectrum = self.cov_collector.process_single_file(mutant_path, is_developer_file=False,
                                                                               test_type=TestType.PASS)
                        self.cov_time += time.time() - cov
                        if pass_spectrum is not None:
                            ana = time.time()
                            self.record_pass_file(mutant_path)
                            new_pass_spectrum_str = utils.coverage_path(pass_spectrum, self.fail_cover_info)
                            # calculate the distance between the new test case and the existing test cases
                            jaccard_distance = get_jaccard_distance(new_pass_spectrum_str, self.pass_pool)
                            self.pass_pool[md5_identity] = new_pass_spectrum_str
                            mutator.score = jaccard_distance * mutator.rate
                            logger.debug(f"LLM Reward: ({round(jaccard_distance, 4)})*({round(mutator.rate, 4)}) = {round(mutator.score, 4)}")
                            self.ana_time += time.time() - ana
                    else:
                        els = time.time()
                        cropped_stack_trace = cut_stack_trace(console_output)[1]
                        self.chat_bot.update_history(raw_response,cropped_stack_trace)
                        self.else_time += time.time() - els
                else:
                    self.all_history["fail"] += 1
                self.total_execute_duration += time.time() - start_execute
            except Exception as e:
                self.total_execute_duration += time.time() - start_execute
                self.all_history["error"] += 1
                logger.error(f"Execution error: {e}")
                return

    def developer_selection_execution(self, mutator, start_time):
        self.mutation_history[mutator.mutator_name]["usage"] += 1
        self.all_history["usage"] += 1
        mutator.update_selected()
        start_generate = time.time()  
        pass_file_path = self.get_developer_file(start_time)
        logger.info(pass_file_path)
        self.total_generate_duration += time.time() - start_generate
        start_execute = time.time()  

        if pass_file_path is None:
            self.total_execute_duration += time.time() - start_execute
            return
        else:
            mutator.update_success_mutation()
            self.mutation_history[mutator.mutator_name]["generate"] += 1
            self.all_history["generate"] += 1
            command = f"{self.python_interpreter} -m pytest {pass_file_path}"
            process = subprocess.Popen(f"cd {self.tmp_cov_path} && {command}", shell=True)
            try:
                process.wait(timeout=5 * 60)
                if process.returncode == 0:
                    logger.info(f"SUCCESS: {self.target_bug} {pass_file_path} PASS")
                    mutator.update_success_pass()
                    self.all_history["pass"] += 1
                else:
                    process.terminate()
                    logger.error(f"Command '{command}' returned non-zero exit status {process.returncode}:")
                    self.all_history["fail"] += 1
                    self.total_execute_duration += time.time() - start_execute
                    return
            except subprocess.TimeoutExpired:
                logger.error("Command timed out after 5 minutes")
                process.terminate()
                kill_command = f"pkill -f '{command}'"
                logger.info(f"{kill_command} is running!")
                process = subprocess.Popen(kill_command, shell=True)
                process.wait()
                self.all_history["fail"] += 1
                self.total_execute_duration += time.time() - start_execute
                return
            except Exception as e:
                process.terminate()
                logger.error(f"Exception raised when executing {command}")
                logger.error(str(e))
                self.all_history["fail"] += 1
                self.total_execute_duration += time.time() - start_execute
                return
            self.mutation_history[mutator.mutator_name]["pass"] += 1
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

            pass_spectrum = self.cov_collector.process_single_file(pass_file_path, is_developer_file=True,
                                                                   test_type=TestType.PASS)
            if pass_spectrum is not None:
                self.record_pass_file(save_path)
                new_pass_spectrum_str = utils.coverage_path(pass_spectrum, self.fail_cover_info)
                # calculate the distance between the new test case and the existing test cases
                jaccard_distance = get_jaccard_distance(new_pass_spectrum_str, self.pass_pool)
                self.pass_pool[utils.gen_md5_id(pass_file_path)] = new_pass_spectrum_str
                # for developer test case, we use the md5 of the file path as the identity and do not save the code
                # self.pass_pool[utils.gen_md5_id(pass_file_path)] = None
                mutator.score = jaccard_distance * mutator.rate
                logger.debug(f"Developer Reward: ({jaccard_distance})*({mutator.rate}) = {round(mutator.score, 4)}")
            self.total_execute_duration += time.time() - start_execute


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
            logger.debug(pass_file_path)
            # if utils.gen_md5_id(pass_file_path) not in self.pass_pool.keys():
            #     # self.pass_pool[utils.gen_md5_id(pass_file_path)] = pass_file_path
            #     # print(f"{api}: {pass_file_path}")
            #     logger.info(f"api: pass_file_path - {api}: {pass_file_path}")
            #     return pass_file_path
            # else:
            #     continue
            return pass_file_path

    def our_execution(self, new_code, mutator, mutant):
        new_code_md5 = utils.gen_md5_id(new_code)
        if new_code_md5 in self.pass_pool.keys() or new_code_md5 in self.one_order_pool.keys() or new_code_md5 in self.high_order_pool.keys():
            self.all_history["fail"] += 1
            logger.info(f"Mutation already exists. Skip.")
            return

        # if the code has syntax error, we skip it
        if not utils.check_syntax_error(new_code):
            self.all_history["fail"] += 1
            logger.info(f"Syntax error. Skip.")
            return

        new_code = self.generate_oracle(new_code, new_code_md5)
        if new_code is None:
            return
        mutator.update_success_mutation()
        pass_status, console_output = self.execute_code(new_code, f"{self.target_bug}-buggy")
        remove_anchor_code = remove_oracle(new_code) if self.symptom == Symptom.ASSERT else new_code
        # if the mutant pass the test cases, we collect the spectrum
        if pass_status:
            self.handle_pass_mutant(new_code, remove_anchor_code, mutant, mutator)
        else:
            self.handle_fail_mutant(new_code, remove_anchor_code, mutant, mutator, console_output)

    def execute_code(self, code, interpreter):
        code_md5 = utils.gen_md5_id(code)
        # save the code to result_path
        tmp_path = os.path.join(self.bug_save_dir, self.target_bug, code_md5 + ".py")

        parent_dir = os.path.dirname(tmp_path)
        with open(tmp_path, 'w', encoding='utf8') as fp:
            fp.write(code)
        try:
            logger.debug(self.interpreter_path)
            logger.debug(interpreter)
            python_interpreter = os.path.join(self.interpreter_path, "envs", interpreter, "bin/python")
            logger.debug(python_interpreter)
            command = f"cd {self.tmp_cov_path} && export PYTHONPATH={parent_dir}:$PYTHONPATH && timeout 120 {python_interpreter} -u {tmp_path}"
            logger.debug(command)
            result = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    check=True)
            output = result.stdout
            logger.debug(output)
            logger.info(f"SUCCESS: {self.target_bug} PASS")
            self.all_history["pass"] += 1
            # delete the temp file
            os.remove(tmp_path)
            return True, output
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}:")
            logger.debug(e.stderr)
            logger.debug(f"{'=' * 10}Failed Mutant{'=' * 10}")
            logger.debug(code)
            self.all_history["fail"] += 1
            # delete the temp file
            os.remove(tmp_path)
            return False, str(e.stderr)
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after 2 minutes")
            self.all_history["fail"] += 1
            return False, None

    def handle_pass_mutant(self, new_code, remove_anchor_code, mutant, mutator):
        self.mutation_history[mutator.mutator_name]["pass"] += 1
        mutator.update_success_pass()
        # generate pass mutant
        logger.debug("==========Insert Code into Pass Pool==========")
        pass_mutant = Mutant(new_code)
        pass_mutant.history = mutant.history.copy()
        pass_mutant.history.append(mutator.mutator_name)
        # the md5 of the pass_mutant is the identity of the pass code
        md5_identity = utils.gen_md5_id(new_code)
        # self.pass_pool[md5_identity] = pass_mutant
        mutant_path = self.save_mutation(pass_mutant, md5_identity)

        pass_spectrum = self.cov_collector.process_single_file(mutant_path, is_developer_file=False,
                                                               test_type=TestType.PASS)
        if pass_spectrum is not None:
            # if mutant.order < self.max_mutation_order - 1:
            #     logger.debug("==========Insert Code into High Order Candidate Pool==========")
            #     logger.debug(remove_anchor_code)
            #     high_mutant = Mutant(remove_anchor_code)
            #     high_mutant.history = mutant.history.copy()
            #     high_mutant.history.append(mutator.mutator_name)
            #     self.high_order_pool[utils.gen_md5_id(remove_anchor_code)] = high_mutant
            # else:
            #     logger.debug(
            #         f"Mutant order ({mutant.order}) is larger or equal to max_mutation_order ({self.max_mutation_order}). ")
            self.record_pass_file(mutant_path)
            new_pass_spectrum_str = utils.coverage_path(pass_spectrum, self.fail_cover_info)
            # calculate the distance between the new test case and the existing test cases
            jaccard_distance = get_jaccard_distance(new_pass_spectrum_str, self.pass_pool)
            self.pass_pool[md5_identity] = new_pass_spectrum_str
            mutator.score = jaccard_distance * mutator.rate
            logger.debug(f"{mutator.mutator_name} Reward: ({round(jaccard_distance,4)})*({round(mutator.rate, 4)}) = {round(mutator.score, 4)}")

    def handle_fail_mutant(self, new_code, remove_anchor_code, mutant, mutator, console_output):
        if self.symptom == Symptom.ASSERT:
            files = os.listdir(self.bug_run_infor_dir)
            for item in files:
                item_path = os.path.join(self.bug_run_infor_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
        # not pass. add the mutant to high order pool
        # get the last line of console output
        console_last_line = text_last_line(console_output)
        logger.debug(f"console_last_line: {console_last_line}")
        logger.debug(f"stack_trace_last_line: {self.stack_trace_last_line}")
        if calculate_text_similarity(console_last_line, self.stack_trace_last_line) > 0.9:
            if mutant.order < self.max_mutation_order - 1:
                logger.debug("==========Insert Code into High Order Candidate Pool==========")
                logger.debug(remove_anchor_code)
                high_mutant = Mutant(remove_anchor_code)
                high_mutant.history = mutant.history.copy()
                high_mutant.history.append(mutator.mutator_name)
                self.high_order_pool[utils.gen_md5_id(remove_anchor_code)] = high_mutant
            else:
                logger.debug(
                    f"Mutant order ({mutant.order}) is larger or equal to max_mutation_order ({self.max_mutation_order}). ")

        else:
            logger.debug("console_last_line != self.stack_trace_last_line")

    def save_mutation(self, pass_mutant, code_md5, is_initial_pass=False):
        mutant_name = "initial-" if is_initial_pass else ""
        for op_id, op in enumerate(pass_mutant.history):
            # op_idntfr
            op_splits = [s[:1].upper() for s in op.split("_")]
            op_idntfr = "".join(op_splits)
            mutant_name += f"{op_idntfr}{op_id + 1}-"
        mutant_name = mutant_name + code_md5
        mutant_path = os.path.join(self.bug_save_dir, self.target_bug, f"{self.target_bug}-{mutant_name}.py")
        utils.write_text(f=mutant_path, content=pass_mutant.code)
        return mutant_path

    def save_mutation_by_code(self, code, code_md5, idntfr=None):
        file_name = f"{self.target_bug}-{code_md5}.py" if idntfr is None else f"{self.target_bug}-{idntfr}-{code_md5}.py"
        mutant_path = os.path.join(self.bug_save_dir, self.target_bug, file_name)
        utils.write_text(f=mutant_path, content=code)
        return mutant_path

    def mutation_status(self):
        # Print the status to the log
        logger.info("### Mutation Status ###")
        logger.info(f"Total passing count: {len(self.pass_pool)}")
        usage = 0
        for mutator in self.mutators:
            mutator_level = mutator.mutator_name
            usage += self.mutation_history[mutator_level]['usage']
            logger.info(
                f"{mutator_level}: Usage: {self.mutation_history[mutator_level]['usage']}, Generate: {self.mutation_history[mutator_level]['generate']}, Pass: {self.mutation_history[mutator_level]['pass']}, Exception: {self.mutation_history[mutator_level]['error']}")
        logger.info(
            f"total result: Usage: {self.all_history['usage']}, Generate: {self.all_history['generate']}, Pass: {self.all_history['pass']}, Fail: {self.all_history['fail']}, Exception: {self.all_history['error']}")
        logger.info(f"Total mutation attempts: {usage}")
        logger.info(f"Total try block duration: {self.total_generate_duration:.4f} seconds")
        logger.info(f"Total except block duration: {self.total_execute_duration:.4f} seconds")
        logger.info("### Mutation Status ###")

        # Save the history information to a file
        output_data = {
            # 'mutation_history': self.mutation_history,
            'all_history': self.all_history,
            'total_generate_duration': self.total_generate_duration,
            'total_execute_duration': self.total_execute_duration,
            'md5_time': self.md5_time,
            'exe_time': self.exe_time,
            'cov_time': self.cov_time,
            'ana_time': self.ana_time,
            'else_time': self.else_time,
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
