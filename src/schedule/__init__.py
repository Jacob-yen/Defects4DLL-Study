class AbstractScheduler:
    def __init__(self, root_result_dir,bug_save_dir, method_save_name,framework, fail_bug, max_test_case_count,
                 time_limit_seconds, config, cov_collector, interpreter_path, source_compile_path) -> None:

        self.root_result_dir = root_result_dir
        self.fail_bug = fail_bug
        self.cov_collector = cov_collector
        self.bug_save_dir = bug_save_dir
        self.framework = framework
        self.method_save_name = method_save_name
        self.time_limit_seconds = time_limit_seconds
        self.max_test_case_count = max_test_case_count
        self.config = config
        self.interpreter_path = interpreter_path
        self.source_compile_path = source_compile_path




