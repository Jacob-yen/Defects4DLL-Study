import os
import sys
sys.path.append(os.getcwd())
import ast
import astor
import re
from abc import ABC
from src.mutation.base import mutation_utils
from src.tools.logger_utils import LoggerUtils
from src.tools.enum_class import Framework

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class ControlFlowMutator(ABC):

    def __init__(self,**kwargs):
        self.framework = kwargs.get("framework", None)

    @staticmethod
    def parse_stacktrace(stacktrace, stacktrace_pattern, entry_rules):
        stacktrace_lines = stacktrace.split("\n")
        entry_code_line_num = 0
        code_detail = ""
        # Get the line of code of the last failing test case in the stack
        for index, stack_trace_line in enumerate(stacktrace_lines):
            if len(re.findall(stacktrace_pattern, stack_trace_line)) > 0:
                entry_code_line_num = int(stack_trace_line.split(", line ")[1].split(",")[0])
                code_detail = stacktrace_lines[index + 1]
        is_incorrect_function = False
        for rule in entry_rules:
            if rule in code_detail:
                is_incorrect_function = True
                logger.debug(f"The entry code is assertion. Skip it.")
        return is_incorrect_function, entry_code_line_num


    def mutation_point_scan(self, source_code, stack_trace, mode=None):
        mutation_points = []
        # get the mutation points of delete_entry_code
        if self.framework == Framework.PYTORCH:
            entry_rules = ["assert"]
            pattern = r'File .+?torch-.+?-original\.py", line .+'
        elif self.framework == Framework.TENSORFLOW:
            pattern = r'File .+?tf-.+?-original\.py", line .+'
            entry_rules = ["self.assert"]
        elif self.framework == Framework.JITTOR:
            pattern = r'File .+?jittor-.+?-original\.py", line .+'
            entry_rules = ["self.assert"]
        elif self.framework == Framework.MINDSPORE:
            pattern = r'File .+?ms-.+?-original\.py", line .+'
            entry_rules = ["self.assert"]
        elif self.framework == Framework.PADDLE:
            pattern = r'File .+?paddle-.+?-original\.py", line .+'
            entry_rules = ["self.assert"]
        elif self.framework == Framework.MXNET:
            pattern = r'File .+?mxnet-.+?-original\.py", line .+'
            entry_rules = ["self.assert"]
        else:
            raise ValueError(f"The framework {self.framework} is not supported")
        # we have two kinds of mutation points
        # control_flow and delete_entry_code
        if mode == "delete_entry_code":
            tree = ast.parse(source=source_code, mode='exec')
            is_incorrect_function, entry_code_line_num = self.parse_stacktrace(stack_trace, pattern,entry_rules)
            if not is_incorrect_function:
                delete_entry_code = mutation_utils.DeleteEntryCode(entry_code_line_num)
                delete_entry_code.scan(tree)
                for point in delete_entry_code.mutation_points:
                    mutation_points.append(
                        {"mode": 'delete_entry_code', "point": point, "entry_code_line_num": entry_code_line_num})

        elif mode == "branch_reversal":
            # get the mutation points of branch_reversal and delete_call_code
            tree = ast.parse(source=source_code, mode='exec')
            visitor = mutation_utils.IfNodeReversal()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({"mode": 'branch_reversal', "point": point})

        elif mode == "delete_call_code":
            tree = ast.parse(source=source_code, mode='exec')
            visitor = mutation_utils.CommentOutExpr()
            visitor.scan(tree)
            for point in visitor.mutation_points:
                mutation_points.append({"mode": 'delete_call_code', "point": point})
        else:
            raise ValueError(f"The mutation mode of control flow level mutation {mode} is not supported")
        return mutation_points

    @staticmethod
    def mutate(source_code, mutation_point_item):
        """
        mutate_mode:
        - delete_entry_code: delete the entry code
        - branch_reversal: reverse the branch of if node
        - delete_call_code: comment out the call code
        """
        mutate_mode, mutation_point = mutation_point_item['mode'], mutation_point_item['point']
        tree = ast.parse(source=source_code, mode='exec')
        if mutate_mode == "delete_entry_code":
            entry_code_line_num = mutation_point_item["entry_code_line_num"]
            delete_entry_code = mutation_utils.DeleteEntryCode(entry_code_line_num)
            delete_entry_code.mutate(tree, mutation_point)
        elif mutate_mode == "branch_reversal":
            visitor = mutation_utils.IfNodeReversal()
            visitor.mutate(tree, mutation_point)
        elif mutate_mode == "delete_call_code":
            visitor = mutation_utils.CommentOutExpr()
            visitor.mutate(tree, mutation_point)
        else:
            raise ValueError(f"The mutation mode of control flow level mutation {mutate_mode} is not supported")
        return astor.to_source(tree)

