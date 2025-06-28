import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import ast
import random
import astunparse
import copy
from src.mutation.base import TreeMutator
from src.tools.utils import CodeUtils, assertion_points
from src.tools.logger_utils import LoggerUtils

logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger


class ModifyVariable(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit(self, node):
        # if the node is Assign or AugAssign, we need to scan the value of the node
        # if isinstance(node, (ast.Assign, ast.AugAssign)):
        # if is_statements(node):
            # logger.debug(ast.unparse(node))
        if isinstance(node, ast.List):
            # [****]  ——  [None], None, [[****]], [**], (****), , f
            var_type = "List"
        if isinstance(node, ast.Dict):
            # {**:**}  ——  {}, (**,**)
            var_type = "Dict"
        elif isinstance(node, ast.Constant):
            var_type = "Constant"
        elif isinstance(node, ast.Tuple):
            # (****)  ——  shuffle, (**None*), None
            var_type = "Tuple"
        elif isinstance(node, ast.ListComp):
            # [****]  ——  [None], None, 去掉for, 去掉if
            var_type = "ListComp"
        elif isinstance(node, ast.IfExp):
            # if not, None
            var_type = "IfExp"
        elif isinstance(node, ast.Subscript):
            # [:], \,
            var_type = "Subscript"
        elif isinstance(node, ast.BinOp):
            # 转变为减法, 删除其中一个
            var_type = "BinOp"
        elif isinstance(node, ast.Name):
            # 转变为减法, 删除其中一个
            var_type = "Name"
        else:
            var_type = None
        if var_type and self.mode == "scan":
            print(ast.unparse(node), node)
            point = (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
            # mutation_item = {"mode":f"variable_{var_type}", "point":point}
            self.mutation_points.append(point)
        elif (
                self.mode == "mutate"
                and hasattr(node, "lineno")
                and hasattr(node, "end_lineno")
                and hasattr(node, "col_offset")
                and hasattr(node, "end_col_offset")
                and self.target_point["point"]
                == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
        ):
            logger.debug(f"Before mutation: {ast.unparse(node)}")
            if var_type == "List":
                node = ModifyVariable.handle_List(node)
            elif var_type == "Dict":
                node = ModifyVariable.handle_Dict(node)
            elif var_type == "Constant":
                node = ModifyVariable.handle_Constant(node)
            elif var_type == "Tuple":
                node = ModifyVariable.handle_Tuple(node)
            elif var_type == "ListComp":
                node = ModifyVariable.handle_ListComp(node)
            elif var_type == "IfExp":
                node = ModifyVariable.handle_IfExp(node)
            elif var_type == "Subscript":
                node = ModifyVariable.handle_Subscript(node)
            elif var_type == "BinOp":
                node = ModifyVariable.handle_BinOp(node)
            elif var_type == "Name":
                node = ModifyVariable.handle_Name(node)
            else:
                # initialize a None node
                node.value = ast.Constant(value=None)
                # raise ValueError(f"var_type must be List, Constant, Tuple, ListComp, IfExp, Subscript, BinOp, Name. Got: {var_type}")
            logger.debug(f"After mutation: {ast.unparse(node)}")
            return node

        return self.generic_visit(node)

    # [****]  ——  [None], [], [**]
    @staticmethod
    def handle_List(node):
        # 随机选择一个节点
        list_mutate_rules = {0: "Element2None", 2: "EmptyList", 3: "Element2Delete"}
        list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        if list_mutate_index == 0:
            new_elements = []
            # 对他内部的参数有50%的概率变为None
            for elem in node.elts:
                if random.random() < 0.5:
                    new_elements.append(elem)
                else:
                    new_elements.append(ast.Constant(value=None))
            node = ast.List(elts=new_elements, ctx=ast.Load())
        elif list_mutate_index == 1:
            # 空列表
            node = ast.List(elts=[], ctx=ast.Load())
        elif list_mutate_index == 2:
            new_elements = []
            # 对他内部的参数有50%的概率删除
            for elem in node.elts:
                if random.random() < 0.5:
                    new_elements.append(elem)
            node = ast.List(elts=new_elements, ctx=ast.Load())
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    @staticmethod
    def handle_Dict(node):
        # print(node)
        # 随机选择一个节点
        # list_mutate_rules = {0: "Element2None", 1: "EmptyDict", 2: "Convert2List", 3: "Convert2Tuple"}
        list_mutate_rules = {2: "Convert2List", 3: "Convert2Tuple"}
        # list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        list_mutate_index = random.choice(list(list_mutate_rules.keys()))
        if list_mutate_index == 0:
            new_elements = []
            # 对他内部的参数有50%的概率变为None
            for elem in node.keys:
                if random.random() < 0.5:
                    new_elements.append(elem)
                else:
                    new_elements.append(ast.Constant(value=None))
            node = ast.Dict(keys=new_elements, values=new_elements)
        elif list_mutate_index == 1:
            # 空字典
            node = ast.Dict(keys=[], values=[])
        elif list_mutate_index in [2, 3]:
            # convert the {key1:value1,key2:value2} to [key1,value1,key2,value2]/(key1,value1,key2,value2)
            new_elements = []
            for key, value in zip(node.keys, node.values):
                new_elements.append(key)
                new_elements.append(value)
            if list_mutate_index == 2:
                node = ast.List(elts=new_elements, ctx=ast.Load())
            else:
                node = ast.Tuple(elts=new_elements, ctx=ast.Load())
            # print(ast.unparse(node))
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    @staticmethod
    def handle_Constant(node):
        if isinstance(node.value, (bool)):
            # 反转
            new_value = not node.value
            node = ast.Constant(value=new_value)
        elif isinstance(node.value, (int, float, complex)):
            if random.random() < 0.5:
                const_dict = {0: 0, 1: 1, 2: -1, 3: 'inf'}
                value = const_dict.get(np.random.randint(0, len(const_dict.keys())))
                node = ast.Constant(value=float(value))
            else:
                # cast to float
                if isinstance(node.value, (int)):
                    node = ast.Constant(value=float(node.value))
                elif isinstance(node.value, (float)):
                    node = ast.Constant(value=int(node.value))
        elif isinstance(node.value, (str)):
            # None
            node = ast.Constant(value=None)
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    # (****)  ——  shuffle, (**None*), None, ()
    @staticmethod
    def handle_Tuple(node):
        # list_mutate_rules = {0: "ElementShuffle", 1: "Element2None", 2: "All2None", 3: "EmptyTuple", 4: "RepeatElement"}
        list_mutate_rules = { 4: "RepeatElement"}
        # list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        list_mutate_index = random.choice(list(list_mutate_rules.keys()))
        # list_mutate_index = 2
        if list_mutate_index == 0:
            # 打乱元组中的元素
            new_elements = random.sample(node.elts, len(node.elts))
            node = ast.Tuple(elts=new_elements, ctx=ast.Load())
        elif list_mutate_index == 1:
            new_elements = []
            # 对他内部的参数有50%的概率变为None
            for elem in node.elts:
                if random.random() < 0.5:
                    new_elements.append(elem)
                else:
                    new_elements.append(ast.Constant(value=None))
            node = ast.Tuple(elts=new_elements, ctx=ast.Load())
        elif list_mutate_index == 2:
            # 变为None
            node = ast.Constant(value=None)
        elif list_mutate_index == 3:
            # 空元组
            node = ast.Tuple(elts=[], ctx=ast.Load())
        elif list_mutate_index == 4:
            # repeat the last element
            new_elements = node.elts + [node.elts[-1]]
            node = ast.Tuple(elts=new_elements, ctx=ast.Load())
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    # [****]  ——  [None], [], 去掉if
    @staticmethod
    def handle_ListComp(node):
        list_mutate_rules = {0: "Element2None", 1: "EmptyList", 2: "DeleteIf"}
        list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        # list_mutate_index = 2
        if list_mutate_index == 0:
            # 元素变为None
            node = ast.List(elts=[ast.Constant(value=None)], ctx=ast.Load())
        elif list_mutate_index == 1:
            # 空列表
            node = ast.List(elts=[], ctx=ast.Load())
        elif list_mutate_index == 2:
            # 删除列表表达式的if部分
            if hasattr(node, 'generators'):
                node.generators[0].ifs = []
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    # if not, None
    @staticmethod
    def handle_IfExp(node):
        list_mutate_rules = {0: "SwitchBranch", 1: "All2None"}
        list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        # list_mutate_index = 1
        if list_mutate_index == 0:
            # 将if分支进行翻转
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif list_mutate_index == 1:
            # 变为None
            node = ast.Constant(value=None)
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    # 删除索引[:]
    @staticmethod
    def handle_Subscript(node):
        # 随机选择一个节点
        # 删除索引[:]
        node = node.value
        return node

    # 切换操作符, 删除其中一个
    @staticmethod
    def handle_BinOp(node):
        list_mutate_rules = {0: "SwitchOp", 1: "RemoveLeftOrRight"}
        list_mutate_index = np.random.randint(0, len(list_mutate_rules.keys()))
        # list_mutate_index = 1
        if list_mutate_index == 0:
            # 将二元操作符进行随机的更改
            ops = [ast.Add, ast.Sub, ast.Mult, ast.Div]
            rand_op = random.choice(ops)
            node = ast.BinOp(left=node.left, op=rand_op(), right=node.right)
        elif list_mutate_index == 1:
            # 一半的概率留下左节点, 一半的概率留下右节点
            if random.random() < 0.5:
                node = node.left
            else:
                node = node.right
        else:
            logger.error("The index you selected exceeds the limit!")
        return node

    @staticmethod
    def handle_Name(node):
        # 变为None
        node = ast.Constant(value=None)
        return node


class DeleteEntryCode(ast.NodeTransformer, TreeMutator):
    def __init__(self, entry_code_line_num):
        super().__init__()
        self.entry_code_line_num = entry_code_line_num

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit(self, node):
        if hasattr(node, "lineno") and node.lineno == self.entry_code_line_num:
            if self.mode == "scan":
                # if the node is a statement, we add the position of the statement to the mutation points
                if is_statements(node):
                    self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
                    # only record the parent node
                    return node
                else:
                    logger.debug("The entry code is not a statement. We will not record it.")
            elif self.mode == "mutate" and self.target_point == (
            node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                code_line = astunparse.unparse(node)
                if "anchor" in code_line:
                    return node
                if isinstance(node, ast.Return):
                    node = ast.Return(value=ast.Name(id="None", ctx=ast.Load()))
                else:
                    # we change the node to a pass node
                    node = ast.Pass()
                return node
        return self.generic_visit(node)


class IfNodeReversal(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_If(self, node):
        # the node cannot be if __name__ == "__main__"
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left,
                                                             ast.Name) and node.test.left.id == "__name__":
            return self.generic_visit(node)
        else:
            if self.mode == "scan":
                self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
            elif self.mode == "mutate":
                if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                    # reverse the condition
                    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                    return node
            return self.generic_visit(node)


class CommentOutExpr(ast.NodeTransformer, TreeMutator):
    def __init__(self):
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Expr(self, node):

        code_line = astunparse.unparse(node)
        if not CodeUtils.has_assertion(code_line) and "anchor" not in code_line:
            if self.mode == "scan":
                self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
            elif self.mode == "mutate":
                if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                    node.value = ast.Expr(value=ast.Str(s=f"# {astunparse.unparse(node.value).strip()}"))
                    return node
            else:
                raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


class RemoveAPIKeyword(ast.NodeTransformer, TreeMutator):
    def __init__(self) -> None:
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Call(self, node):
        if len(node.keywords) > 0:
            if self.mode == "scan":
                self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
            elif self.mode == "mutate":
                if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                    # get the number of keyword args
                    num_keywords = len(node.keywords)
                    # randomly remove one
                    if num_keywords > 0:
                        random_keyword = random.choice(node.keywords)
                        node.keywords.remove(random_keyword)
                    return node
            else:
                raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


class RemoveSubscript(ast.NodeTransformer, TreeMutator):
    def __init__(self) -> None:
        super().__init__()

    def scan(self, tree):
        self.mode = "scan"
        self.visit(tree)

    def mutate(self, tree, mutation_point):
        self.mode = "mutate"
        self.target_point = mutation_point
        self.visit(tree)

    def visit_Subscript(self, node):
        if self.mode == "scan":
            self.mutation_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
        elif self.mode == "mutate":
            if self.target_point == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset):
                node = node.value
                return node
        else:
            raise ValueError("mode must be scan or mutate. It seems you directly invoke visit method.")
        return self.generic_visit(node)


def flip_mutation(tree, mutation_point, values):
    lineno, end_lineno, col_offset, end_col_offset = mutation_point
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant,
                             ast.NameConstant)) and node.lineno == lineno and node.end_lineno == end_lineno and node.col_offset == col_offset and node.end_col_offset == end_col_offset:
            if str(node.value) in values:
                node.value = values[0] if str(node.value) == values[1] else values[1]
    return tree


def target_value_scan(tree, targets):
    target_points = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            if "cuda" in targets:
                if str(node.value) == 'cpu' or str(node.value).startswith('cuda'):
                    target_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
            else:
                if str(node.value) in targets:
                    target_points.append((node.lineno, node.end_lineno, node.col_offset, node.end_col_offset))
    assertions = [t[0] for t in assertion_points(tree)]
    for target_point in target_points:
        if target_point[0] in assertions:
            # print(f"remove assertion {target_point[0]}")
            target_points.remove(target_point)
    return target_points


def is_statements(node):
    return isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Expr, ast.Assign, ast.AugAssign,
                             ast.AnnAssign, ast.Return, ast.Delete, ast.Pass, ast.Import, ast.ImportFrom, ast.Global,
                             ast.Nonlocal, ast.Assert, ast.Raise, ast.Try, ast.With, ast.AsyncWith))

