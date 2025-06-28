"""
These classes are fake enum classes.
"""


class Granularity:
    BLOCK = 1
    FUNCTION = 2
    FILE = 3
    DICT = {"block": BLOCK, "function": FUNCTION, "file": FILE}
    NAME = {BLOCK:"block", FUNCTION:"function", FILE:"file"}
    TOTAL = [BLOCK, FUNCTION, FILE]


class Approach:
    DEVELOPER = "developer"
    DOCTER = "docter"
    DEEPREL = "deeprel"
    OUR = "our"
    LLM = "llm"
    HYBRID = "hybrid"
    RULE = "rule"
    BASELINE = [DEVELOPER, DOCTER, DEEPREL]
    TOTAL = [DEVELOPER, DOCTER, DEEPREL, HYBRID, OUR, LLM, RULE]


class Framework:
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    JITTOR = "jittor"
    MXNET = "mxnet"
    PADDLE = "paddlepaddle"
    MINDSPORE = "mindspore"
    TOTAL = [TENSORFLOW, PYTORCH, JITTOR, MXNET, PADDLE, MINDSPORE]


class TestType:
    PASS = "pass"
    FAIL = "fail"
    TOTAL = [PASS, FAIL]


class Symptom:
    CRASH = "crash"
    ASSERT = "assert"
    TOTAL = [CRASH, ASSERT]