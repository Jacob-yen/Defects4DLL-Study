import random
MAX_SHAPE_NUMBER = 10

def int_generator(s=None,e=None):
    s = 1 if s is None else s
    e = 100 if e is None else e
    return random.randint(s, e)

def float_generator(s=None,e=None):
    s = 0 if s is None else s
    e = 100 if e is None else e
    return random.uniform(s, e)

def bool_generator(s=None,e=None):
    return random.choice([True, False])


def fetch_generator(t):
    # use match to determin the generator
    if 'int' in t:
        return int_generator
    elif 'float' in t:
        return float_generator
    elif 'bool' in t:
        return bool_generator
    else:
        raise NotImplementedError
    
supportted_types = [
    'int','float','float32','bool',
]

torch_dtypes = ["float","float16","float32","float64",
                "int","uint8","int16","int32","int64",
                "bool","half"]

tf_dtypes = ["bool", "float16", "float32", "float64", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                 "int16", "int8", "complex64", "complex128", "string", "qint8", "quint8", "qint16", "quint16", "qint32",
                 "bfloat16"]
