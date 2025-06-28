import re
import ast
import astunparse
from src.tools.enum_class import Framework

ANCHOR1 = '"""anchor start"""'
ANCHOR2 = '"""anchor end"""'
marker = "<*>"


def parse_output(output):
    output_lines = [s[len(marker):].strip() for s in output.split("\n") if s.startswith(marker)]
    return output_lines


def remove_oracle(code):
    # we need to remove the coe between the anchor
    lines = code.split('\n')
    start_anchor = None
    end_anchor = None
    for i in range(len(lines)):
        if ANCHOR1 in lines[i].strip():
            start_anchor = i
        if ANCHOR2 in lines[i].strip():
            end_anchor = i
    if start_anchor is None or end_anchor is None:
        raise Exception("No anchor found in the code.")
    else:
        # remove the code between the anchor
        return '\n'.join(lines[:start_anchor + 1] + lines[end_anchor:])

class CodeInstrumenter:

    @staticmethod
    def target_assertion_location(trace, filename):
        # print(filename)
        # pattern = rf'File ".+?/{re.escape(filename)}", line (\d+),'
        # pattern = rf'File ".+?/{re.escape(filename)}"|"{re.escape(filename)}", line (\d+),'
        pattern = rf'File ".*{re.escape(filename)}", line (\d+)'
        matches = re.findall(pattern, trace)
        line_number = int(matches[-1])
        return line_number

    @staticmethod
    def process_assertion(code, irrelevant_linenos, target_lineno):
        lines = code.split('\n')
        for line_number in irrelevant_linenos:
            s_lineo, e_lineno = line_number.split('-')
            s_lineo, e_lineno = int(s_lineo) - 1, int(e_lineno) - 1
            # the line no is not start with zero.
            if s_lineo != e_lineno:
                # multi lines
                for i in range(s_lineo, e_lineno + 1):
                    indent = CodeInstrumenter.get_indent(lines[i])
                    lines[i] = indent + '""" ' + lines[i][len(indent):] + '"""'
            else:
                # single line
                indent = CodeInstrumenter.get_indent(lines[s_lineo])
                lines[line_number] = indent + '""" ' + lines[line_number][len(indent):] + '"""'

        s_lineo, e_lineno = target_lineno.split('-')
        s_lineo, e_lineno = int(s_lineo) - 1, int(e_lineno) - 1
        # insert the anchor before target assertion
        indent = CodeInstrumenter.get_indent(lines[s_lineo])
        lines.insert(s_lineo, indent + ANCHOR2)
        lines.insert(s_lineo, indent + ANCHOR1)
        # remove the assertion
        new_lines = lines[:s_lineo + 2] + lines[e_lineno + 2 + 1:]
        return '\n'.join(new_lines)

    @staticmethod
    def get_indent(line):
        indent = ''
        for char in line:
            if char.isspace():
                indent += char
            else:
                break
        return indent

    @staticmethod
    def parse_assertion(stmt):
        tree = ast.parse(stmt, mode='exec')
        input_parameters = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    parameter_text = astunparse.unparse(arg).strip()
                    input_parameters.append(parameter_text)
                # only the first call
                break
        return input_parameters

    @staticmethod
    def variable_dtype_code(commented_code, input_params, target_assertion, framework):
        # add the variable monitor
        dtype_stmts = []
        variables = []
        for param_idx, param in enumerate(input_params):
            stmt = f"global tmp_monitor{param_idx}"
            dtype_stmts.append(stmt)
            variables.append(f"tmp_monitor{param_idx}")
            stmt = f"tmp_monitor{param_idx} = {param}"
            dtype_stmts.append(stmt)
            # stmt = f"print('{marker}', tmp_monitor{param_idx})"
            # stmt = f"print('{marker}', type(tmp_monitor{param_idx}))"
            if framework == Framework.PYTORCH:
                stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, torch.Tensor))"
                dtype_stmts.append(stmt)
            else:
                stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, tf.Tensor))"
                dtype_stmts.append(stmt)
            stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, list))"
            dtype_stmts.append(stmt)
            stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, dict))"
            dtype_stmts.append(stmt)
            stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, np.ndarray))"
            dtype_stmts.append(stmt)
            stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, int))"
            dtype_stmts.append(stmt)
            stmt = f"print('{marker}', isinstance(tmp_monitor{param_idx}, float))"
            dtype_stmts.append(stmt)

        indent = CodeInstrumenter.get_indent(target_assertion)
        dtype_stmts = [indent + stmt for stmt in dtype_stmts]
        dtype_code = CodeInstrumenter.instrument_code(commented_code, dtype_stmts)
        return dtype_code, variables

    @staticmethod
    def variable_monitor_detection_code(monitors, output, monitor_types, input_params, file_dir, framework,
                                        assert_type):
        # add the variable monitor
        monitor_code = []
        # stmts = []
        # dtype_stmts = []
        # variables = []
        other_index = 0
        for param_idx, param in enumerate(input_params):
            monitor_type = monitor_types[param_idx]
            tmp_monitor = monitors[param_idx]

            stmt = f"global {tmp_monitor}"
            monitor_code.append(stmt)
            stmt = f"{tmp_monitor} = {param}"
            monitor_code.append(stmt)
            if monitor_type == "tensor":
                stmt = f"{tmp_monitor}_array = np.load('{file_dir}_{tmp_monitor}.npy')"
                monitor_code.append(stmt)
                if framework == Framework.PYTORCH:
                    stmt = f'{tmp_monitor}_oracle = torch.tensor({tmp_monitor}_array)'
                    monitor_code.append(stmt)
                else:
                    stmt = f'{tmp_monitor}_oracle = tf.convert_to_tensor({tmp_monitor}_array)'
                    monitor_code.append(stmt)

                stmt = f'{assert_type}({tmp_monitor}, {tmp_monitor}_oracle)'
                monitor_code.append(stmt)
            elif monitor_type == "list" or monitor_type == "dict":
                stmt = f"with open('{file_dir}_{tmp_monitor}.pickle', 'rb') as file:"
                monitor_code.append(stmt)
                stmt = f"   {tmp_monitor}_oracle = pickle.load(file)"
                monitor_code.append(stmt)
                stmt = f'{assert_type}({tmp_monitor}, {tmp_monitor}_oracle)'
                monitor_code.append(stmt)
            elif monitor_type == "ndarray":
                stmt = f"{tmp_monitor}_oracle = np.load('{file_dir}_{tmp_monitor}.npy')"
                monitor_code.append(stmt)
                stmt = f'{assert_type}({tmp_monitor}, {tmp_monitor}_oracle)'
                monitor_code.append(stmt)
            elif monitor_type == "float" and assert_type != "self.assertEqual":
                stmt = f"with open('{file_dir}_{tmp_monitor}.txt', 'r') as file:"
                monitor_code.append(stmt)
                stmt = f"   {tmp_monitor}_oracle = float(file.read())"
                monitor_code.append(stmt)
                stmt = f'{assert_type}({tmp_monitor}, {tmp_monitor}_oracle)'
                monitor_code.append(stmt)
            else:
                actual_values = [s[len(marker):].strip() for s in output.split("\n") if s.startswith(marker)]
                if len(actual_values) <= other_index:
                    return monitor_code, False
                stmt = f'{assert_type}(str({tmp_monitor}),"""{actual_values[other_index]}""")'
                monitor_code.append(stmt)
                other_index += 1

        return monitor_code, True

    # {0: "tensor", 1: "list", 2: "dict", 3: "ndarray", 4: "int", 5: "float", 6: "other"}
    @staticmethod
    def variable_monitor_gain_code(input_params, target_assertion, monitors, monitor_types, assert_type, framework,
                                   file_dir):
        # add the variable monitor
        monitor_code = []
        for param_idx, param in enumerate(input_params):
            monitor_type = monitor_types[param_idx]
            tmp_monitor = monitors[param_idx]

            stmt = f"global {tmp_monitor}"
            monitor_code.append(stmt)
            stmt = f"{tmp_monitor} = {param}"
            monitor_code.append(stmt)
            if monitor_type == "tensor":
                if framework == Framework.PYTORCH:
                    stmt = f"np.save('{file_dir}_{tmp_monitor}.npy', {tmp_monitor}.cpu().detach().numpy())"
                    monitor_code.append(stmt)
                else:
                    stmt = f"np.save('{file_dir}_{tmp_monitor}.npy', {tmp_monitor}.cpu().numpy())"
                    monitor_code.append(stmt)
            elif monitor_type == "list" or monitor_type == "dict":
                stmt = f"with open('{file_dir}_{tmp_monitor}.pickle', 'wb') as file:"
                monitor_code.append(stmt)
                stmt = f"   pickle.dump({tmp_monitor}, file)"
                monitor_code.append(stmt)
            elif monitor_type == "ndarray":
                stmt = f"np.save('{file_dir}_{tmp_monitor}.npy', {tmp_monitor})"
                monitor_code.append(stmt)
            elif monitor_type == "float" and assert_type != "self.assertEqual":
                stmt = f"with open('{file_dir}_{tmp_monitor}.txt', 'w') as file:"
                monitor_code.append(stmt)
                stmt = f"   file.write(str({tmp_monitor}))"
                monitor_code.append(stmt)
            else:
                stmt = f"print('{marker}', {tmp_monitor})"
                monitor_code.append(stmt)

        indent = CodeInstrumenter.get_indent(target_assertion)

        monitor_code = [indent + stmt for stmt in monitor_code]
        return monitor_code

    @staticmethod
    def instrument_code(code, monitor_code):
        # check whether the code contains the anchor
        if ANCHOR1 not in code or ANCHOR2 not in code:
            print(code)
            raise ValueError("The code has no anchor! please check the mutation process")
        else:
            # insert the monitor_code between the anchors
            lines = code.split('\n')
            for idx, line in enumerate(lines):
                if ANCHOR1 in line:
                    indent = CodeInstrumenter.get_indent(line)
                    indented_code = [indent + line for line in monitor_code]
                    lines = lines[:idx + 1] + indented_code + lines[idx + 1:]
                    return '\n'.join(lines)
            else:
                print(code)
                raise ValueError("The code has no anchor but passes the check!")

    @staticmethod
    def insert_oracle(code, stmts):
        # we need to remove the coe between the anchor
        lines = code.split('\n')
        print(lines)
        for i in range(len(lines)):
            if ANCHOR2 in lines[i].strip():
                end_anchor = i
                indent = CodeInstrumenter.get_indent(lines[end_anchor])
                for idx, stmt in enumerate(stmts):
                    lines.insert(end_anchor + idx, indent + stmt)
                return '\n'.join(lines)
        else:
            raise Exception("No Ending anchor found in the code.")
        # insert the code between line at position i
