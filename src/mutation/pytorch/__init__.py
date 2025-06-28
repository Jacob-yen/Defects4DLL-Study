api_sub_mutators = [
    "api_parameter",
    "api_replace",
    "api_decorator",
    "keyword_removing"
]

control_flow_sub_mutators = [
    "branch_reversal",
    "delete_call_code",
    "delete_entry_code"
]

device_sub_mutators = [
    "flip",
    "remove_cuda_to_device",
    "remove_device_call"
]

variable_sub_mutators = [
    "modify_variable",
    "boolean",
    "subscript_removing"
]

graph_sub_mutators = [
    'insert_no_grad',
    'remove_jitCalls',
    'add_jitCalls',
    'remove_jitDecorators',
    'return_none',
    'return_input',
    'remove_layer'
]

sub_mutators_mapping = {
    "api": api_sub_mutators,
    "control_flow": control_flow_sub_mutators,
    "device": device_sub_mutators,
    "variable": variable_sub_mutators,
    "graph": graph_sub_mutators
}