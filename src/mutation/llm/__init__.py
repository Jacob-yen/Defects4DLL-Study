system_prompt_template = """You are a deep learning programmer with expertise in writing {} programs in Python."""
user_prompt_template = """You are provided with a program that triggers a bug in {}.\n
You are required to modify the program to make it run successfully with no exception.
Program: 
{}

Exception Stack Trace: 
{}

Please return the modified program surrounded by ```python ``` and explain the reason of your modification step by step.
Response:
"""
feedback_template = """Your solution is wrong! The return program still throws an exception. 
Exception Stack Trace: 
{}

Please try again to change the program to make it run successfully with no exception.
"""
