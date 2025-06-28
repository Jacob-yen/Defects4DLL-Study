import os
import sys
sys.path.append(os.getcwd())
from src.tools.utils import LLMBot
import re
def check_response(text):
    matches = re.findall(r'```python.*?```', text, re.DOTALL)
    return len(matches) == 1

def parse_response(text):
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None

class LLMMutator():
    @staticmethod
    def conduct_llm_mutation(framework, source_code, traceback, chatbot:LLMBot):
        prompt = "You are an expert deep learning programmer with lots of experience programming Python code using {}\n" \
                 "The following code will trigger an bug in {}.\n" \
                 "Code:\n{}\nTraceback:\n{}\n" \
                 "You are reuqired to understand the code and strack trace information, and return the modified code that will not trigger the bug.\n" \
                 "Please should return the code surrounded by ```python ``` and explain the reason of your modification.\n" \
                 "Response:\n".format(framework,framework,source_code,traceback)
        messages = [{"role": "user", "content": prompt},]
        print(prompt)
        raw_response = chatbot.chat_completion(messages)

        print("Raw Response: \n", raw_response)
        # check whether the response contains only one ```python and ```
        if check_response(raw_response):
            if code:=parse_response(raw_response):
                return code
            else:
                return source_code
        else:
            return source_code


if __name__ == "__main__":
    pass