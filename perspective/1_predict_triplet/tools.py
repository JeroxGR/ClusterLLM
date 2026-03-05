import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from llm_client import delayed_completion, extract_content

def prepare_data(prompt, datum):
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
    return prompt + input_txt + postfix

def post_process(completion, choices):
    content = extract_content(completion)
    result = []
    for choice in choices:
        choice_txt = "Choice" + choice
        if choice_txt in content:
            result.append(choice)
    return content, result