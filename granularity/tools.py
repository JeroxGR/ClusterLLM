import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_client import delayed_completion, extract_content

def post_process(completion):
    content = extract_content(completion)
    result = []
    if 'Yes' in content and 'No' not in content:
        result.append('Yes')
    elif 'No' in content and 'Yes' not in content:
        result.append('No')
    return content, result