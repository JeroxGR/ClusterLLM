import json, os
import argparse
import random
from copy import deepcopy
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--pred_path", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--data_path", default=None, type=str)
parser.add_argument("--e5", action="store_true",
                    help="E5 does not allow instructions.")
args = parser.parse_args()

with open(args.pred_path, 'r') as f:
    pred_data = json.load(f)

with open(args.data_path, 'r') as f:
    inp_data = [json.loads(l) for l in f.readlines()]

# the same prompt because it is used for clustering on the same dataset
if not args.e5:
    with open("prompts.json", 'r') as f:
        prompt = json.load(f)[args.dataset]
else:
    prompt = 'query: '

out_data = []
for pd in pred_data:

    # skip if output is None
    if pd['output'] is None:
        continue

    # make sure output is a string
    out = str(pd['output']).strip()

    if out not in ['1', '2']:
        print(pd)
        continue

    if out == '1':
        pos = inp_data[pd['choice1_idx']]['input']
        neg = inp_data[pd['choice2_idx']]['input']

    elif out == '2':
        neg = inp_data[pd['choice1_idx']]['input']
        pos = inp_data[pd['choice2_idx']]['input']
    else:
        print(pd)
        continue
    out_data.append({
        'query': [prompt, inp_data[pd['query_idx']]['input']],
        'pos': [prompt, pos],
        'neg': [prompt, neg],
        'task_name': args.dataset,
        'query_idx': pd['query_idx'],
        'choice1_idx': pd['choice1_idx'],
        'choice2_idx': pd['choice2_idx']
    })

print(out_data[0])
print(out_data[-10])

output_path = os.path.join(args.output_path, args.pred_path.split("/")[-1].replace("-pred.json", "-train.json"))
assert not os.path.exists(output_path)
print(output_path)
with open(output_path, 'w') as f:
    json.dump(out_data, f)