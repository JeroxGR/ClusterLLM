from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

import os
import argparse
import json
import openai
import numpy as np
from tqdm import tqdm
from tools import delayed_completion, prepare_data, post_process


def entropy_score(d):
    """
    Estimate uncertainty using embedding distance.
    Larger distance difference → lower entropy
    Smaller difference → harder example
    """

    if "dist1" in d and "dist2" in d:
        p = abs(d["dist1"] - d["dist2"])
        return -p
    return 0


def cross_encoder_decision(query, c1, c2):

    pairs = [
        (query, c1),
        (query, c2)
    ]

    scores = cross_encoder.predict(pairs)

    if scores[0] > scores[1]:
        return ["Choice 1"]
    else:
        return ["Choice 2"]


def predict(args):

    openai.organization = args.openai_org
    openai.api_key = os.getenv("OPENAI_API_KEY")

    pred_path = args.data_path.split("/")[-1].replace(
        ".json",
        f"-{args.model_name}{'-temp' + str(round(args.temperature, 1)) if args.temperature > 0 else ''}-pred.json"
    )

    pred_path = os.path.join("predicted_triplet_results", pred_path)

    print("Save in:", pred_path)

    if os.path.exists(pred_path):

        with open(pred_path, "r") as f:
            data = json.load(f)

    else:

        with open(args.data_path, "r") as f:
            data = json.load(f)

    with open("prompts.json", "r") as f:

        prompts = json.load(f)
        task_prompt = prompts[args.dataset]

    for d in data:

        if "prepared" not in d:
            d["prepared"] = prepare_data(task_prompt, d)

    # ---------------------------------------------------
    # NEW PART — Proxy Teacher Selection
    # ---------------------------------------------------

    entropies = [entropy_score(d) for d in data]

    order = np.argsort(entropies)

    llm_budget = int(len(data) * 0.2)

    hard_idx = set(order[-llm_budget:])
    easy_idx = set(order[:-llm_budget])

    print("CrossEncoder triplets:", len(easy_idx))
    print("LLM triplets:", len(hard_idx))

    # ---------------------------------------------------

    for idx, datum in tqdm(enumerate(data), total=len(data)):

        if "prediction" in datum:
            continue

        query = datum["query"]
        c1 = datum["choice1"]
        c2 = datum["choice2"]

        # ---------------------------------------
        # EASY → CrossEncoder
        # ---------------------------------------

        if idx in easy_idx:

            pred = cross_encoder_decision(query, c1, c2)

            data[idx]["content"] = "CrossEncoder"
            data[idx]["prediction"] = pred

        # ---------------------------------------
        # HARD → LLM
        # ---------------------------------------

        else:

            messages = [
                {"role": "user", "content": datum["prepared"]}
            ]

            completion, error = delayed_completion(
                delay_in_seconds=args.delay,
                max_trials=args.max_trials,
                model=args.model_name,
                messages=messages,
                max_tokens=10,
                temperature=args.temperature
            )

            if completion is None:

                print(f"Saving data after {idx + 1} inference.")

                with open(pred_path, "w") as f:
                    json.dump(data, f)

                print(error)
                break

            else:

                content, results = post_process(completion, datum["options"])

                data[idx]["content"] = content
                data[idx]["prediction"] = results

        if idx % args.save_every == 0 and idx > 0:

            print(f"Saving data after {idx + 1} inference.")

            with open(pred_path, "w") as f:
                json.dump(data, f)

    with open(pred_path, "w") as f:
        json.dump(data, f)

    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--openai_org", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1)

    args = parser.parse_args()

    predict(args)