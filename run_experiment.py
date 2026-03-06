#!/usr/bin/env python3
"""
Run ClusterLLM triplet prediction across multiple Groq models and generate a comparison report.
Handles Groq free-tier rate limits with proper delays between models.
"""
import os
import sys
import json
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
TRIPLET_DIR = os.path.join(ROOT, "perspective", "1_predict_triplet")
SAMPLED_FILE = os.path.join(
    TRIPLET_DIR, "sampled_triplet_results",
    "banking77_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100.json"
)

MODELS = [
    ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ("qwen/qwen3-32b", "Qwen3 32B"),
    ("openai/gpt-oss-20b", "OpenAI GPT-OSS 20B"),
    ("llama-3.1-8b-instant", "Llama 3.1 8B"),
]

DELAY = 4
MAX_TRIALS = 10


def get_pred_path(model_name):
    safe = model_name.replace("/", "-")
    return os.path.join(
        TRIPLET_DIR, "predicted_triplet_results",
        f"banking77_embed=instructor_s=small_m=1024_d=67.0_sf_choice_seed=100-{safe}-pred.json"
    )


def count_done(model_name):
    path = get_pred_path(model_name)
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        data = json.load(f)
    done = sum(1 for d in data if "prediction" in d)
    return done, len(data)


def run_model(model_name, label):
    """Run triplet prediction for a single model with retries."""
    done, total = count_done(model_name)
    if total > 0 and done >= total:
        print(f"  SKIP {label}: already complete ({done}/{total})")
        return 0
    print(f"  Starting {label}: {done}/{total or '?'} done so far...")

    cmd = [
        sys.executable, "predict_triplet.py",
        "--dataset", "banking77",
        "--data_path", SAMPLED_FILE,
        "--openai_org", "",
        "--model_name", model_name,
        "--temperature", "0",
        "--delay", str(DELAY),
        "--max_trials", str(MAX_TRIALS),
        "--save_every", "50",
    ]

    result = subprocess.run(cmd, cwd=TRIPLET_DIR)
    done, total = count_done(model_name)
    print(f"  Result {label}: {done}/{total}")
    return result.returncode


def analyze_predictions(model_name, label):
    """Analyze prediction results for a model."""
    pred_file = get_pred_path(model_name)
    if not os.path.exists(pred_file):
        return None

    with open(pred_file) as f:
        data = json.load(f)

    total = len(data)
    predicted = [d for d in data if "prediction" in d]
    n_pred = len(predicted)
    valid = sum(1 for d in predicted if len(d.get("prediction", [])) == 1)
    choice1 = sum(1 for d in predicted if d.get("prediction") == [" 1"])
    choice2 = sum(1 for d in predicted if d.get("prediction") == [" 2"])
    ambiguous = n_pred - valid

    # Check accuracy if ground truth exists
    correct = 0
    has_gt = False
    for d in predicted:
        if "answer" in d and len(d.get("prediction", [])) == 1:
            has_gt = True
            if d["prediction"][0] == d["answer"]:
                correct += 1

    return {
        "label": label,
        "model": model_name,
        "total": total,
        "predicted": n_pred,
        "valid": valid,
        "choice1": choice1,
        "choice2": choice2,
        "ambiguous": ambiguous,
        "completion": round(n_pred / total * 100, 1) if total > 0 else 0,
        "valid_rate": round(valid / n_pred * 100, 1) if n_pred > 0 else 0,
        "accuracy": round(correct / valid * 100, 1) if has_gt and valid > 0 else None,
    }


def generate_report(results):
    """Generate a comparison report."""
    report = []
    report.append("=" * 70)
    report.append("  ClusterLLM Multi-Model Comparison Report (Banking77)")
    report.append("=" * 70)
    report.append("")
    report.append("Dataset: Banking77 (3080 samples, 77 intent classes)")
    report.append("Task: Triplet prediction (1024 triplets)")
    report.append("Provider: Groq (free tier)")
    report.append("")

    hdr = f"{'Model':<22} {'Done':>10} {'Valid':>6} {'Valid%':>7} {'C1':>5} {'C2':>5} {'Ambig':>6}"
    report.append(hdr)
    report.append("-" * 70)

    for r in results:
        if r is None:
            continue
        report.append(
            f"{r['label']:<22} "
            f"{r['predicted']:>4}/{r['total']:<5} "
            f"{r['valid']:>5} "
            f"{r['valid_rate']:>6.1f}% "
            f"{r['choice1']:>5} "
            f"{r['choice2']:>5} "
            f"{r['ambiguous']:>5}"
        )

    report.append("-" * 70)
    report.append("")
    report.append("Legend:")
    report.append("  Done   = Predictions completed / Total triplets")
    report.append("  Valid  = Predictions with exactly one choice (Choice 1 or 2)")
    report.append("  Valid% = % of predictions in correct format")
    report.append("  C1/C2  = How many times the model chose Choice 1 vs Choice 2")
    report.append("  Ambig  = Ambiguous/invalid responses (no clear single choice)")
    report.append("")

    return "\n".join(report)


def main():
    os.makedirs(os.path.join(TRIPLET_DIR, "predicted_triplet_results"), exist_ok=True)
    print("ClusterLLM Multi-Model Experiment Runner")
    print(f"Models to test: {len(MODELS)}")
    for model_name, label in MODELS:
        print(f"  - {label} ({model_name})")

    # Run each model with retries
    for i, (model_name, label) in enumerate(MODELS):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(MODELS)}] {label}")
        print(f"{'='*60}")
        for attempt in range(5):
            rc = run_model(model_name, label)
            done, total = count_done(model_name)
            if total > 0 and done >= total:
                break
            if attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"  Incomplete ({done}/{total}), retry after {wait}s cooldown...")
                time.sleep(wait)
        if i < len(MODELS) - 1:
            print(f"  Waiting 30s before next model...")
            time.sleep(30)

    # Analyze results
    print("\n\nAnalyzing results...")
    results = []
    for model_name, label in MODELS:
        r = analyze_predictions(model_name, label)
        results.append(r)

    # Generate and print report
    report = generate_report(results)
    print(report)

    # Save report
    report_path = os.path.join(ROOT, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
