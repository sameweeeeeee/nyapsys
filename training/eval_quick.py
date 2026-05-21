import json
import argparse
from pathlib import Path

import requests


def load_eval_data(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def query_model(prompt: str, model_path: str, max_tokens: int = 100) -> str:
    response = requests.post(
        "http://127.0.0.1:8080/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def check_repetition(text: str, min_repeat_len: int = 10) -> float:
    words = text.split()
    if len(words) < min_repeat_len * 2:
        return 0.0
    repeats = 0
    for i in range(len(words) - min_repeat_len):
        chunk = " ".join(words[i:i + min_repeat_len])
        if chunk in " ".join(words[i + min_repeat_len:]):
            repeats += 1
    return repeats / max(len(words) - min_repeat_len, 1)


def check_instruction_following(prompt: str, response: str, expected_keywords: list) -> bool:
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in expected_keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model_url", type=str, default="http://127.0.0.1:8080")
    args = parser.parse_args()

    eval_data = load_eval_data(Path(args.eval_data))[:args.n_samples]
    print(f"Evaluating {len(eval_data)} samples...")

    results = {"perplexity_scores": [], "repetition_rates": [], "instruction_following": []}

    for i, sample in enumerate(eval_data):
        prompt = sample.get("prompt", "")
        expected = sample.get("expected_keywords", [])

        try:
            response = query_model(prompt, args.model)
            repetition = check_repetition(response)
            follows = check_instruction_following(prompt, response, expected) if expected else True

            results["repetition_rates"].append(repetition)
            results["instruction_following"].append(follows)

            if i % 20 == 0:
                print(f"Sample {i}: repetition={repetition:.2%}, follows={follows}")
        except Exception as e:
            print(f"Error on sample {i}: {e}")

    avg_repetition = sum(results["repetition_rates"]) / len(results["repetition_rates"]) if results["repetition_rates"] else 0
    avg_instruction = sum(results["instruction_following"]) / len(results["instruction_following"]) if results["instruction_following"] else 0

    print("\n=== Evaluation Results ===")
    print(f"Samples evaluated: {len(eval_data)}")
    print(f"Average repetition rate: {avg_repetition:.2%} (healthy: < 5%)")
    print(f"Instruction following: {avg_instruction:.2%} (healthy: > 70%)")

    if avg_repetition > 0.05:
        print("WARNING: High repetition rate — consider reducing LR or adding repetition penalty")
    if avg_instruction < 0.70:
        print("WARNING: Low instruction following — add more instruction tuning data")


if __name__ == "__main__":
    main()