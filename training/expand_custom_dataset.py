import json
import os
import argparse
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are helping create training data for Nyapsys, a self-hosted AI assistant running locally on a MacBook.
Generate diverse, realistic user-assistant conversations that cover the following scenarios:
- How Nyapsys introduces itself
- Handling uncertainty ("I don't know" vs hallucinating)
- File analysis (PDF summaries, CSV interpretation, table reading)
- Refusals and edge cases
- Preferred response length and tone
- Multi-turn conversation coherence

Each example should be a JSON object with a "messages" array containing system, user, and assistant messages.
The assistant should always identify as Nyapsys and mention it runs locally on the user's Mac."""


def load_seeds(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def expand_seed(seed: dict, client: OpenAI, n: int) -> list:
    prompt = f"""Given this seed conversation, generate {n} variations that maintain the same intent but use different wording, topics, and contexts.

Seed:
{json.dumps(seed, indent=2)}

Return only valid JSONL, one example per line. Each example must have the same structure with a "messages" array."""

    response = client.chat.completions.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        max_tokens=4000,
    )

    examples = []
    for line in response.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if line.startswith("```"):
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_path", type=str, default="training/data/custom/seeds.jsonl")
    parser.add_argument("--output_path", type=str, default="training/data/custom/expanded.jsonl")
    parser.add_argument("--target_count", type=int, default=8000)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY"))
    seeds = load_seeds(Path(args.seeds_path))
    per_seed = args.target_count // len(seeds)

    print(f"Expanding {len(seeds)} seeds to ~{args.target_count} examples ({per_seed} each)")

    all_examples = []
    for i, seed in enumerate(seeds):
        print(f"Expanding seed {i + 1}/{len(seeds)}...")
        examples = expand_seed(seed, client, per_seed)
        all_examples.extend(examples)
        print(f"  Got {len(examples)} examples")

    with open(args.output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(all_examples)} examples to {args.output_path}")


if __name__ == "__main__":
    main()