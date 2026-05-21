import json
import random
from pathlib import Path
from datasets import load_dataset
from dataset_config import PRETRAIN_DATASETS, TARGET_PRETRAIN_TOKENS, TRAIN_EVAL_SPLIT, MIN_TOKENS, MAX_TOKENS
from datasketch import MinHash, MinHashLSH

OUTPUT_DIR = Path(__file__).parent / "data" / "tokenized"
CUSTOM_DATA_PATH = Path(__file__).parent / "data" / "custom" / "seeds.jsonl"
random.seed(42)

CHAT_TEMPLATE = "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}\n"
SYSTEM_MESSAGE = "You are Nyapsys, a self-hosted AI assistant running locally on your Mac. You answer questions accurately, read and analyse files, and understand images. Be concise but thorough. If you are unsure, say so."


def simple_tokenize(text: str) -> list:
    return text.split()


def deduplicate_with_minhash(samples: list, threshold: float = 0.85) -> list:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique = []
    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        tokens = simple_tokenize(text)
        if len(tokens) < 10:
            continue
        m = MinHash(num_perm=128)
        for token in tokens:
            m.update(token.encode("utf8"))
        try:
            lsh.insert(f"sample_{i}", m)
            unique.append(sample)
        except:
            unique.append(sample)
    return unique


def filter_and_truncate(text: str) -> str | None:
    tokens = simple_tokenize(text)
    if len(tokens) < MIN_TOKENS:
        return None
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    return " ".join(tokens)


def format_instruction(example: dict) -> dict | None:
    user = example.get("instruction", "")
    assistant = example.get("output", "")
    if not user or not assistant:
        return None
    text = CHAT_TEMPLATE.format(system=SYSTEM_MESSAGE, user=user, assistant=assistant)
    filtered = filter_and_truncate(text)
    return {"text": filtered} if filtered else None


def format_custom(example: dict) -> dict | None:
    messages = example.get("messages", [])
    if not messages:
        return None
    parts = [f"<|system|>\n{SYSTEM_MESSAGE}"]
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    text = "\n".join(parts)
    filtered = filter_and_truncate(text)
    return {"text": filtered} if filtered else None


def load_dataset_samples(config, target_count: int) -> list:
    if config.hf_id == "local":
        if not CUSTOM_DATA_PATH.exists():
            return []
        with open(CUSTOM_DATA_PATH) as f:
            data = [json.loads(line) for line in f]
        return [r for r in (format_custom(d) for d in data) if r]

    formatted = []
    try:
        data = load_dataset(config.hf_id, split="train", streaming=True)
        for i, ex in enumerate(data):
            if i > target_count * 3:
                break
            text = ex.get("text", "")
            if not text:
                continue
            filtered = filter_and_truncate(text)
            if filtered:
                formatted.append({"text": filtered})
    except Exception as e:
        print(f"  Error loading {config.hf_id}: {e}")
    return formatted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_train_data = []
    all_eval_data = []
    total_tokens = 0

    print(f"Target pretrain tokens: {TARGET_PRETRAIN_TOKENS:,}")
    print(f"{'Dataset':<25} {'Mix%':>6} {'Samples':>10}")
    print("-" * 45)

    for config in PRETRAIN_DATASETS:
        target_count = int(TARGET_PRETRAIN_TOKENS * config.mix_percent / 100 / 5)
        samples = load_dataset_samples(config, target_count)
        print(f"  {config.name:<23} {config.mix_percent:>5.0f}% {len(samples):>10,}")

        if not samples:
            print(f"  Skipping: no samples loaded")
            continue

        samples = deduplicate_with_minhash(samples)
        print(f"  After dedup: {len(samples):,}")

        random.shuffle(samples)
        split = int(len(samples) * TRAIN_EVAL_SPLIT)
        all_train_data.extend(samples[:split])
        all_eval_data.extend(samples[split:])

        sample_tokens = sum(len(s["text"].split()) for s in samples[:100])
        estimated_tokens = int(sample_tokens / 100 * len(samples))
        total_tokens += estimated_tokens

    random.shuffle(all_train_data)
    random.shuffle(all_eval_data)

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for item in all_train_data:
            f.write(json.dumps(item) + "\n")
    with open(OUTPUT_DIR / "eval.jsonl", "w") as f:
        for item in all_eval_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n=== Token Count Report ===")
    print(f"Total estimated tokens: {total_tokens:,} / {TARGET_PRETRAIN_TOKENS:,}")
    print(f"Train samples: {len(all_train_data):,}")
    print(f"Eval samples: {len(all_eval_data):,}")
    print(f"Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
