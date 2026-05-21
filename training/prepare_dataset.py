import json
import random
from pathlib import Path
from datasets import load_dataset
from dataset_config import PRETRAIN_DATASETS, TARGET_PRETRAIN_TOKENS, TRAIN_EVAL_SPLIT, MIN_TOKENS, MAX_TOKENS

OUTPUT_DIR = Path(__file__).parent / "data" / "tokenized"
CUSTOM_DATA_PATH = Path(__file__).parent / "data" / "custom" / "seeds.jsonl"
random.seed(42)

SYSTEM_MESSAGE = "You are Nyapsys, a self-hosted AI assistant running locally on your Mac. You answer questions accurately, read and analyse files, and understand images. Be concise but thorough. If you are unsure, say so."

BATCH_SIZE = 50000


def simple_tokenize(text: str) -> list:
    return text.split()


def filter_and_truncate(text: str) -> str | None:
    tokens = simple_tokenize(text)
    if len(tokens) < MIN_TOKENS:
        return None
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    return " ".join(tokens)


def stream_dataset(config, target_count: int, train_file, eval_file):
    if config.hf_id == "local":
        if not CUSTOM_DATA_PATH.exists():
            return 0, 0
        with open(CUSTOM_DATA_PATH) as f:
            data = [json.loads(line) for line in f]
        count = 0
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue
            parts = [f"<|system|>\n{SYSTEM_MESSAGE}"]
            for msg in messages:
                parts.append(f"<|{msg['role']}|>\n{msg['content']}")
            text = "\n".join(parts)
            filtered = filter_and_truncate(text)
            if filtered:
                f = train_file if random.random() < TRAIN_EVAL_SPLIT else eval_file
                f.write(json.dumps({"text": filtered}) + "\n")
                count += 1
        return count, 0

    train_count = 0
    eval_count = 0
    batch = []

    try:
        ds = load_dataset(config.hf_id, split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i > target_count * 3:
                break
            text = ex.get("text", "")
            if not text:
                continue
            filtered = filter_and_truncate(text)
            if filtered:
                batch.append({"text": filtered})

                if len(batch) >= BATCH_SIZE:
                    for sample in batch:
                        f = train_file if random.random() < TRAIN_EVAL_SPLIT else eval_file
                        f.write(json.dumps(sample) + "\n")
                        if f is train_file:
                            train_count += 1
                        else:
                            eval_count += 1
                    batch = []

                    if (train_count + eval_count) % 100000 == 0:
                        print(f"    Progress: {train_count + eval_count:,} samples written")

        if batch:
            for sample in batch:
                f = train_file if random.random() < TRAIN_EVAL_SPLIT else eval_file
                f.write(json.dumps(sample) + "\n")
                if f is train_file:
                    train_count += 1
                else:
                    eval_count += 1

    except Exception as e:
        print(f"  Error loading {config.hf_id}: {e}")
    return train_count, eval_count


AVG_TOKENS_PER_SAMPLE = 500


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_train = 0
    total_eval = 0
    total_tokens = 0

    print(f"Target pretrain tokens: {TARGET_PRETRAIN_TOKENS:,}")
    print(f"Estimated avg tokens per sample: {AVG_TOKENS_PER_SAMPLE}")
    print(f"{'Dataset':<25} {'Mix%':>6} {'Target samples':>15} {'Train':>10} {'Eval':>10}")
    print("-" * 70)

    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"

    with open(train_path, "w") as train_file, open(eval_path, "w") as eval_file:
        for config in PRETRAIN_DATASETS:
            target_tokens = int(TARGET_PRETRAIN_TOKENS * config.mix_percent / 100)
            target_count = int(target_tokens / AVG_TOKENS_PER_SAMPLE)
            print(f"  {config.name:<23} {config.mix_percent:>5.0f}% {target_count:>15,}", end="", flush=True)

            tc, ec = stream_dataset(config, target_count, train_file, eval_file)
            total_train += tc
            total_eval += ec
            print(f" {tc:>10,} {ec:>10,}")

            if tc + ec > 0:
                sample_tokens = 0
                with open(train_path) as f:
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        try:
                            sample_tokens += len(json.loads(line)["text"].split())
                        except:
                            pass
                avg_tokens = sample_tokens / max(total_train, 1) * 100
                total_tokens += int(avg_tokens * tc)

    print(f"\n=== Token Count Report ===")
    print(f"Total estimated tokens: {total_tokens:,} / {TARGET_PRETRAIN_TOKENS:,}")
    print(f"Train samples: {total_train:,}")
    print(f"Eval samples: {total_eval:,}")
    print(f"Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
