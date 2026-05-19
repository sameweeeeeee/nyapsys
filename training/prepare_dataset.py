import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_config import DATASET_MIX, TRAIN_EVAL_SPLIT

OUTPUT_DIR = Path(__file__).parent / "data" / "tokenized"
CUSTOM_DATA_PATH = Path(__file__).parent / "data" / "custom.jsonl"
random.seed(42)

CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
SYSTEM_MESSAGE = "You are Nyapsys, a helpful AI assistant. You answer questions accurately, read and analyse files, and understand images. Be concise but thorough. If you are unsure, say so."


def format_instruction(example: dict) -> dict:
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    prompt += CHAT_TEMPLATE.format(role="user", content=example.get("instruction", ""))
    prompt += CHAT_TEMPLATE.format(role="assistant", content=example.get("output", ""))
    return {"text": prompt}


def format_qa(context: str, question: str, answer: str) -> dict:
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    prompt += CHAT_TEMPLATE.format(role="user", content=f"Context: {context}\n\nQuestion: {question}")
    prompt += CHAT_TEMPLATE.format(role="assistant", content=answer)
    return {"text": prompt}


def format_custom(example: dict) -> dict:
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    for msg in example.get("messages", []):
        prompt += CHAT_TEMPLATE.format(role=msg["role"], content=msg["content"])
    return {"text": prompt}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_train_data = []
    all_eval_data = []

    for config in DATASET_MIX:
        if config.hf_id == "local":
            if not CUSTOM_DATA_PATH.exists():
                continue
            with open(CUSTOM_DATA_PATH) as f:
                data = [json.loads(line) for line in f]
            formatted = [format_custom(d) for d in data]
        else:
            try:
                data = load_dataset(config.hf_id, split="train")
                if config.hf_id in ["yahma/alpaca-cleaned", "rajpurkar/squad_v2"]:
                    formatted = [format_instruction(ex) for ex in data[:5000]]
                else:
                    formatted = [{"text": ex.get("text", str(ex))} for ex in data[:10000]]
            except:
                continue

        random.shuffle(formatted)
        split = int(len(formatted) * TRAIN_EVAL_SPLIT)
        all_train_data.extend(formatted[:split])
        all_eval_data.extend(formatted[split:])

    random.shuffle(all_train_data)
    random.shuffle(all_eval_data)

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for item in all_train_data:
            f.write(json.dumps(item) + "\n")
    with open(OUTPUT_DIR / "eval.jsonl", "w") as f:
        for item in all_eval_data:
            f.write(json.dumps(item) + "\n")

    print(f"Train: {len(all_train_data)}, Eval: {len(all_eval_data)}")


if __name__ == "__main__":
    main()