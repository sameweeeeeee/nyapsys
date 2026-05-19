import json
import random
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from dataset_config import (
    DATASET_MIX,
    TARGET_TRAIN_SAMPLES,
    TRAIN_EVAL_SPLIT,
    CHAT_TEMPLATE,
    SYSTEM_MESSAGE,
    VISION_SYSTEM_INJECT,
    MAX_SEQ_LENGTH,
)

MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = Path(__file__).parent / "data" / "prepared"
CUSTOM_DATA_PATH = Path(__file__).parent / "data" / "custom.jsonl"

random.seed(42)


def format_alpaca(example):
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    prompt += CHAT_TEMPLATE.format(role="user", content=example["instruction"])
    prompt += CHAT_TEMPLATE.format(role="assistant", content=example["output"])
    return {"text": prompt}


def format_squad(example):
    context = example["context"]
    question = example["question"]
    answers = example["answers"]["text"]
    answer = answers[0] if answers else "No answer available"

    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    prompt += CHAT_TEMPLATE.format(
        role="user", content=f"Context: {context}\n\nQuestion: {question}"
    )
    prompt += CHAT_TEMPLATE.format(role="assistant", content=answer)
    return {"text": prompt}


def format_llava(example):
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    prompt += VISION_SYSTEM_INJECT

    if "image" in example:
        prompt += "[Image provided] "
    if "conversations" in example:
        for conv in example["conversations"]:
            if conv["from"] == "human":
                prompt += conv["value"].replace("<image>\n", "")
            else:
                prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                prompt += conv["value"]
                prompt += "<|eot_id|>"
    else:
        prompt += example.get("question", "")

    return {"text": prompt}


def format_lmsys(example):
    conversations = example["conversation"]
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)

    for msg in conversations:
        role = "user" if msg["role"] == "user" else "assistant"
        prompt += CHAT_TEMPLATE.format(role=role, content=msg["content"])

    return {"text": prompt}


def format_custom(example):
    prompt = CHAT_TEMPLATE.format(role="system", content=SYSTEM_MESSAGE)
    for msg in example.get("messages", []):
        prompt += CHAT_TEMPLATE.format(role=msg["role"], content=msg["content"])
    return {"text": prompt}


FORMatters = {
    "yahma/alpaca-cleaned": format_alpaca,
    "rajpurkar/squad_v2": format_squad,
    "HuggingFaceH4/llava-instruct-mix-vsft": format_llava,
    "lmsys/lmsys-chat-1m": format_lmsys,
    "local": format_custom,
}


def load_and_format(dataset_config, tokenizer, target_count: int):
    if dataset_config.hf_id == "local":
        if not CUSTOM_DATA_PATH.exists():
            print(f"Custom data file not found: {CUSTOM_DATA_PATH}")
            return []
        with open(CUSTOM_DATA_PATH) as f:
            data = [json.loads(line) for line in f]
        formatted = [format_custom({"messages": d}) for d in data]
        return formatted[:target_count]

    print(f"Loading {dataset_config.name}...")
    ds = load_dataset(dataset_config.hf_id, split="train")

    formatter = FORMatters.get(dataset_config.hf_id)
    if not formatter:
        print(f"No formatter for {dataset_config.hf_id}, skipping")
        return []

    formatted = [formatter(ex) for ex in ds]

    if dataset_config.hf_id == "lmsys/lmsys-chat-1m":
        formatted = formatted[:5000]

    if dataset_config.hf_id == "HuggingFaceH4/llava-instruct-mix-vsft":
        formatted = formatted[:3000]

    return formatted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_formatted = []

    for config in DATASET_MIX:
        target = int(TARGET_TRAIN_SAMPLES * config.mix_percent / 100)
        formatted = load_and_format(config, tokenizer, target)
        print(f"  {config.name}: {len(formatted)} examples")
        all_formatted.extend(formatted)

    random.shuffle(all_formatted)

    split_idx = int(len(all_formatted) * TRAIN_EVAL_SPLIT)
    train_data = all_formatted[:split_idx]
    eval_data = all_formatted[split_idx:]

    print(f"\nTotal: {len(all_formatted)}")
    print(f"Train: {len(train_data)}")
    print(f"Eval: {len(eval_data)}")

    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(eval_path, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved to {train_path} and {eval_path}")


if __name__ == "__main__":
    main()