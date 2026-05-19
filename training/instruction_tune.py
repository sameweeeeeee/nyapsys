import argparse
from pathlib import Path
import json

import torch
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="training/data/tokenized")
    args = parser.parse_args()

    print(f"Loading {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = "<|pad|>"

    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = load_jsonl(Path(args.data_path) / "train.jsonl")
    train_ds = Dataset.from_list(train_data)

    training_args = SFTConfig(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=4,
                               gradient_accumulation_steps=4, learning_rate=2e-4, fp16=True, logging_steps=10,
                               save_strategy="steps", save_steps=100)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_ds, tokenizer=tokenizer)
    print("Training...")
    trainer.train()

    output_path = Path(args.output_dir) / "final"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()