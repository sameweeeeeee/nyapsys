import json
from pathlib import Path
from typing import Optional

from unsloth import FastVisionModel
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset


MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = Path("./output/nyapsys-lora")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DATASET_PATH = Path("./data/prepared/")

MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CHECKPOINT_INTERVAL = 200
LOG_INTERVAL = 10


def load_dataset() -> tuple[Dataset, Dataset]:
    train_path = DATASET_PATH / "train.jsonl"
    eval_path = DATASET_PATH / "eval.jsonl"

    def load_jsonl(path: Path) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    print(f"Loaded {len(train_ds)} train, {len(eval_ds)} eval samples")
    return train_ds, eval_ds


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        lora_config=LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    train_ds, eval_ds = load_dataset()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOG_INTERVAL,
        save_strategy="steps",
        save_steps=CHECKPOINT_INTERVAL,
        eval_strategy="steps",
        eval_steps=CHECKPOINT_INTERVAL,
        per_device_eval_batch_size=BATCH_SIZE,
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Training complete!")


if __name__ == "__main__":
    main()