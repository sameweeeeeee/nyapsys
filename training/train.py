import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from model_config import get_config


class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.lm_head(x)


class TransformerLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, config["intermediate_size"]),
            nn.GELU(),
            nn.Linear(config["intermediate_size"], hidden_size),
        )
        self.norm1 = nn.RMSNorm(hidden_size, eps=config["rms_norm_eps"])
        self.norm2 = nn.RMSNorm(hidden_size, eps=config["rms_norm_eps"])

    def forward(self, x, attention_mask=None):
        attn_out, _, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


def load_dataset(path: Path, max_length: int = 4096):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            tokens = list(map(int, text.split()))[:max_length]
            if tokens:
                data.append(torch.tensor(tokens, dtype=torch.long))
    return data


def collate(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch])
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="1b")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Training {args.config}: {config}")

    train_path = Path(args.data_path) / "train.jsonl"
    eval_path = Path(args.data_path) / "eval.jsonl"

    train_data = load_dataset(train_path, config["max_position_embeddings"])
    eval_data = load_dataset(eval_path, config["max_position_embeddings"])
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    model = Transformer(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_data) * args.max_epochs // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, config["vocab_size"]), batch[:, 1:].reshape(-1), ignore_index=0)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        ckpt_path = f"{args.output_dir}/checkpoint-{epoch + 1}.pt"
        torch.save({"model": model.state_dict(), "config": config}, ckpt_path)
        print(f"Saved: {ckpt_path}")

    torch.save({"model": model.state_dict(), "config": config}, f"{args.output_dir}/final.pt")
    print("Done!")


if __name__ == "__main__":
    main()