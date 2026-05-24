import os
import argparse
import subprocess
import random
import numpy as np
from pathlib import Path
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup, PreTrainedModel, PretrainedConfig
from torch.optim import Adafactor
from tqdm import tqdm

from model_config import get_config

torch.set_float32_matmul_precision('high')


class NyapsysMoEConfig(PretrainedConfig):
    model_type = "nyapsys_moe"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class NyapsysMoEForCausalLM(PreTrainedModel):
    config_class = NyapsysMoEConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = MoEModel(config)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits, router_logits = self.model(input_ids, attention_mask, use_checkpoint=False)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"logits": logits, "loss": loss, "router_logits": router_logits[-1] if router_logits else None}

    def generate(self, input_ids, max_new_tokens=256, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(generated)
            next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id:
                break
        return generated


def save_as_hf(model, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hf_config = NyapsysMoEConfig(**config, eos_token_id=2, bos_token_id=1, pad_token_id=0, vocab_size=config["vocab_size"], hidden_size=config["hidden_size"])
    hf_model = NyapsysMoEForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict())
    hf_model.save_pretrained(output_dir)
    hf_config.save_pretrained(output_dir)
    print(f"HuggingFace model saved to {output_dir}")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated


class MoEModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.rope = RotaryEmbedding(
            dim=config["hidden_size"] // config["num_attention_heads"],
            max_seq_len=config["max_position_embeddings"],
            theta=config["rope_theta"],
        )

    def forward(self, input_ids, attention_mask=None, use_checkpoint: bool = False):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        cos, sin = self.rope(x, seq_len)

        all_router_logits = []
        for layer in self.layers:
            if use_checkpoint:
                x, router_logits = checkpoint(layer, x, cos, sin, attention_mask, use_reentrant=False)
            else:
                x, router_logits = layer(x, cos, sin, attention_mask)
            all_router_logits.append(router_logits)

        x = self.norm(x)
        return self.lm_head(x), all_router_logits


class MoELayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm1 = nn.RMSNorm(hidden_size, eps=config["rms_norm_eps"])
        self.norm2 = nn.RMSNorm(hidden_size, eps=config["rms_norm_eps"])

        self.num_experts = config["num_experts"]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, config["expert_intermediate_size"]),
                nn.GELU(),
                nn.Linear(config["expert_intermediate_size"], hidden_size),
            ) for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(hidden_size, self.num_experts)
        self.num_experts_per_token = config["num_experts_per_token"]

    def _apply_attention(self, x, cos, sin, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_out)

    def forward(self, x, cos, sin, attention_mask=None):
        attn_out = self._apply_attention(self.norm1(x), cos, sin, attention_mask)
        x = x + attn_out

        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)

        expert_out = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            if mask.any():
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                probs = topk_probs[mask].unsqueeze(-1)
                expert_out[token_indices] += expert_output * probs

        x = x + self.norm2(expert_out.view(batch_size, seq_len, hidden_size))
        return x, router_logits


def compute_loss(logits, labels, router_logits, num_experts, router_aux_loss_coef):
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    router_probs = F.softmax(router_logits, dim=-1)
    expert_usage = router_probs.mean(dim=0)
    target_usage = torch.ones_like(expert_usage) / num_experts
    aux_loss = F.mse_loss(expert_usage, target_usage)
    return ce_loss + router_aux_loss_coef * aux_loss


def log_router_stats(router_logits):
    probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    return f"Expert load: {[f'{p:.2%}' for p in probs.tolist()]}"


def load_dataset(path: Path, max_length: int = 4096, limit: int = None):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue
            tokens = text.split()[:max_length]
            if tokens:
                data.append(tokens)
    return data


def pre_tokenize_dataset(path: Path, max_length: int = 4096, vocab_size: int = 32000, limit: int = None):
    """Pre-tokenize dataset using chunked processing to avoid OOM."""
    cache_dir = path.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    tokens_path = cache_dir / "tokens.npy"
    lengths_path = cache_dir / "lengths.npy"
    
    if tokens_path.exists() and lengths_path.exists():
        print(f"Loading cached tokens from {cache_dir}")
        return np.memmap(tokens_path, dtype=np.int32, mode='r'), np.load(lengths_path)
    
    print(f"Pre-tokenizing {path} (limit={limit})...")
    CHUNK = 50000
    chunk_files = []
    lengths = []
    count = 0
    
    with open(path) as f:
        while True:
            chunk_ids = []
            for _ in range(CHUNK):
                line = f.readline()
                if not line:
                    break
                if limit and count >= limit:
                    break
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue
                tokens = text.split()[:max_length]
                if not tokens:
                    continue
                ids = [abs(hash(t)) % vocab_size for t in tokens]
                chunk_ids.append(np.array(ids, dtype=np.int32))
                lengths.append(len(ids))
                count += 1
            
            if not chunk_ids:
                break
            
            chunk_flat = np.concatenate(chunk_ids)
            cf = cache_dir / f"chunk_{count:07d}.npy"
            np.save(cf, chunk_flat)
            chunk_files.append(cf)
            print(f"  Tokenized {count:,} samples...", flush=True)
            
            if limit and count >= limit:
                break
    
    print(f"  Merging {len(chunk_files)} chunks into final arrays...")
    lengths_arr = np.array(lengths, dtype=np.int32)
    total_tokens = lengths_arr.sum()
    
    mm = np.memmap(tokens_path, dtype=np.int32, mode='write', shape=(total_tokens,))
    offset = 0
    for cf in chunk_files:
        data = np.load(cf)
        n = len(data)
        mm[offset:offset + n] = data
        offset += n
        cf.unlink()
    mm.flush()
    del mm
    
    np.save(lengths_path, lengths_arr)
    print(f"  Saved {total_tokens * 4 / 1e9:.1f}GB tokens, {len(lengths):,} sequences")
    
    return np.memmap(tokens_path, dtype=np.int32, mode='r'), np.load(lengths_path)


class MMapDataset:
    """Memory-mapped dataset for fast random access without loading into RAM."""
    def __init__(self, tokens: np.ndarray, lengths: np.ndarray):
        self.tokens = tokens
        self.lengths = lengths
        self.count = len(lengths)
        self.offsets = np.cumsum(lengths) - lengths
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        start = self.offsets[idx]
        length = self.lengths[idx]
        return torch.from_numpy(self.tokens[start:start + length].copy())


def collate_fn(batch):
    """Collate function for DataLoader - handles variable length sequences."""
    # Convert numpy arrays to tensors if needed
    tensors = [torch.tensor(x, dtype=torch.long) if isinstance(x, np.ndarray) else x for x in batch]
    max_len = max(len(x) for x in tensors)
    padded = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in tensors])
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="2b_moe")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = get_config(args.config)
    if args.smoke_test:
        from model_config import CONFIG_3B_MOE_SMOKE, CONFIG_2B_MOE_SMOKE
        config = CONFIG_3B_MOE_SMOKE if "3b" in args.config else CONFIG_2B_MOE_SMOKE
    print(f"Training {args.config}: {config}")

    train_path = Path(args.data_path) / "train.jsonl"
    if args.smoke_test:
        train_data = load_dataset(train_path, config["max_position_embeddings"], limit=10000)
    else:
        tokens, lengths = pre_tokenize_dataset(train_path, config["max_position_embeddings"], config["vocab_size"], limit=args.dataset_limit)
        train_data = MMapDataset(tokens, lengths)
    print(f"Train: {len(train_data)} samples")

    model = MoEModel(config)
    total_batches = len(train_data) // args.batch_size
    total_optim_steps = total_batches // args.gradient_accumulation_steps
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "latest":
            local_dir = args.output_dir if not args.output_dir.startswith("gs://") else "./checkpoints"
            ckpt = None
            if os.path.isdir(local_dir):
                local_ckpts = sorted(Path(local_dir).glob("step-*.pt"),
                                   key=lambda p: int(p.stem.split("-")[1]))
                if local_ckpts:
                    ckpt = str(local_ckpts[-1])
                    print(f"Found local checkpoint: {ckpt}")
            if not ckpt and args.output_dir.startswith("gs://"):
                result = subprocess.run(
                    ["gsutil", "ls", f"{args.output_dir}step-*.pt"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    ckpt = sorted(result.stdout.strip().split())[-1]
            if ckpt:
                args.resume_from_checkpoint = ckpt
            else:
                print("No checkpoints found, starting fresh")
                args.resume_from_checkpoint = None
        if args.resume_from_checkpoint:
            print(f"Resuming from {args.resume_from_checkpoint}")
            ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
            state_dict = ckpt["model"]
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            loaded_step = ckpt.get("step", 0)
            if loaded_step > total_optim_steps:
                loaded_step = loaded_step // args.gradient_accumulation_steps
                print(f"  (converted from batch-counted step)")
            args._checkpoint_step = loaded_step
            args._checkpoint_data = ckpt
            print(f"Loaded checkpoint from optimizer step {args._checkpoint_step}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    optimizer = Adafactor(model.parameters(), lr=args.learning_rate)
    total_steps = args.max_steps if args.max_steps else total_optim_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    if hasattr(args, "_checkpoint_data") and "scheduler" in args._checkpoint_data:
        try:
            scheduler.load_state_dict(args._checkpoint_data["scheduler"])
            print(f"Restored scheduler (last_epoch={scheduler.last_epoch}, up to {total_steps} total)")
        except Exception as e:
            print(f"Warning: could not restore scheduler: {e}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    optim_step = getattr(args, "_checkpoint_step", 0)
    batch_count = optim_step * args.gradient_accumulation_steps
    if optim_step > 0:
        print(f"Resuming training from optimizer step {optim_step} (batch {batch_count})")
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    done = False

    for batch in pbar:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, router_logits = model(batch[:, :-1], use_checkpoint=True)
            loss = compute_loss(logits, batch[:, 1:], router_logits[-1], config["num_experts"], config["router_aux_loss_coef"])
            loss = loss / args.gradient_accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write(f"[Optim step {optim_step}] WARNING: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        batch_count += 1

        if batch_count % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            optim_step += 1

            if optim_step % 100 == 0:
                actual_loss = loss.item() * args.gradient_accumulation_steps
                expert_str = log_router_stats(router_logits[-1])
                pbar.set_postfix({"loss": f"{actual_loss:.4f}", "experts": expert_str})
                tqdm.write(f"[Step {optim_step}] loss={actual_loss:.4f} | {expert_str}")
                try:
                    with open("/tmp/expert_stats.json", "w") as ef:
                        ef.write(json.dumps({"step": optim_step, "expert_str": expert_str}) + "\n")
                except Exception:
                    pass

            if optim_step % 2000 == 0 and not args.smoke_test:
                ckpt_dir = args.output_dir if not args.output_dir.startswith("gs://") else "./checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"step-{optim_step}.pt")
                unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model": unwrapped.state_dict(), "config": config, "step": optim_step,
                            "scheduler": scheduler.state_dict(), "total_steps": total_steps}, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                if args.output_dir.startswith("gs://"):
                    gcs_path = f"{args.output_dir}step-{optim_step}.pt"
                    result = subprocess.run(["gsutil", "cp", ckpt_path, gcs_path], capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        print(f"Uploaded to {gcs_path}")
                    else:
                        print(f"WARNING: GCS upload failed: {result.stderr}")

            if args.max_steps and optim_step >= args.max_steps:
                done = True
                break
        if done:
            break

    ckpt_dir = args.output_dir if not args.output_dir.startswith("gs://") else "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": config}, final_path)
    print(f"Saved final model: {final_path}")
    if args.output_dir.startswith("gs://"):
        gcs_path = f"{args.output_dir}final.pt"
        result = subprocess.run(["gsutil", "cp", final_path, gcs_path], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"Uploaded to {gcs_path}")
    print("Done!")

    if not args.smoke_test:
        print("\nConverting to HuggingFace format for instruction tuning...")
        hf_dir = "/tmp/hf_model"
        save_as_hf(model, config, hf_dir)
        if args.output_dir.startswith("gs://"):
            subprocess.run(["gsutil", "-m", "cp", "-r", hf_dir, f"{args.output_dir}hf_model"], capture_output=True)
        else:
            subprocess.run(["cp", "-r", hf_dir, os.path.join(args.output_dir, "hf_model")])
        print("HuggingFace model saved — ready for instruction_tune.py")


if __name__ == "__main__":
    main()
