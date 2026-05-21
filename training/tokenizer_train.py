from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import argparse


def train_tokenizer(files: list[str], vocab_size: int = 32000, output_dir: str = "training/tokenizer"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>", "<|sep|>"],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output", type=str, default="training/tokenizer")
    args = parser.parse_args()
    train_tokenizer([args.corpus], args.vocab_size, args.output)