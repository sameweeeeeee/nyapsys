import argparse
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_and_export(model_path: str, output_path: str):
    print(f"Loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    adapter_path = Path(model_path) / "adapter_config.json"
    if adapter_path.exists():
        print("Merging LoRA...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    merged_path = Path(output_path).parent / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    return str(merged_path)


def convert_to_gguf(merged_path: str, gguf_path: str, out_type: str = "q4_k_m"):
    llama_cpp_path = Path("/opt/llama.cpp")
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"llama.cpp not found at {llama_cpp_path}")

    cmd = ["python", str(convert_script), merged_path, "--outfile", gguf_path, "--outtype", out_type]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {result.stderr}")
    print(f"GGUF saved: {gguf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--skip_gguf", action="store_true")
    args = parser.parse_args()

    merged = merge_and_export(args.model_path, args.output_path)
    if not args.skip_gguf:
        convert_to_gguf(merged, args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()