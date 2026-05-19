import os
import subprocess
from pathlib import Path

from unsloth import FastVisionModel
from peft import PeftModel
import torch


MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LORA_PATH = Path("./output/nyapsys-lora")
MERGED_PATH = Path("./output/nyapsys-merged")
GGUF_PATH = Path("./output/Nyapsys-11B-Vision.Q4_K_M.gguf")
LLAMA_CPP_PATH = Path("/opt/llama.cpp")

OUTPUT_TYPE = "q4_k_m"


def merge_and_save():
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=False,
    )

    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, str(LORA_PATH))

    print("Merging adapters...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_PATH}...")
    model.save_pretrained(str(MERGED_PATH))
    tokenizer.save_pretrained(str(MERGED_PATH))

    return model, tokenizer


def convert_to_gguf():
    print(f"Converting to GGUF via {LLAMA_CPP_PATH}...")

    if not LLAMA_CPP_PATH.exists():
        raise FileNotFoundError(
            f"llama.cpp not found at {LLAMA_CPP_PATH}. "
            "Run scripts/setup_gpu_droplet.sh first."
        )

    convert_script = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Convert script not found: {convert_script}")

    cmd = [
        "python",
        str(convert_script),
        str(MERGED_PATH),
        "--outfile",
        str(GGUF_PATH),
        "--outtype",
        OUTPUT_TYPE,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.returncode}")

    print(result.stdout)

    if not GGUF_PATH.exists():
        raise FileNotFoundError(f"GGUF file not created: {GGUF_PATH}")

    size_gb = GGUF_PATH.stat().st_size / (1024**3)
    print(f"GGUF file size: {size_gb:.2f} GB")

    if size_gb < 5.0 or size_gb > 7.0:
        print(f"WARNING: Expected ~5.96 GB, got {size_gb:.2f} GB")


def main():
    if GGUF_PATH.exists():
        print(f"GGUF already exists at {GGUF_PATH}, skipping conversion")
    else:
        merge_and_save()
        convert_to_gguf()

    print(f"\nFinal GGUF: {GGUF_PATH}")
    print("Ready for upload to DO Spaces")


if __name__ == "__main__":
    main()