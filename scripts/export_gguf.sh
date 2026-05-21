#!/bin/bash
set -e
echo "=== Nyapsys GGUF export ==="

MODEL_PATH="${1:-./training/instruct-output/final}"
OUTPUT_PATH="${2:-./Nyapsys-2B-MoE.Q4_K_M.gguf}"
OUT_TYPE="${3:-q4_k_m}"

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_PATH"
echo "Type: $OUT_TYPE"

LLAMA_CPP_PATH="${LLAMA_CPP_PATH:-/opt/llama.cpp}"
CONVERT_SCRIPT="$LLAMA_CPP_PATH/convert_hf_to_gguf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
  echo "ERROR: llama.cpp not found at $LLAMA_CPP_PATH"
  echo "Run: git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_PATH && make -C $LLAMA_CPP_PATH -j$(nproc)"
  exit 1
fi

python "$CONVERT_SCRIPT" "$MODEL_PATH" --outfile "$OUTPUT_PATH" --outtype "$OUT_TYPE"

FILE_SIZE=$(stat -f%z "$OUTPUT_PATH" 2>/dev/null || stat -c%s "$OUTPUT_PATH")
if [ "$FILE_SIZE" -lt 1000000000 ]; then
  echo "ERROR: GGUF too small ($FILE_SIZE bytes). Export may have failed."
  exit 1
fi

echo "GGUF size: $(ls -lh "$OUTPUT_PATH" | awk '{print $5}')"
echo "=== Export complete ==="
