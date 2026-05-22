#!/bin/bash
while true; do
  if ! ps aux | grep -q "[t]rain.py"; then
    if grep -q "Done!" /tmp/train.log 2>/dev/null; then
      echo "Training completed gracefully at $(date)" >> /tmp/crash.log
      exit 0
    fi
    echo "CRASHED at $(date)" >> /tmp/crash.log
    source ~/nyapsys/venv/bin/activate 2>/dev/null || source ~/venv/bin/activate
    cd ~/nyapsys/training
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    nohup python3 train.py \
      --config 3b_moe \
      --data_path data/tokenized/ \
      --output_dir gs://nyapsys-training/checkpoints/ \
      --resume_from_checkpoint latest \
      --learning_rate 3e-4 \
      --warmup_steps 2000 \
      --batch_size 2 \
      --gradient_accumulation_steps 8 \
      --num_workers 0 \
      --max_steps 994000 \
      > /tmp/train.log 2>&1 &
  fi
  sleep 120
done
