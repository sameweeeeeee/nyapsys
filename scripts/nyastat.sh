#!/bin/bash
ZONE="${NYAPSYS_ZONE:-us-east1-d}"
INSTANCE="${NYAPSYS_INSTANCE:-nyapsys-training}"

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --tunnel-through-iap \
  --command="echo '=== Nyapsys Training Status ==='
echo 'Instance: RUNNING'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1
grep -oP '\d+/\d+|loss=\d+\.\d+' /tmp/train.log 2>/dev/null | tail -4 | tr '\n' ' '
echo ''
ckpts=\$(gsutil ls gs://nyapsys-training/checkpoints/step-*.pt 2>/dev/null | wc -l)
echo \"Checkpoints: \$ckpts saved (GCS)\"
ps -o etime= -p \$(pgrep -f 'train.py' 2>/dev/null) 2>/dev/null | tr '\n' ' '
echo ''
echo '---'
tail -1 /tmp/train.log 2>/dev/null" \
  -- -o ConnectTimeout=5 2>&1 | grep -v WARNING | grep -v 'To increase'
