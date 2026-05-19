#!/bin/bash
set -e
echo "=== Nyapsys compute droplet setup ==="

curl -fsSL https://get.docker.com | sh
systemctl enable docker && systemctl start docker
apt-get install -y docker-compose-plugin git curl

VOLUME_DEVICE=/dev/sda
VOLUME_MOUNT=/volumes

if ! blkid $VOLUME_DEVICE > /dev/null 2>&1; then
  echo "Formatting block volume..."
  mkfs.ext4 $VOLUME_DEVICE
fi

mkdir -p $VOLUME_MOUNT
mount $VOLUME_DEVICE $VOLUME_MOUNT
grep -q "$VOLUME_DEVICE" /etc/fstab || \
  echo "$VOLUME_DEVICE $VOLUME_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab

mkdir -p $VOLUME_MOUNT/models $VOLUME_MOUNT/chromadb $VOLUME_MOUNT/sqlite

git clone https://github.com/sameweeeeeee/nyapsys /opt/nyapsys
cd /opt/nyapsys

echo "=== Setup complete ==="
echo "Next: scp .env root@DROPLET_IP:/opt/nyapsys/.env"
echo "Then: bash scripts/download_model.sh"
echo "Then: docker compose up -d"