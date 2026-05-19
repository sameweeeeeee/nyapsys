#!/bin/bash
cloudflared tunnel login
cloudflared tunnel create nyapsys
TUNNEL_ID=$(cloudflared tunnel list --output json | jq -r '.[] | select(.name=="nyapsys") | .id')
cat > ~/.cloudflared/config.yml <<EOF
tunnel: nyapsys
credentials-file: ~/.cloudflared/$TUNNEL_ID.json
ingress:
  - hostname: api.yourdomain.com
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF
cloudflared tunnel route dns nyapsys api.yourdomain.com
cloudflared service install
echo "Tunnel ready: https://api.yourdomain.com"