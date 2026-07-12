#!/usr/bin/env bash
# serve_fleet.sh — Tier-4 TRUE multi-GPU scale-out: N Ollama daemons, one pinned
# per GPU, each on its own port (11434..11434+N-1). Clients (and the vault
# binary's query-embed) hit the per-GPU ports directly, so the nginx round-robin
# LB is OPTIONAL (off by default — set FLEET_LB=1 to enable it on :8080).
#
# Safe to launch over ssh (`ssh box "bash serve_fleet.sh"`): the daemons are
# started fully detached (new session via setsid, stdin from /dev/null, all fds
# to log files), so they never hold the ssh channel open and the command returns
# as soon as the daemons are warmed. (#620)
set -euo pipefail
PFS="${PFS:-/lambda/nfs/perseus-vault-fs-south}"
export OLLAMA_MODELS="$PFS/models/ollama"
NGPU="${NGPU:-8}"
BASE_PORT=11434
LOG="$PFS/logs/fleet"
FLEET_LB="${FLEET_LB:-0}"          # 1 = also bring up the nginx LB on :8080
LB_PORT="${LB_PORT:-8080}"
mkdir -p "$LOG"

echo "==> stopping any systemd ollama (we run our own pinned daemons)"
sudo systemctl stop ollama 2>/dev/null || true
pkill -f 'ollama serve' 2>/dev/null || true
sleep 2

echo "==> launching $NGPU pinned Ollama daemons (one per GPU), fully detached"
UPSTREAMS=""
for i in $(seq 0 $((NGPU-1))); do
  port=$((BASE_PORT + i))
  # setsid + </dev/null + fds->log detaches the daemon from this (possibly ssh)
  # session so `ssh box "bash serve_fleet.sh"` returns cleanly once warmed,
  # instead of blocking on the never-exiting `ollama serve` holding the channel.
  setsid env CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=60m \
    ollama serve </dev/null >"$LOG/ollama_gpu$i.log" 2>&1 &
  UPSTREAMS="$UPSTREAMS    server 127.0.0.1:$port;\n"
  echo "   GPU $i -> 127.0.0.1:$port"
done

echo "==> waiting for daemons + warming nomic-embed-text on each GPU"
WARM_PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  port=$((BASE_PORT + i))
  for t in $(seq 1 30); do curl -sf http://127.0.0.1:$port/api/tags >/dev/null 2>&1 && break; sleep 2; done
  # warm the embed model into THIS gpu's daemon
  curl -s http://127.0.0.1:$port/api/embed -d '{"model":"nomic-embed-text","input":"warm"}' >/dev/null &
  WARM_PIDS+=($!)
done
# Wait ONLY for the warm curls -- a bare `wait` also blocks on the detached
# `ollama serve` daemons above, which never exit, hanging this script forever.
wait "${WARM_PIDS[@]}"
echo "   all daemons warmed"

if [ "$FLEET_LB" != "1" ]; then
  echo
  echo "FLEET LIVE: $NGPU daemons (LB disabled — clients hit per-GPU ports directly)"
  echo "  per-GPU direct: 127.0.0.1:$BASE_PORT .. 127.0.0.1:$((BASE_PORT+NGPU-1))"
  echo "  (set FLEET_LB=1 to also bring up the nginx round-robin LB on :$LB_PORT)"
  exit 0
fi

echo "==> FLEET_LB=1: installing + configuring nginx round-robin LB on :$LB_PORT"
# nginx is optional and only reached here when explicitly enabled, so a slow/
# stuck apt lock can never block daemon availability for the default path. Bound
# the install so a held dpkg lock fails fast instead of wedging under pipefail.
if ! command -v nginx >/dev/null; then
  if ! sudo timeout 180 apt-get install -y -qq nginx >/dev/null 2>&1; then
    echo "   WARNING: nginx install failed/timed out — daemons are up on the" >&2
    echo "   per-GPU ports; continuing without the LB." >&2
    echo
    echo "FLEET LIVE (no LB): 127.0.0.1:$BASE_PORT .. 127.0.0.1:$((BASE_PORT+NGPU-1))"
    exit 0
  fi
fi
sudo tee /etc/nginx/conf.d/ollama_lb.conf >/dev/null <<EOF
upstream ollama_fleet {
$(echo -e "$UPSTREAMS")
    keepalive 32;
}
server {
    listen $LB_PORT;
    location / {
        proxy_pass http://ollama_fleet;
        # Ollama yields an empty body through the proxy unless we speak HTTP/1.1
        # with keep-alive cleared and buffering off; without these the LB 200s on
        # /api/tags but returns empty /api/embed responses.
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host 127.0.0.1;
        proxy_buffering off;
        proxy_connect_timeout 5s;
        proxy_read_timeout 300s;
    }
}
EOF
# drop the default server that also listens on 80 (avoid conflicts), reload
sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo nginx -t && sudo systemctl restart nginx
sleep 2

echo "==> verify balancer"
curl -s http://127.0.0.1:$LB_PORT/api/embed -d '{"model":"nomic-embed-text","input":"lb test"}' \
  | python3 -c "import sys,json; print('LB embed dim:', len(json.load(sys.stdin)['embeddings'][0]))"
echo
echo "FLEET LIVE: $NGPU daemons behind nginx at http://localhost:$LB_PORT"
echo "  per-GPU direct: 127.0.0.1:$BASE_PORT .. 127.0.0.1:$((BASE_PORT+NGPU-1))"
