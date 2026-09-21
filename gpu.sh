#!/bin/bash
# The GPU lock. Every command that loads a model goes through this, from any
# peer, so two sessions cannot race past a "nvidia-smi looked idle" check.
#
#   ./gpu.sh <name> <cmd...>       run <cmd> holding the lock, log to runs/<name>.log
#   ./gpu.sh --who                 show who holds it
#
# Refuses immediately if someone else holds the lock (no queueing: the holder
# decides when to hand over, in chat). Waits for VRAM to be reclaimed before
# starting, appends EXIT=$? to the log, releases on exit or kill.
set -u
LOCK=runs/.gpu.lock
mkdir -p runs

if [ "${1:-}" = "--who" ]; then
  [ -s "$LOCK" ] && cat "$LOCK" || echo "free"
  exit 0
fi
[ $# -ge 2 ] || { echo "usage: $0 <name> <cmd...> | --who" >&2; exit 2; }

name=$1; shift
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "GPU BUSY — held by: $(cat "$LOCK")" >&2
  exit 75
fi
printf '%s pid=%s peer=%s name=%s\n' "$(date +%H:%M:%S)" "$$" "${PEER_NAME:-?}" "$name" >"$LOCK"
trap ': >"$LOCK"' EXIT

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 1500 ] && break
  [ "$i" -eq 60 ] && { echo "TIMEOUT: VRAM still ${used}MiB after 3 min" >&2; exit 1; }
  sleep 3
done

export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
       VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
log=runs/$name.log
echo "=== $(date -Is) $* ===" >>"$log"
"$@" >>"$log" 2>&1
rc=$?
echo "EXIT=$rc" >>"$log"
exit $rc
