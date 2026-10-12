#!/usr/bin/env bash
# Frame capture and flicker check (tools/frames.mjs). On a cluster login node every run is dispatched to the
# LSF interactive queue (about 2 s to start), because headless Chromium with software WebGL is
# CPU-heavy and login nodes kill heavy processes. Elsewhere it runs locally, at most four at a time.
#   SMOKE_LOCAL=1 tools/frames.sh ...   forces a local run
cd "$(dirname "$0")/.." || exit 1
if [[ -z "$SMOKE_LOCAL" && "$(hostname)" == login* ]] && command -v bsub >/dev/null; then
  # drop LSF's banner lines so stdout stays the JSON report
  bsub -I -q interactive -G grp_vision_docling -n 4 -R "rusage[mem=8GB] span[hosts=1]" -W 40 \
    env SMOKE_LOCAL=1 timeout 2300 node tools/frames.mjs "$@" 2>&1 | sed -u '/^Job <[0-9]*> is submitted/d; /^<<.*>>$/d'
  exit "${PIPESTATUS[0]}"
fi
for i in 1 2 3 4; do
  exec 9>"tools/.frames.$i.lock"
  if flock -n 9; then exec timeout 300 node tools/frames.mjs "$@"; fi
done
exec 9>"tools/.frames.1.lock"
flock -w 2400 9 && exec timeout 300 node tools/frames.mjs "$@"
