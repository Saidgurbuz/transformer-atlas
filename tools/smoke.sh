#!/usr/bin/env bash
# Serialised wrapper: at most two headless browsers at a time on the shared login node.
cd "$(dirname "$0")/.." || exit 1
for i in 1 2; do
  exec 9>"tools/.smoke.$i.lock"
  if flock -n 9; then exec timeout 300 node tools/smoke.mjs "$@"; fi
done
exec 9>"tools/.smoke.1.lock"
flock -w 2400 9 && exec timeout 300 node tools/smoke.mjs "$@"
