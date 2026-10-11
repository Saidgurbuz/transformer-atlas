#!/usr/bin/env bash
# Serialised wrapper: at most one headless browser at a time on the shared login node.
cd "$(dirname "$0")/.." || exit 1
exec flock -w 600 tools/.smoke.lock timeout 180 node tools/smoke.mjs "$@"
