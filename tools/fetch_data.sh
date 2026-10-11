#!/usr/bin/env bash
# Download the public-domain training corpus used by tools/train_presets.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -sL --max-time 120 -o data/alice_raw.txt https://www.gutenberg.org/cache/epub/11/pg11.txt
echo "data/alice_raw.txt: $(wc -c < data/alice_raw.txt) bytes"
