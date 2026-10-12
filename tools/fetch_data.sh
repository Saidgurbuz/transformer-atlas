#!/usr/bin/env bash
# Download the training corpora used by the preset trainers.
#   data/alice_raw.txt            tools/train_presets.py      (char presets; public domain, Project Gutenberg)
#   data/tinystories/*.txt        tools/train_bpe_preset.py   (subword preset; TinyStoriesV2-GPT4, CDLA-Sharing-1.0)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/tinystories
curl -sL --max-time 120 -o data/alice_raw.txt https://www.gutenberg.org/cache/epub/11/pg11.txt
echo "data/alice_raw.txt: $(wc -c < data/alice_raw.txt) bytes"
TS=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
curl -sL --max-time 900 -o data/tinystories/valid.txt "$TS/TinyStoriesV2-GPT4-valid.txt"
curl -sL --max-time 1800 -r 0-299999999 -o data/tinystories/train_slice.txt "$TS/TinyStoriesV2-GPT4-train.txt"   # first 300 MB
echo "data/tinystories: $(du -sh data/tinystories | cut -f1)"
