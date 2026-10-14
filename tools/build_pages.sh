#!/usr/bin/env bash
# Prepare the same publishable files locally and in GitHub Actions.
#   tools/build_pages.sh            writes dist/pages/
set -euo pipefail
cd "$(dirname "$0")/.."

node tools/gen_codemap.mjs --check
node tools/build_single.mjs

OUTPUT="$PWD/dist/pages"
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"
cp -r index.html favicon.svg styles src "$OUTPUT/"
cp dist/transformer-atlas.html "$OUTPUT/transformer-atlas.html"
touch "$OUTPUT/.nojekyll"
node --input-type=module - "$OUTPUT/deployment.json" "$(git rev-parse HEAD)" <<'JS'
import { writeFileSync } from 'node:fs';
import { VERSION } from './src/version.js';
writeFileSync(process.argv[2], JSON.stringify({ version: VERSION, sourceCommit: process.argv[3] }) + '\n');
JS
echo 'Pages files prepared in dist/pages/'
