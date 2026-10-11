// Bundle the whole app (code, CSS, trained weights) into one HTML file.
//   node tools/build_single.mjs
// Writes dist/transformer-explorer.html (opens from disk, no server needed) and
// dist/artifact.html (same content without the document skeleton, for publishing).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const esbuild = require('./node_modules/esbuild');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const result = await esbuild.build({
  entryPoints: [join(root, 'src/main.js')],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  minify: true,
  write: false,
  legalComments: 'none',
});
const js = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = readFileSync(join(root, 'styles/main.css'), 'utf8');
const fonts = 'https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Instrument+Sans:wght@400..700&family=JetBrains+Mono:wght@400..700&display=swap';

const inner = `<title>Transformer Explorer</title>
<link rel="stylesheet" href="${fonts}">
<style>${css}</style>
<div id="app"></div>
<script type="module">${js}</script>`;

const page = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Interactive, line-by-line tour of a decoder-only Transformer.">
</head>
<body>
${inner}
</body>
</html>
`;
mkdirSync(join(root, 'dist'), { recursive: true });
writeFileSync(join(root, 'dist/transformer-explorer.html'), page);
writeFileSync(join(root, 'dist/artifact.html'), inner + '\n');
console.log(`dist/transformer-explorer.html  ${(page.length / 1e6).toFixed(2)} MB`);
