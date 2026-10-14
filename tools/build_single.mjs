// Bundle the whole app (code, CSS, trained weights) into one HTML file.
//   node tools/build_single.mjs
// Writes dist/transformer-atlas.html (opens from disk, no server needed) and
// dist/artifact.html (same content without the document skeleton, for publishing).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';
import { MODEL_PY } from '../src/generated/model_py.js';
import { REFERENCE_PY } from '../src/generated/reference_py.js';
import { REFERENCE_FILES } from '../src/core/reference-files.js';

const require = createRequire(import.meta.url);
const esbuild = require('./node_modules/esbuild');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
if (MODEL_PY !== readFileSync(join(root, 'reference/model.py'), 'utf8')) throw new Error('Embedded model.py is stale; run node tools/gen_codemap.mjs before building.');
if (JSON.stringify(Object.keys(REFERENCE_PY)) !== JSON.stringify(REFERENCE_FILES) || REFERENCE_FILES.some((file) => REFERENCE_PY[file] !== readFileSync(join(root, 'reference', file), 'utf8'))) throw new Error('Embedded lesson references are stale; run node tools/gen_codemap.mjs.');

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
// KaTeX CSS with its woff2 fonts inlined as data URIs, then the app styles
const katexDir = join(root, 'src/vendor/katex');
const katexCss = readFileSync(join(katexDir, 'katex.min.css'), 'utf8').replace(/url\((fonts\/[^)]+\.woff2)\)/g,
  (_, f) => `url(data:font/woff2;base64,${readFileSync(join(katexDir, f)).toString('base64')})`);
const css = [katexCss, readFileSync(join(root, 'styles/main.css'), 'utf8'), readFileSync(join(root, 'styles/atlas.css'), 'utf8')].join('\n');
const fonts = 'https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Instrument+Sans:wght@400..700&family=JetBrains+Mono:wght@400..700&display=swap';
const siteScript = `<script>${readFileSync(join(root, 'src/core/site.js'), 'utf8').replace(/<\/script/gi, '<\\/script')}</script>`;
const favicon = `<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,${encodeURIComponent(readFileSync(join(root, 'favicon.svg'), 'utf8'))}">`;

const inner = `<title>Transformer Atlas</title>
${favicon}
<link rel="stylesheet" href="${fonts}">
<style>${css}</style>
<div id="app"></div>
<script type="module">${js}</script>`;

const page = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
${siteScript}
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Interactive, line-by-line tour of a decoder-only Transformer.">
</head>
<body>
${inner}
</body>
</html>
`;
mkdirSync(join(root, 'dist', 'artifact'), { recursive: true });
// multi-file variant for hosts that prefer a small page: index.html + app.js + app.css
writeFileSync(join(root, 'dist/artifact/app.js'), result.outputFiles[0].text);
writeFileSync(join(root, 'dist/artifact/app.css'), css);
writeFileSync(join(root, 'dist/artifact/index.html'), `<title>Transformer Atlas</title>
${favicon}
${siteScript}
<link rel="stylesheet" href="${fonts}">
<link rel="stylesheet" href="app.css">
<div id="app"></div>
<script type="module" src="app.js"></script>
`);
writeFileSync(join(root, 'dist/transformer-atlas.html'), page);
writeFileSync(join(root, 'dist/artifact.html'), siteScript + '\n' + inner + '\n');
console.log(`dist/transformer-atlas.html  ${(page.length / 1e6).toFixed(2)} MB`);
