// Motion check: capture a sequence of frames of one view and measure how much each frame differs
// from the one before. A still scene must not change at all; anything above zero is flicker,
// shimmer or an animation that never settles. Also writes a contact sheet of the frames.
//   tools/frames.sh --view model --frames 12 --every 250 --out dir [--wait 6000] [--theme dark]
//                   [--preset id] [--query root=b] [--eval "js"] [--move]   (--move: sweep the mouse
//                   across the canvas while capturing, to catch hover flicker)
// Prints JSON: per-step changed-pixel fractions and the largest one.
import http from 'node:http';
import { readFile, stat, mkdir, writeFile } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { inflateSync } from 'node:zlib';

const require = createRequire(import.meta.url);
const { chromium } = require('./node_modules/playwright');
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };
if (!process.env.SMOKE_KEEP_TMPDIR) process.env.TMPDIR = '/tmp';      // see smoke.mjs: GPFS TMPDIR makes Chromium crawl
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : dflt; };
const flag = (name) => args.includes(`--${name}`);

// --- minimal PNG decoder (8-bit RGB/RGBA, non-interlaced: what Chromium screenshots are)
function decodePNG(buf) {
  let p = 8, width = 0, height = 0, type = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p), kind = buf.toString('ascii', p + 4, p + 8), data = buf.subarray(p + 8, p + 8 + len);
    if (kind === 'IHDR') { width = data.readUInt32BE(0); height = data.readUInt32BE(4); type = data[9]; }
    else if (kind === 'IDAT') idat.push(data);
    else if (kind === 'IEND') break;
    p += 12 + len;
  }
  const bpp = type === 6 ? 4 : 3, stride = width * bpp, raw = inflateSync(Buffer.concat(idat)), out = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)], src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), row = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? out[row + x - bpp] : 0, b = y ? out[row - stride + x] : 0, c = x >= bpp && y ? out[row - stride + x - bpp] : 0;
      let v = src[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[row + x] = v & 255;
    }
  }
  return { width, height, bpp, px: out };
}
// fraction of pixels whose largest channel difference exceeds a small threshold, and where they are
function diff(A, B, thr = 12) {
  let n = 0, x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
  for (let y = 0; y < A.height; y++) for (let x = 0; x < A.width; x++) {
    const i = (y * A.width + x) * A.bpp;
    if (Math.max(Math.abs(A.px[i] - B.px[i]), Math.abs(A.px[i + 1] - B.px[i + 1]), Math.abs(A.px[i + 2] - B.px[i + 2])) > thr) {
      n++; if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y;
    }
  }
  return { changed: +(n / (A.width * A.height)).toFixed(5), box: n ? [x0, y0, x1, y1] : null };
}

const server = http.createServer(async (req, res) => {
  try {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let p = normalize(join(ROOT, url));
    if (!p.startsWith(ROOT)) { res.writeHead(403).end(); return; }
    if ((await stat(p)).isDirectory()) p = join(p, 'index.html');
    res.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(await readFile(p));
  } catch { res.writeHead(404).end('not found'); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] });
try {
  const out = opt('out', '/tmp/frames'), N = Number(opt('frames', 12)), every = Number(opt('every', 250));
  await mkdir(out, { recursive: true });
  const width = Number(opt('width', 1440)), height = Number(opt('height', 900));
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: opt('theme', 'light'), deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/fonts\.(googleapis|gstatic)/.test(m.text())) errors.push(m.text()); });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const params = [opt('preset', null) && `preset=${opt('preset')}`, opt('query', null)].filter(Boolean);
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html${params.length ? `?${params.join('&')}` : ''}#${opt('view', 'model')}`, { waitUntil: 'load' });
  await page.waitForTimeout(Number(opt('wait', 6000)));
  if (opt('eval', null)) await page.evaluate(`(async () => (${opt('eval')}))()`);
  const frames = [];
  for (let k = 0; k < N; k++) {
    if (flag('move')) await page.mouse.move(width * (0.3 + 0.4 * (k / Math.max(1, N - 1))), height * (0.35 + 0.3 * Math.sin(k)));
    const png = await page.screenshot();
    await writeFile(join(out, `f${String(k).padStart(2, '0')}.png`), png);
    frames.push(png);
    if (k < N - 1) await page.waitForTimeout(every);
  }
  const dec = frames.map(decodePNG), steps = [];
  for (let k = 1; k < dec.length; k++) steps.push(diff(dec[k - 1], dec[k]));
  // contact sheet: the frames in a grid, each labelled with its time
  const cols = Math.min(4, N), scale = 0.4, tw = Math.round(width * scale), th = Math.round(height * scale);
  const sheet = await ctx.newPage();
  await sheet.setViewportSize({ width: cols * tw, height: Math.ceil(N / cols) * (th + 20) });
  await sheet.setContent(`<body style="margin:0;display:grid;grid-template-columns:repeat(${cols},${tw}px);background:#777;font:600 12px monospace">${frames.map((f, k) => `<div><div style="height:20px;background:#222;color:#fff;padding:2px 6px">t = ${k * every} ms${k ? ` · Δ ${(steps[k - 1].changed * 100).toFixed(2)}%` : ''}</div><img style="width:${tw}px;height:${th}px;display:block" src="data:image/png;base64,${f.toString('base64')}"></div>`).join('')}</body>`);
  await sheet.screenshot({ path: join(out, 'sheet.png') });
  console.log(JSON.stringify({ frames: N, every, maxChanged: Math.max(0, ...steps.map((s) => s.changed)), steps, errors, sheet: join(out, 'sheet.png') }, null, 1));
} finally {
  await browser.close();
  server.close();
}
