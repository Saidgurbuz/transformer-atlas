// Headless browser check for one or all views.
//   tools/smoke.sh --view attention [--theme dark] [--width 420] [--out shot.png] [--eval "js expr"] [--full]
//   tools/smoke.sh --all            (every view, light + dark, desktop; prints a table)
//   tools/smoke.sh --all --preset tinystories   (any mode: open that model via ?preset=)
//   tools/smoke.sh --all --shots dir --wait 2500 (also save every view's screenshot)
//   tools/smoke.sh --all --shard 2/6             (every 6th view from the 3rd: parallel sweeps)
//   tools/smoke.sh --file dist/transformer-atlas.html --view head   (single-file build via file://)
//   tools/smoke.sh --url https://saidgurbuz.github.io/transformer-atlas/ --view model   (deployed site)
// Prints JSON per view: console errors, page errors, horizontal overflow, stub status.
// Always run through tools/smoke.sh, which allows one browser at a time on the login node.
import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('./node_modules/playwright');
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };

// Chromium keeps its profile and HTTP cache under os.tmpdir(). On the cluster TMPDIR points at
// the shared GPFS filesystem, where every cache write syncs slowly: a page took ~30 s to load
// instead of 0.5 s. Node-local /tmp avoids that.
if (!process.env.SMOKE_KEEP_TMPDIR) process.env.TMPDIR = '/tmp';
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : dflt; };
const flag = (name) => args.includes(`--${name}`);

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
const base = opt('url', null) ?? (opt('file', null) ? `file://${join(ROOT, opt('file'))}` : `http://127.0.0.1:${server.address().port}/index.html`);

const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] });

async function check(view, { theme = 'light', width = 1440, height = 900, out = null, evalJs = null, full = false, wait = 900 } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: theme, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const errors = [], warnings = [];
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error') (/fonts\.(googleapis|gstatic)|ERR_NAME_NOT_RESOLVED|ERR_INTERNET/.test(t) ? warnings : errors).push(t);
  });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('requestfailed', (r) => { if (!/fonts\.(googleapis|gstatic)/.test(r.url())) errors.push(`request failed: ${r.url()}`); });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const q = opt('preset', null) ? `?preset=${opt('preset')}` : '';
  await page.goto(`${base}${q}#${view}`, { waitUntil: 'load' });
  await page.waitForTimeout(wait);
  const info = await page.evaluate(() => ({
    title: document.title,
    overflowX: document.documentElement.scrollWidth > window.innerWidth + 1,
    viewOverflow: (() => { const m = document.querySelector('.main'); return m ? m.scrollWidth > m.clientWidth + 1 : false; })(),
    stub: !!document.querySelector('.placeholder-view, .atlas-layer:not(.hidden) .k-auto'),
    failed: !!document.querySelector('#view-root > .callout.warn, .atlas-error'),
    textLength: (document.querySelector('.atlas-layer:not(.hidden)') || document.querySelector('#view-root'))?.innerText.length ?? 0,
  }));
  let evalResult;
  if (evalJs) {
    try { evalResult = await page.evaluate(`(async () => (${evalJs}))()`); } catch (e) { evalResult = `eval error: ${e.message}`; }
    await page.waitForTimeout(400);
  }
  if (out) await page.screenshot({ path: out, fullPage: full });
  await ctx.close();
  return { view, theme, width, ...info, errors, warnings: warnings.length, evalResult, screenshot: out };
}

try {
  if (flag('all')) {
    const { PATTERNS } = await import('../src/atlas/nodes.js');
    // --shard i/n: every n-th view starting at i, so a sweep can run as n parallel jobs
    const [si, sn] = opt('shard', '0/1').split('/').map(Number);
    const VIEWS = PATTERNS.map((p) => ({ id: p.pattern.replace(':l', '0').replace(':h', '0').replace(':n', '1') })).filter((_, i) => i % sn === si);
    const rows = [];
    const shots = opt('shots', null);    // --shots dir: also save <view>-<theme>.png there
    for (const v of VIEWS) for (const theme of opt('themes', 'light,dark').split(',')) {
      const r = await check(v.id, { theme, wait: Number(opt('wait', 700)), out: shots ? join(shots, `${v.id}-${theme}.png`) : null });
      rows.push(r);
      console.log(`${(r.errors.length || r.failed ? 'FAIL' : r.stub ? 'stub' : 'ok').padEnd(5)} ${v.id.padEnd(11)} ${theme.padEnd(5)} errors=${r.errors.length} overflow=${r.overflowX || r.viewOverflow} ${r.errors.slice(0, 2).join(' | ').slice(0, 200)}`);
    }
    if (flag('mobile')) for (const v of VIEWS) {
      const r = await check(v.id, { width: 400, height: 860, wait: 700 });
      console.log(`${(r.errors.length || r.failed ? 'FAIL' : 'ok').padEnd(5)} ${v.id.padEnd(11)} 400px errors=${r.errors.length} overflow=${r.overflowX}`);
    }
  } else {
    const r = await check(opt('view', 'overview'), {
      theme: opt('theme', 'light'), width: Number(opt('width', 1440)), height: Number(opt('height', 900)),
      out: opt('out', null), evalJs: opt('eval', null), full: flag('full'), wait: Number(opt('wait', 900)),
    });
    console.log(JSON.stringify(r, null, 2));
  }
} finally {
  await browser.close();
  server.close();
}
