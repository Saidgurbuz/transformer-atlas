// Verify hosted initialization and scene payloads without sending test visits to Google.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../src/core/site.js', import.meta.url), 'utf8');
function environment(href) {
  const scripts = [], listeners = new Map();
  const window = { location: new URL(href), addEventListener: (name, fn) => listeners.set(name, fn) };
  const document = { getElementById: id => scripts.find(s => s.id === id),
    createElement: () => ({}), head: { append: script => scripts.push(script) } };
  return { window, document, scripts, listeners };
}
for (const url of ['http://localhost:8000/', 'http://127.0.0.1:8000/transformer-atlas/',
  'file:///tmp/transformer-atlas.html', 'https://example.org/transformer-atlas/', 'https://saidgurbuz.github.io/other/']) {
  const env = environment(url);
  runInNewContext(source, env);
  assert.equal(env.scripts.length, 0, url);
  assert.equal(env.window.dataLayer, undefined, url);
}
for (const path of ['/transformer-atlas/', '/transformer-atlas/transformer-atlas.html']) {
  const env = environment(`https://saidgurbuz.github.io${path}?preset=tinystories&text=not-for-measurement#model`);
  runInNewContext(source, env);
  runInNewContext(source, env);
  assert.equal(env.scripts.length, 1, 'one async loader');
  assert.equal(env.scripts[0].async, true);
  assert.equal(env.scripts[0].src, 'https://www.googletagmanager.com/gtag/js?id=G-8J3PCF3R9G');
  const commands = env.window.dataLayer.map(args => Array.from(args));
  assert.equal(commands.length, 2, 'initialization is idempotent');
  assert.equal(commands[0][0], 'js');
  assert.equal(commands[1][0], 'config');
  assert.equal(commands[1][1], 'G-8J3PCF3R9G');
  assert.equal(commands[1][2].allow_ad_personalization_signals, false);
  env.listeners.get('atlas:view')({ detail: { id: 'head.0.1', title: 'Head 1' } });
  const event = env.window.dataLayer.at(-1);
  assert.equal(event[0], 'event');
  assert.equal(event[1], 'atlas_view');
  assert.equal(event[2].view_id, 'head.0.1');
  assert.equal(event[2].page_location, `https://saidgurbuz.github.io${path}#head.0.1`);
  assert.ok(!JSON.stringify(event).includes('not-for-measurement'), 'scene events exclude query values');
  env.listeners.get('atlas:view')({ detail: {} });
  assert.equal(env.window.dataLayer.length, 3, 'invalid events ignored');
}
console.log('Hosted integration verified: local/offline isolation, one loader, configured stream and bounded scene payloads.');
