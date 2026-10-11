// Boot: build the shell, run the first forward pass, start routing.
import { store } from './core/store.js';
import { buildShell } from './core/shell.js';
import { startRouter } from './core/router.js';
import { PRESETS } from './presets/index.js';

const shell = buildShell(document.getElementById('app'), {});
store.init();
if (PRESETS.length) {
  try { await store.loadPreset(PRESETS[0].id); } catch (e) { console.error('preset load failed', e); }
}
startRouter(shell);
window.__te = { store };   // handy for the console and for smoke tests
