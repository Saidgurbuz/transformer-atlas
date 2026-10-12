// Boot: model + first forward pass, then the full-screen Atlas.
import { store } from './core/store.js';
import { startAtlas } from './atlas/atlas.js';
import { PRESETS } from './presets/index.js';

store.init();
if (PRESETS.length) {
  try { await store.loadPreset(PRESETS[0].id); } catch (e) { console.error('preset load failed', e); }
}
const atlas = startAtlas(document.getElementById('app'));
window.__te = { store, atlas };   // handy for the console and for smoke tests
