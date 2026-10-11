# 3D explorer: plug-in guide

The explorer (`#explorer`, `src/views/explorer.js`) draws the forward pass as a **tower of
tensor sheets** with Three.js r160 (vendored in `src/vendor/three`).

- **x = token position**, **y = computation order** (input at the bottom), **z = feature**
  (or key position / head band). Every traced tensor of rank ≥ 2 can be a sheet.
- Collapsed layers show `attn.probs`, `resid_mid`, `mlp.act`, `resid_out`. An open layer shows
  every traced tensor of that layer in execution order.
- The cinematic (Play) walks `Object.keys(state.trace)` in order, flies the camera to each
  sheet, reveals it token by token, highlights its line in `reference/model.py` and shows its
  equation (KaTeX) in the caption.

Read first: `src/scene/engine3d.js`, `src/scene/sheet.js`, `src/scene/tower.js`,
`src/views/explorer.js` (plug-in api section), `docs/AGENT_GUIDE.md` (trace keys, store, design
rules), `docs/MATH_GUIDE.md` (math catalogue).

## Writing a plug-in

`src/scene/parts/<name>.js`:

```js
export default {
  id: 'attentionBeams',
  title: 'Attention beams',
  install(api) {
    const group = new api.THREE.Group();
    api.engine.scene.add(group);
    const offSync = api.onSync(() => rebuild());
    const offSel = api.onSelect((sel) => rebuild());
    const offStep = api.onStep(({ key, sheet, duration, playing }) => { /* react to the cinematic */ });
    const offAnim = api.stepAnimation(/\.attn\.probs$/, async (key, sheet, { duration }) => {
      // custom reveal for this step; must end with the sheet fully revealed
      await api.engine.tween({ duration: duration * 0.8, update: (e) => sheet.setReveal(e) });
    });
    const offTool = api.addTool({ label: 'Beams', title: 'Toggle attention beams', run: () => toggle() });
    const offCtx = api.addContextItems((info) => info?.traceKey?.endsWith('.attn.probs') ? [{ label: 'Trace this attention', run: () => {} }] : []);
    return () => { offSync(); offSel(); offStep(); offAnim(); offTool(); offCtx(); group.removeFromParent(); /* dispose geometries & materials */ };
  },
};
```

### api

| member | what |
|---|---|
| `THREE` | the three.js module |
| `engine` | `SceneEngine`: `scene`, `camera`, `controls`, `renderer`, `setRenderFn(fn(scene, camera) | null)`, `onResize(fn(w, h)) -> remove`, `tween({duration, delay, easing, update(e, raw)}) -> Promise(.cancel)`, `onFrame(fn(t)) -> remove` (return false when nothing changed), `flyTo({position, target, duration})`, `framing(box3, dir, padding)`, `addPickable(mesh)`, `removePickable`, `invalidate()`, `isDark` |
| `tower` | `sheets: Map<key, Sheet>`, `layout.items [{key, y, group}]`, `layout.plates [{id, layer, sub, range:[y0,y1], keys}]`, `expanded: Set`, `xOf(t)`, `frontZ()`, `boxOf(keys)`, `boxOfGroup(g)`, `boxAll()`, `T`, `root` (THREE.Group) |
| `Sheet` | `group` (position.y = height), `top` (pickable mesh), `width`, `depth`, `rows`, `bands`, `cols`, `tensor`, `zOf(band, x)`, `setReveal(0..1)`, `setData(tensor, {animate, engine})`, `setOpacity(a)`, `pulse(engine)`, `kind` |
| `state` | live store state (`trace`, `config`, `ids`, `tokenizer`, `sel`, `model`) |
| `store` | `select({token, layer, head})`, `setText`, `setConfig`, `loadPreset`, `notice` |
| `on(event, fn)` | store events, auto-removed when the view unmounts |
| `onSync / onSelect / onStep / addContextItems / stepAnimation / addTool` | hooks, each returns a remover |
| `focusKey(key)`, `focusGroup(g)`, `overview()`, `play()`, `pause()`, `isPlaying()`, `speed()` | explorer controls |
| `themeColor('--c-q')` | `THREE.Color` from a CSS token (re-read on `api.on`… theme: rebuild on `onSync`, which runs on theme change) |
| `label(html, {className, kind, interactive})` | `CSS2DObject` HTML label (`.userData.el` is its element) |
| `ease`, `CELL_X`, `CELL_Z`, `hud`, `stage` | easing functions, sheet cell sizes, overlay elements |

World positions: token t is at `x = tower.xOf(t)`; a sheet's top face is at
`y = sheet.group.position.y + sheet.thickness / 2`; feature column x of band b is at
`z = sheet.zOf(b, x)` (feature 0 at the front, +z, facing the default camera).

## Rules

- **Correct**: every number drawn comes from `state.trace`, `state.model` or the engine; never
  invent values. Use the same colours as the rest of the app (`themeColor('--c-<kind>')`).
- **Smooth**: 60 fps on a laptop. Reuse geometries; prefer `InstancedMesh`, `Points`,
  `LineSegments` with BufferGeometry updates over many meshes; move work into shaders where it
  helps. Never allocate per frame in `onFrame`. Dispose everything you create.
- **Calm**: animations explain one thing at a time, ease in and out, and settle. Respect
  `reducedMotion()` (from `src/ui/index.js`): jump to the end state.
- **Composable**: never edit files other than your own part. Do not change sheets' layout
  positions; draw alongside them. Leave the sheet's reveal at 1 when your step animation ends.
- **Robust**: any config (1-8 layers, GQA, RoPE/ALiBi, post-LN, swiglu, any mask), T from 1 to
  64, theme switches, the view unmounting mid-animation.

## Checking

WebGL runs in headless Chromium through SwiftShader (slow but correct):

```bash
tools/smoke.sh --view explorer --wait 3000 --out shot.png --eval "(async () => { ... ; await new Promise(r => setTimeout(r, 2000)); return 1 })()"
```
Use `window.__te.store` to load presets (`await window.__te.store.loadPreset('induction')`),
select tokens/heads, and click HUD buttons (`document.querySelector('.hud-tools')` contains
the tool buttons; `.hud-cine .btn.primary` is Play). Take several screenshots at different
moments of your animation and look at each.
