// HTTP overview: one native render in either theme, no idle GPU work or growing theme resources.
// tools/smoke.sh --view model --preset tinystories --reduced-motion --dpr 2 --eval-file tests/browser_render_budget.js
(async () => {
  const { SceneEngine, THREE } = await import(new URL('src/scene/engine3d.js', location.href).href);
  const { applyTheme } = await import(new URL('src/core/panels.js', location.href).href);
  const { store, atlas } = window.__te;
  const canvas = atlas.stack().at(-1).el.querySelector('.scene-canvas');
  const pick = SceneEngine.prototype.pick;
  let engine, checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  try {
    SceneEngine.prototype.pick = function(e) { engine = this; return pick.call(this, e); };
    canvas.dispatchEvent(new MouseEvent('click', { clientX: -10000, clientY: -10000 }));
  } finally { SceneEngine.prototype.pick = pick; }
  assert(!!engine, 'overview engine captured through its real canvas');
  engine.pause();
  const R = engine.renderer, render = R.render, resources = {}, budgets = [];
  let calls = 0, offscreen = 0;
  R.render = function(...args) {
    calls++; if (this.getRenderTarget()) offscreen++;
    return render.apply(this, args);
  };
  const frame = () => {
    calls = offscreen = 0;
    engine.invalidate(); engine.tick(performance.now());
    assert(calls === 1 && offscreen === 0, 'one scene render directly to the native framebuffer');
    assert(engine.renderFn === null, 'no full-frame postprocessing pipeline');
    assert(!R.getContext().isContextLost(), 'WebGL context remains healthy');
    return { draws: R.info.render.calls, textures: R.info.memory.textures, geometries: R.info.memory.geometries };
  };
  try {
    for (const theme of ['light', 'dark', 'light', 'dark', 'light', 'dark']) {
      applyTheme(theme);
      const current = frame();
      if (resources[theme]) {
        assert(current.textures === resources[theme].textures, `${theme}: texture budget ${current.textures} versus ${resources[theme].textures}`);
        assert(current.geometries === resources[theme].geometries, 'theme changes retain a stable geometry budget');
      } else resources[theme] = current;
      budgets.push({ theme, ...current });
    }
    const ratio = Math.min(devicePixelRatio, 2), size = R.getSize(new THREE.Vector2());
    assert(canvas.width === Math.floor(size.x * ratio) && canvas.height === Math.floor(size.y * ratio), 'native antialiasing retains the expected high-DPI resolution');
    const camera = engine.camera.position.clone();
    engine.camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.08);
    engine.camera.lookAt(engine.controls.target);
    frame();
    engine.camera.position.copy(camera); engine.camera.lookAt(engine.controls.target); frame();
    const cfg = store.state.config;
    store.setConfig({ n_layer: 8 }); frame();
    assert(store.state.config.n_layer === 8, 'all eight layers use the same bounded rendering path');
    store.setConfig({ n_layer: cfg.n_layer }); frame();
    const originalWidth = engine.container.style.width;
    engine.container.style.width = `${Math.max(320, size.x - 100)}px`;
    engine.resize(); frame();
    assert(R.getSize(new THREE.Vector2()).x === Math.max(320, size.x - 100), 'viewport resize updates the native framebuffer');
    engine.container.style.width = originalWidth; engine.resize(); frame();
    engine.resume();
    await new Promise(r => setTimeout(r, 400));
    calls = 0;
    await new Promise(r => setTimeout(r, 300));
    assert(calls === 0, 'a settled overview issues no further WebGL renders');
    return { checks, dpr: devicePixelRatio, budgets, idleRenders: calls };
  } finally { R.render = render; engine.resume(); }
})()
