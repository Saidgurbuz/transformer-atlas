// HTTP app: verify the gestures through real 3D ray hits, including responsive layouts.
// tools/smoke.sh --view model --preset tinystories --eval-file tests/browser_token_focus.js
(async () => {
  const { store, atlas } = window.__te;
  const { SceneEngine, THREE } = await import(new URL('src/scene/engine3d.js', location.href).href);
  let checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  const settle = async () => {
    await new Promise(r => setTimeout(r, 150));
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
  };
  const routed = async id => {
    for (let n = 0; n < 80 && atlas.stack().at(-1)?.id !== id; n++) await new Promise(r => setTimeout(r, 50));
    assert(atlas.stack().at(-1)?.id === id, `gesture navigates to ${id}`);
    await settle();
  };
  const root = () => atlas.stack().at(-1).el;
  // Observe the existing engine during an actual canvas event; picking remains unchanged.
  function captureEngine() {
    const canvas = root().querySelector('.scene-canvas'), pick = SceneEngine.prototype.pick;
    let engine;
    try {
      SceneEngine.prototype.pick = function(e) { engine = this; return pick.call(this, e); };
      canvas.dispatchEvent(new MouseEvent('click', { clientX: -10000, clientY: -10000 }));
    } finally { SceneEngine.prototype.pick = pick; }
    assert(!!engine, 'current overview engine is reachable through its canvas');
    return engine;
  }
  let engine = captureEngine();
  function point(token) {
    engine.scene.updateMatrixWorld(true);
    engine.camera.updateMatrixWorld(true);
    const mesh = [...engine.pickables].find(m => m.userData.token === token);
    assert(!!mesh, `input token ${token} is visible`);
    const p = mesh.getWorldPosition(new THREE.Vector3()).project(engine.camera);
    const r = engine.renderer.domElement.getBoundingClientRect();
    const event = { clientX: r.left + (p.x + 1) * r.width / 2, clientY: r.top + (1 - p.y) * r.height / 2, bubbles: true };
    assert(engine.pick(event)?.object.userData.token === token, 'projected point hits the real token card');
    return event;
  }
  const shift = token => engine.renderer.domElement.dispatchEvent(new MouseEvent('click', { ...point(token), shiftKey: true }));
  const context = token => engine.renderer.domElement.dispatchEvent(new MouseEvent('contextmenu', { ...point(token), button: 2 }));
  const action = label => [...document.querySelectorAll('.ctx-item')].find(b => b.querySelector('span')?.textContent === label);
  function prediction() {
    const st = store.state, V = st.config.vocab_size, q = st.sel.token;
    const values = st.trace.probs.data.subarray(q * V, (q + 1) * V);
    let best = 0;
    for (let v = 1; v < V; v++) if (values[v] > values[best]) best = v;
    assert(root().querySelector('.ma-win b')?.textContent === st.tokenizer.label(best), 'prediction uses the focused token’s probability row');
    assert(root().querySelector('.ma-win em')?.textContent === `next after “${st.tokenizer.label(st.ids[q])}”`, 'prediction names its actual context position');
  }
  const last = store.state.ids.length - 1, camera = engine.camera.position.clone();
  shift(0); await settle();
  assert(atlas.stack().at(-1).id === 'model' && store.state.sel.token === 0, 'Shift-click focuses without zooming');
  assert(engine.camera.position.distanceTo(camera) < 1e-6, 'focusing keeps the camera framing');
  prediction();
  shift(last); await settle();
  assert(store.state.sel.token === last, 'Shift-click can return to continuation after the full input');
  prediction();

  context(0); await settle();
  assert(store.state.sel.token === last, 'opening the menu does not change the prediction');
  assert(!!action('Focus token') && !!action('Zoom into tokens'), 'token menu offers focus and zoom separately');
  const menu = document.querySelector('.ctx-menu').getBoundingClientRect();
  assert(menu.left >= 0 && menu.right <= innerWidth + 1 && menu.bottom <= innerHeight + 1, 'token menu fits the viewport');
  action('Focus token').click(); await settle();
  assert(atlas.stack().at(-1).id === 'model' && store.state.sel.token === 0, 'menu focus stays in the overview');
  prediction();

  engine.renderer.domElement.dispatchEvent(new MouseEvent('click', point(last)));
  await routed('tokens');
  assert(atlas.stack().at(-1).id === 'tokens' && store.state.sel.token === last, 'ordinary click keeps token zoom');
  await atlas.go('model'); await settle(); engine = captureEngine();
  context(0); await settle(); action('Zoom into tokens').click();
  await routed('tokens');
  assert(atlas.stack().at(-1).id === 'tokens' && store.state.sel.token === 0, 'menu zoom keeps the clicked token selected');
  await atlas.go('model'); await settle(); engine = captureEngine();
  store.setText('Lily had a red ball. One day, she will'); await settle();
  assert(store.state.sel.token === store.state.ids.length - 1, 'new input resumes continuation after explicit focus');
  prediction();
  context(0); await settle();
  return { checks, token: store.state.sel.token, tokens: store.state.ids.length };
})()
