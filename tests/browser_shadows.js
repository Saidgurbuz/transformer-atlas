// Run without reduced motion: tools/smoke.sh --view model --preset tinystories --wait 5000 --eval-file tests/browser_shadows.js
(async () => {
  const { SceneEngine, THREE } = await import(new URL('src/scene/engine3d.js', location.href).href);
  const { applyTheme } = await import(new URL('src/core/panels.js', location.href).href);
  const { atlas } = window.__te;
  const host = atlas.stack().at(-1).el.querySelector('.ma'), canvas = host.querySelector('.scene-canvas');
  const pick = SceneEngine.prototype.pick;
  let engine, checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  try {
    SceneEngine.prototype.pick = function(e) { engine = this; return pick.call(this, e); };
    canvas.dispatchEvent(new MouseEvent('click', { clientX: -10000, clientY: -10000 }));
  } finally { SceneEngine.prototype.pick = pick; }
  engine.pause();
  const R = engine.renderer, gl = R.getContext(), results = [];
  // Renderer.info resets after the shadow pass, so count actual WebGL draws.
  const drawMethods = ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced'];
  const originals = new Map();
  let gpuDraws = 0;
  for (const name of drawMethods) if (typeof gl[name] === 'function') {
    originals.set(name, gl[name]);
    gl[name] = function(...args) { gpuDraws++; return originals.get(name).apply(gl, args); };
  }
  const frame = () => { gpuDraws = 0; engine.invalidate(); engine.tick(performance.now()); return gpuDraws; };
  const pixels = () => {
    const data = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    assert(gl.getError() === gl.NO_ERROR, 'framebuffer readback succeeds');
    return data;
  };
  try {
    assert(R.shadowMap.enabled && !R.shadowMap.autoUpdate, 'shadows remain enabled with explicit updates');
    for (const theme of ['light', 'dark']) {
      applyTheme(theme);
      assert(R.shadowMap.needsUpdate, 'theme/geometry rebuild refreshes shadows');
      const initial = frame();
      assert(!R.shadowMap.needsUpdate, 'shadow update is consumed by rendering');
      const cached = frame(), image = pixels();
      R.shadowMap.needsUpdate = true;
      const forced = frame(), freshImage = pixels();
      assert(forced > cached && initial > cached, 'unchanged geometry skips the depth-map draw work');
      assert(image.every((v, i) => v === freshImage[i]), 'cached and freshly rendered shadows produce identical pixels');
      const original = engine.camera.position.clone();
      engine.camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.08);
      engine.camera.lookAt(engine.controls.target);
      const moved = frame();
      assert(moved < forced, 'camera movement reuses the existing depth map');
      R.shadowMap.needsUpdate = true;
      const movedFresh = frame();
      assert(movedFresh > moved, 'forced shadow rendering adds work in the same camera position');
      engine.camera.position.copy(original); engine.camera.lookAt(engine.controls.target); frame();
      for (const ms of [0, 1200, 2800, 4000]) {
        host.__scrub(ms);
        assert(R.shadowMap.needsUpdate, 'animated pins/bars invalidate their shadows');
        assert(frame() > cached, 'animation renders the changed shadow geometry');
        assert(!R.shadowMap.needsUpdate, 'animation shadow updates complete');
      }
      // Scrub intentionally keeps a frame frozen. Finish a real user-triggered
      // pass to verify that ordinary animation returns to cached rendering.
      const play = host.querySelector('.ma-cap button');
      assert(!!play, 'forward-pass control remains available');
      play.click();
      await new Promise(r => setTimeout(r, 4050));
      frame();
      assert(frame() < forced, 'completed animation returns to cached shadows');
      results.push({ theme, drawsWithShadowUpdate: forced, drawsWithCachedShadows: cached });
    }
    assert(!gl.isContextLost(), 'context stays healthy');
    return { checks, results };
  } finally {
    for (const [name, fn] of originals) gl[name] = fn;
    engine.resume();
  }
})()
