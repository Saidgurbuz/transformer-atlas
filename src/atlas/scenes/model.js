// Root scene: the decoder-only Transformer as a calm, jewel-like 3D stack. (Chosen over an
// alternative, scenes/model-b.js, still previewable with ?root=b, in a blind design review.)
//
// Every layer is present as a pair: attention (left, lower) and the MLP (right, higher: it runs
// second), on a thin glass plate. The residual stream is the luminous spine between them; each
// sub-layer hangs off it by one connector and adds back at a ⊕ (x + f(x)). Live data, drawn as
// geometry rather than busy textures: each head is a small screen with its real attention pattern,
// the MLP's hidden units stand as a field of pins whose heights are their real activations, and
// the LM head carries the real next-token distribution. Nothing moves on its own except one calm
// forward pass (a light travelling up the stream); hover only changes colour.
import { SceneEngine, THREE, label } from '../../scene/engine3d.js';
import { RoundedBoxGeometry } from '../../vendor/three/RoundedBoxGeometry.js';
import { RoomEnvironment } from '../../vendor/three/RoomEnvironment.js';
import { EffectComposer } from '../../vendor/three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from '../../vendor/three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '../../vendor/three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '../../vendor/three/addons/postprocessing/OutputPass.js';
import { Line2 } from '../../vendor/three/addons/lines/Line2.js';
import { LineMaterial } from '../../vendor/three/addons/lines/LineMaterial.js';
import { LineGeometry } from '../../vendor/three/addons/lines/LineGeometry.js';
import { h, reducedMotion, cssVar, openContextMenu } from '../../ui/index.js';
import { useStyle } from '../kit.js';

// ------------------------------------------------------------------ geometry (world units)
const MW = 4.4, MD = 2.75, MH = 0.5, MR = 0.17;   // a module body (the MLP; attention widens with its heads)
const GAP = 1.05;                                  // from the stream to each module: room for the arrows into the ⊕
const PD = MD + 0.55, PT = 0.09;                   // layer plate depth and thickness (its width follows the modules)
const HEAD_W = 1.55;                               // world width per attention head: each pattern stays readable
const RISE = 0.92;                                 // the MLP sits this much higher: it runs second
const PITCH = 2.95;                                // layer to layer
const SPINE_R = 0.15;                              // the residual stream: the hero of the figure
const TOPK_UNITS = 28;                             // the MLP shows its most active units, not all of them
// body colours: the saturated (light-theme) palette in both themes, so dark mode never turns pastel
const BODY = { embed: '#0e8f86', pos: '#2b8bc9', attn: '#8250df', mlp: '#c98a06', norm: '#6b7a90', resid: '#4b54d6', head: '#d4435f' };
const FOV = 27;
const VIEW = new THREE.Vector3(Math.sin(0.4), 0.3, Math.cos(0.4)).normalize();   // a calm three-quarter view

useStyle('model-a', `
.ma { position: absolute; inset: 0; }
.ma-lab { display: grid; justify-items: end; gap: 2px; white-space: nowrap; pointer-events: none; text-align: right; }
.ma-lab b { font: 650 13.5px/1.15 var(--font-display); color: var(--fg); letter-spacing: -.005em; }
.ma-lab span { font: 500 11px/1.2 var(--font-mono); color: var(--fg-3); }
.ma-lab i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--kind); margin-right: 7px; vertical-align: 1px; }
.ma-lab.sel b { color: var(--accent); }
.ma-hover { pointer-events: none; padding: 7px 11px; border-radius: 9px; background: color-mix(in srgb, var(--surface) 92%, transparent); border: 1px solid var(--line); box-shadow: 0 8px 24px rgba(0,0,0,.12); white-space: nowrap; opacity: 0; transition: opacity .16s; }
.ma-hover.on { opacity: 1; }
.ma-hover b { display: block; font: 650 13px/1.2 var(--font-body); color: var(--fg); }
.ma-hover span { font: 500 11px/1.3 var(--font-mono); color: var(--fg-3); }
.ma-next { pointer-events: none; display: grid; gap: 7px; padding: 12px 14px 13px; min-width: 190px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 88%, transparent); border: 1px solid var(--line); box-shadow: 0 12px 34px rgba(0,0,0,.10); opacity: 0; transition: opacity .5s; }
.ma-next.on { opacity: 1; }
.ma-next .t { font: 600 10.5px/1 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-3); }
.ma-next .r { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 9px; font: 600 13px/1 var(--font-mono); color: var(--fg); }
.ma-next .r em { font-style: normal; color: var(--fg-3); font-weight: 500; }
.ma-next .bar { height: 5px; border-radius: 3px; background: color-mix(in srgb, var(--c-head) 22%, transparent); overflow: hidden; min-width: 70px; }
.ma-next .bar > div { height: 100%; border-radius: 3px; background: var(--c-head); }
.ma-next .r.win { color: var(--c-head); }
.ma-cap { display: grid; justify-items: center; gap: 4px; white-space: nowrap; }
.ma-cap span { font: 500 11.5px/1.2 var(--font-mono); color: var(--fg-3); pointer-events: none; }
.ma-cap button { pointer-events: auto; margin-top: 4px; font: 600 11.5px/1 var(--font-mono); color: var(--accent); background: var(--surface); border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent); border-radius: 999px; padding: 7px 13px; cursor: pointer; }
.ma-cap button:hover { background: color-mix(in srgb, var(--accent) 10%, var(--surface)); }
.ma-win { pointer-events: none; display: grid; justify-items: center; gap: 3px; white-space: nowrap; opacity: 0; transition: opacity .5s; }
.ma-win.on { opacity: 1; }
.ma-win b { font: 700 30px/1 var(--font-mono); color: var(--fg); letter-spacing: -.02em; }
.ma-win span { font: 650 13px/1 var(--font-mono); color: var(--c-head); }
.ma-win em { font: 500 11px/1.2 var(--font-mono); font-style: normal; color: var(--fg-3); }
.ma-win b .sp { font-size: .5em; opacity: .4; font-style: normal; margin-right: 1px; }
.ma-legend { display: flex; gap: 12px; justify-content: center; font: 600 11px/1 var(--font-mono); color: var(--fg-2); pointer-events: none; margin-top: 6px; }
.ma-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.ma-lab .d { display: block; font: 500 10.5px/1.35 var(--font-mono); color: var(--fg-3); }
.ma-side { display: grid; gap: 2px; white-space: nowrap; pointer-events: none; }
.ma-side b { font: 650 13.5px/1.15 var(--font-display); color: var(--fg); }
.ma-side span { font: 500 11px/1.2 var(--font-mono); color: var(--fg-3); }
.ma-side i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--c-resid); margin-right: 7px; vertical-align: 1px; }
`);

export default {
  mount(el, ctx) {
    const host = h('div', { class: 'ma' });
    el.append(host);
    const engine = new SceneEngine(host, { onHover: (info) => setHover(info), onPick: () => {}, onContext: onTokenContext, onTheme: () => { look(); build(); } });
    const R = engine.renderer, cam = engine.camera;
    cam.fov = FOV; cam.near = 0.5; cam.far = 600;          // a tight depth range: no z-fighting on thin faces
    cam.updateProjectionMatrix();
    R.shadowMap.enabled = true;
    R.shadowMap.type = THREE.PCFSoftShadowMap;
    engine.controls.minDistance = 10;
    engine.controls.maxDistance = 160;
    engine.controls.maxPolarAngle = Math.PI * 0.6;
    engine.controls.enablePan = false;
    engine.hemi.intensity = 0.0;
    engine.sun.intensity = 0.0;
    const pmrem = new THREE.PMREMGenerator(R);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    engine.scene.environment = envTex;

    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(-7, 26, 14);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.03;
    key.shadow.radius = 4;
    const fill = new THREE.DirectionalLight(0xdfe6ff, 0.6);
    fill.position.set(12, 8, 10);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x8a93a6, 0.7);
    const rim = new THREE.DirectionalLight(0xc9d4ff, 1.0);             // outlines the silhouettes from behind
    rim.position.set(6, 10, -16);
    engine.scene.add(key, key.target, fill, hemi, rim);

    const root = new THREE.Group();
    engine.scene.add(root);

    // ---------------------------------------------------------------- look: theme, tone, bloom
    const dark = () => engine.isDark;
    const tok = (name, fb = '#888888') => { const c = new THREE.Color(); try { c.setStyle(cssVar(name) || fb); } catch { c.setStyle(fb); } return c; };
    const hue = (kind) => tok(`--c-${kind}`);                          // UI tokens: data and accents
    const bodyHue = (kind) => new THREE.Color(BODY[kind] ?? cssVar(`--c-${kind}`) ?? '#888888');
    const hex = (c) => `#${c.getHexString()}`;
    let composer = null, bloom = null, bgTex = null;
    function look() {
      R.toneMapping = dark() ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
      R.toneMappingExposure = dark() ? 1.0 : 1;
      key.intensity = dark() ? 1.9 : 2.0;
      rim.intensity = dark() ? 1.4 : 0.5;
      hemi.intensity = dark() ? 0.5 : 0.75;
      hemi.groundColor.set(dark() ? 0x1a2030 : 0x9aa3b4);
      engine.scene.fog = null;
      // a soft studio backdrop (screen-space, so it never moves), dithered against banding
      bgTex?.dispose();
      const c = document.createElement('canvas'); c.width = c.height = 512;
      const g = c.getContext('2d'), base = tok('--bg');
      const mid = dark() ? new THREE.Color('#151b29') : new THREE.Color('#ffffff');
      const grad = g.createRadialGradient(256, 210, 10, 256, 256, 400);
      grad.addColorStop(0, hex(mid)); grad.addColorStop(1, hex(base));
      g.fillStyle = grad; g.fillRect(0, 0, 512, 512);
      const img = g.getImageData(0, 0, 512, 512);
      for (let i = 0; i < img.data.length; i += 4) { const n = (Math.random() - 0.5) * 3; img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n; }
      g.putImageData(img, 0, 0);
      bgTex = new THREE.CanvasTexture(c);
      bgTex.colorSpace = THREE.SRGBColorSpace;
      engine.scene.background = bgTex;
      // dark: a gentle bloom on the luminous data only (the stream, patterns, active units, the winner); light: plain MSAA
      if (dark() && !composer) {
        const w = Math.max(1, host.clientWidth), hh = Math.max(1, host.clientHeight);
        const rt = new THREE.WebGLRenderTarget(w, hh, { type: THREE.HalfFloatType, samples: 4 });
        composer = new EffectComposer(R, rt);
        composer.setPixelRatio(R.getPixelRatio());
        composer.addPass(new RenderPass(engine.scene, cam));
        bloom = new UnrealBloomPass(new THREE.Vector2(w, hh), 0.55, 0.4, 0.97);   // high threshold: only emissive data glows
        composer.addPass(bloom);
        composer.addPass(new OutputPass());
        composer.setSize(w, hh);
      }
      engine.setRenderFn(dark() ? () => composer.render() : null);
      engine.invalidate();
    }
    const offResize = engine.onResize((w, hh) => { composer?.setSize(w, hh); });

    // ---------------------------------------------------------------- materials and helpers
    let parts = [], labels = [], disposables = [], flow = [], faceCam = [], hot = null, play = null, bounds = null, hoverLab = null, nextCard = null, winLab = null;
    let camKey = '';                                                   // the ⊕ badges turn to the viewer only when the camera moves
    function bodyMat(kind, { k = 0, opacity = 1, rough = 0.32 } = {}) {
      const c = bodyHue(kind).lerp(tok('--bg'), k);
      return new THREE.MeshPhysicalMaterial({
        color: c, roughness: rough, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.18, envMapIntensity: dark() ? 0.75 : 0.9,
        emissive: bodyHue(kind), emissiveIntensity: 0, transparent: opacity < 1, opacity,
      });
    }
    function body(kind, w, hgt, d, opts = {}) {
      const m = new THREE.Mesh(new RoundedBoxGeometry(w, hgt, d, 5, Math.min(opts.r ?? MR, hgt / 2.05, w / 2.05, d / 2.05)), opts.mat ?? bodyMat(kind, opts));
      m.castShadow = opts.cast ?? true;
      m.receiveShadow = true;
      root.add(m);
      return m;
    }
    function part(mesh, data) { Object.assign(mesh.userData, data); engine.addPickable(mesh); parts.push(mesh); return mesh; }
    const V3 = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    function css(html, pos, { cls = '', center = [0.5, 0.5], interactive = false, kind = null } = {}) {
      const l = label(html, { className: cls, interactive, kind });
      l.position.copy(pos); l.center.set(...center);
      root.add(l); labels.push(l);
      return l;
    }
    const lineMats = [];
    function line(points, color, width = 2.2, opacity = 1) {
      const geo = new LineGeometry();
      geo.setPositions(points.flatMap((p) => [p.x, p.y, p.z]));
      const mat = new LineMaterial({ color, linewidth: width, transparent: opacity < 1, opacity, worldUnits: false });
      mat.resolution.set(host.clientWidth || 1, host.clientHeight || 1);
      lineMats.push(mat);
      const l = new Line2(geo, mat);
      l.computeLineDistances();
      root.add(l);
      return l;
    }
    const offLineRes = engine.onResize((w, hh) => lineMats.forEach((m) => m.resolution.set(w, hh)));
    function canvasTex(c) {
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = R.capabilities.getMaxAnisotropy();
      disposables.push(t);
      return t;
    }
    function glyph(text, w, hgt, fg) {
      const c = document.createElement('canvas'), sc = 220;
      c.width = Math.round(w * sc); c.height = Math.round(hgt * sc);
      const g = c.getContext('2d');
      g.fillStyle = fg;
      g.font = `650 ${Math.round(hgt * sc * 0.5)}px "JetBrains Mono", ui-monospace, monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(text, c.width / 2, c.height / 2 + 2);
      return canvasTex(c);
    }
    // a head's pattern: each query row scaled to its own max, masked cells flat (no hatching: it shimmers)
    function patternTex(probs, mask, hh, q) {
      const T = probs.shape[1], S = 192, c = document.createElement('canvas');
      c.width = c.height = S;
      const g = c.getContext('2d'), cell = S / T;
      const lo = new THREE.Color(dark() ? '#0e0c1a' : '#ffffff'), hi = new THREE.Color(dark() ? '#cdb8ff' : '#5b21b6');
      const off = new THREE.Color(dark() ? '#08070f' : '#eef0f5');       // masked cells: the causal triangle shows as a quiet tint
      g.fillStyle = hex(off); g.fillRect(0, 0, S, S);
      for (let i = 0; i < T; i++) {
        let mx = 0;
        for (let j = 0; j < T; j++) mx = Math.max(mx, probs.data[(hh * T + i) * T + j]);
        for (let j = 0; j < T; j++) {
          if (mask && !(mask.data[i * T + j] > 0)) continue;
          const v = Math.pow(probs.data[(hh * T + i) * T + j] / (mx || 1), dark() ? 0.5 : 0.4);
          g.fillStyle = hex(lo.clone().lerp(hi, v));
          g.fillRect(Math.floor(j * cell), Math.floor(i * cell), Math.ceil(cell) + 1, Math.ceil(cell) + 1);
        }
      }
      g.strokeStyle = hex(tok('--accent')); g.lineWidth = Math.max(3, cell * 0.35);
      g.strokeRect(1.5, q * cell, S - 3, cell);
      return canvasTex(c);
    }

    // ---------------------------------------------------------------- build
    function clear() {
      for (const p of parts) engine.removePickable(p);
      parts = []; flow = []; faceCam = []; play = null; hot = null; camKey = '';
      for (const l of labels) l.userData.el.remove();
      labels = []; hoverLab = null; nextCard = null; winLab = null;
      disposables.forEach((t) => t.dispose()); disposables = [];
      lineMats.length = 0;
      root.traverse((o) => { o.geometry?.dispose?.(); if (o.material) [].concat(o.material).forEach((m) => m.dispose()); });
      root.clear();
    }

    function build() {
      clear();
      const st = ctx.state, cfg = st.config, tr = st.trace, T = st.ids.length, L = cfg.n_layer, H = cfg.n_head;
      const q = Math.min(st.sel.token, T - 1), selL = Math.min(st.sel.layer, L - 1), selH = Math.min(st.sel.head, H - 1);
      const pre = cfg.norm_placement === 'pre', res = cfg.residual !== false;
      const normName = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
      const actName = cfg.activation === 'swiglu' ? 'SwiGLU' : cfg.activation === 'gelu' ? 'GELU' : 'ReLU';
      const maskName = { causal: 'causal', none: 'bidirectional', sliding: `sliding window ${cfg.window}`, prefix: 'prefix-LM' }[cfg.mask] || cfg.mask;
      const ink = tok('--fg'), narrow = host.clientWidth < 760;
      const V = cfg.vocab_size, d = cfg.d_model;
      const MWA = Math.min(7.2, Math.max(MW, H * HEAD_W + 0.4));     // attention slab: wide enough for readable heads
      const AX = -(GAP + MWA / 2), MXR = GAP + MW / 2;             // module centres: attention left of the stream, MLP right
      const PW = MWA + MW + 2 * GAP + 0.5, CX = (MW - MWA) / 2;    // plate width and centre (the stream stays at x = 0)

      // ------------------------------------------------ input tokens: a neat row of cards
      const lab = (i) => ctx.label(i);
      const big = narrow ? 1.3 : 1.4;                                 // cards large enough to read; fewer of them on phones
      const cw = (i) => Math.max(0.5, 0.15 * Array.from(lab(i)).length + 0.26) * big;
      let show = [...Array(T).keys()];
      const budget = narrow ? PW * 0.58 : Math.min(PW * 0.74, 9.5), rowW = (ids) => ids.reduce((a, i) => a + cw(i) + 0.1, 0);
      if (rowW(show) > budget) {
        const keep = new Set([q]);
        let a = 0, b = T - 1, right = true;
        while (a <= b) {
          const c2 = right ? b-- : a++;
          keep.add(c2);
          if (rowW([...keep]) + 0.5 > budget) { keep.delete(c2); break; }
          right = !right;
        }
        show = [...keep].sort((x, z) => x - z);
      }
      const total = rowW(show) + (show.some((i, k) => k && i !== show[k - 1] + 1) ? 0.4 : 0);
      let x = CX - total / 2, prev = -1;
      const cardH = 0.6 * big, cardY = 0.36, cardZ = PD / 2 + 0.75, embY = 0.95, embH = 0.44;
      const cardBg = dark() ? new THREE.Color('#1d2433') : new THREE.Color('#ffffff');
      show.forEach((i) => {
        if (prev >= 0 && i !== prev + 1) {
          for (let k = 0; k < 3; k++) { const dd = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ color: tok('--fg-3') })); dd.position.set(x + 0.08 + k * 0.11, cardY, cardZ); root.add(dd); }
          x += 0.4;
        }
        const w = cw(i), sel = i === q, cx = x + w / 2;
        const card = body('embed', w, cardH, 0.06, { r: 0.05, mat: new THREE.MeshPhysicalMaterial({ color: sel ? tok('--accent').lerp(cardBg, 0.8) : cardBg, roughness: 0.4, clearcoat: 0.6, envMapIntensity: 0.6, emissive: bodyHue('embed'), emissiveIntensity: 0 }) });
        card.position.set(cx, cardY + (sel ? 0.1 : 0), cardZ);
        const face = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.92, cardH * 0.86), new THREE.MeshBasicMaterial({ map: glyph(lab(i), w * 0.92, cardH * 0.86, hex(sel ? tok('--accent') : ink)), transparent: true, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2 }));
        face.position.z = 0.031;
        card.add(face);
        part(card, { node: 'tokens', title: `“${lab(i)}”`, sub: `position ${i} · id ${st.ids[i]} · Shift-click to focus`, token: i, trace: 'emb.tok' });
        line([V3(cx, cardY + cardH / 2 + (sel ? 0.1 : 0), cardZ), V3(CX + (cx - CX) * 0.82, embY, PD / 2 - 0.2)], sel ? tok('--accent') : tok('--fg-3'), sel ? 1.6 : 1, sel ? 0.9 : 0.35);
        x += w + 0.1; prev = i;
      });
      const capIn = css(`<span><b style="font:650 13px var(--font-display);color:var(--fg)">Input tokens</b>&nbsp;&nbsp;${T} tokens${narrow ? '' : `${show.length < T ? ` · ${show.length} shown` : ''} · ${st.tokenizer.kind === 'bpe' ? 'subword BPE' : 'characters'}`}</span>${reducedMotion() ? '' : '<button type="button">▶ Run a forward pass</button>'}${narrow ? `<div class="ma-legend">${[['attn', 'Attention'], ['mlp', 'MLP'], ['resid', 'Stream'], ['head', 'LM head']].map(([k, n]) => `<span><i style="background:${BODY[k]}"></i>${n}</span>`).join('')}</div>` : ''}`,
        V3(CX, cardY - 1.45, cardZ + 1.6), { cls: 'ma-cap', interactive: true });
      capIn.userData.el.querySelector('button')?.addEventListener('click', (e) => { e.stopPropagation(); forwardPass(); });

      // ------------------------------------------------ embedding, with the positions engraved on its face
      const embW = PW * 0.9, embD = PD * 0.92;
      const emb = body('embed', embW, embH, embD, { r: 0.18 });
      emb.position.set(CX, embY + embH / 2, 0);
      part(emb, { node: 'embedding', trace: 'emb.out', title: 'Embedding', sub: `token row of W_E${cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal' ? ' + position' : ''}` });
      flow.push({ type: 'glow', obj: emb });
      const posNote = { learned: '+ learned positions', sinusoidal: '+ sinusoidal positions', rope: 'positions: RoPE, inside attention', alibi: 'positions: ALiBi, inside attention', none: 'no positions' }[cfg.pos_encoding] || '';
      railLabel('Embedding', [`${V.toLocaleString()} → ${d}`, posNote], 'embed', embY + embH / 2);
      const absPos = cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal';
      if (absPos) {
        // the positional encoding: its own click target on the embedding's face; a sine engraved only when it is one
        const pts = [], wx = embW * 0.3, x0 = CX - embW / 2 + 0.6, zf = embD / 2 + 0.006;
        for (let k = 0; k <= 60; k++) pts.push(V3(x0 + wx * k / 60, embY + embH / 2 + 0.11 * Math.sin(k / 60 * Math.PI * 4), zf));
        if (cfg.pos_encoding === 'sinusoidal') line(pts, bodyHue('embed').lerp(new THREE.Color(dark() ? '#bff7ef' : '#033b37'), 0.6), 2.6);
        const hit = new THREE.Mesh(new THREE.BoxGeometry(wx + 0.3, embH, 0.2), new THREE.MeshBasicMaterial({ visible: false }));
        hit.position.set(x0 + wx / 2, embY + embH / 2, embD / 2);
        root.add(hit);
        part(hit, { node: 'positional', trace: 'emb.pos', title: 'Positional encoding', sub: `${cfg.pos_encoding}, added to each token's row` });
      }
      const ySpine0 = embY + embH;

      // ------------------------------------------------ the layers
      const Y0 = ySpine0 + 1.0;
      for (let l = 0; l < L; l++) {
        const yP = Y0 + l * PITCH, sel = l === selL;
        const plate = body('norm', PW, PT, PD, { r: 0.045, cast: false, mat: new THREE.MeshPhysicalMaterial({
          color: dark() ? new THREE.Color('#252d42') : new THREE.Color('#ffffff'), roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.1,
          transparent: true, opacity: dark() ? 0.05 : 0.4, envMapIntensity: 0.8, emissive: bodyHue('resid'), emissiveIntensity: 0, depthWrite: false }) });
        plate.position.set(CX, yP + PT / 2, 0);
        plate.receiveShadow = false;
        plate.renderOrder = -1;
        part(plate, { node: `block.${l}`, trace: `L${l}.resid_out`, title: `Layer ${l}`, sub: 'click to open the block', selectLayer: l });
        const e = PD / 2, w2 = PW / 2, yE = yP + PT + 0.004;
        line([V3(CX - w2, yE, e), V3(CX + w2, yE, e), V3(CX + w2, yE, -e), V3(CX - w2, yE, -e), V3(CX - w2, yE, e)], sel ? tok('--accent') : hue('resid'), sel ? 2 : 1.2, sel ? 0.95 : (dark() ? 0.35 : 0.3));
        const attDetail = `${H} heads · ${maskName}${cfg.n_kv_head < H ? ` · ${cfg.n_kv_head === 1 ? 'MQA' : `GQA ${H}/${cfg.n_kv_head}`}` : ''}${cfg.pos_encoding === 'rope' ? ' · RoPE' : cfg.pos_encoding === 'alibi' ? ' · ALiBi' : ''}`;
        const mlpDetail = `${actName} · ${d} → ${cfg.d_ff} → ${d}`;
        railLabel(`Layer ${l}`, sel ? [`attention: ${attDetail}`, `then MLP: ${mlpDetail}`] : [], 'resid', yP + (sel ? 0.95 : 0.7), PD / 2, CX - w2 - 0.25, sel);

        // attention: the body, and its heads standing along the back as screens with their real patterns
        const yA = yP + PT + MH / 2 + 0.02;
        const att = body('attn', MWA, MH, MD);
        att.position.set(AX, yA, 0);
        part(att, { node: `attention.${l}`, trace: `L${l}.attn.out`, title: `Attention · layer ${l}`, sub: attDetail });
        flow.push({ type: 'glow', obj: att });
        const probs = tr[`L${l}.attn.probs`];
        const span = MWA - 0.3, slot = span / H, sw = Math.min(1.5, slot * 0.92), sh = sw * 1.02;
        for (let hh = 0; hh < H; hh++) {
          const isSel = sel && hh === selH;
          const scr = body('attn', sw, sh, 0.04, { r: 0.035, cast: true, mat: new THREE.MeshPhysicalMaterial({ color: isSel ? tok('--accent') : bodyHue('attn').lerp(new THREE.Color(dark() ? '#000000' : '#ffffff'), dark() ? 0.25 : 0.1), roughness: 0.3, clearcoat: 1, envMapIntensity: 0.6, emissive: bodyHue('attn'), emissiveIntensity: 0 }) });
          scr.position.set(AX - span / 2 + (hh + 0.5) * slot, yA + MH / 2 + 0.03 + sh / 2, -MD / 2 + 0.36);
          if (probs) {
            const face = new THREE.Mesh(new THREE.PlaneGeometry(sw * 0.9, sh * 0.9), new THREE.MeshBasicMaterial({ map: patternTex(probs, tr.mask, hh, q), toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2 }));
            face.position.z = 0.022;
            face.userData.lit = dark() ? 1.35 : 1;                       // dark: the patterns themselves glow a little
            face.material.color.setScalar(face.userData.lit);
            scr.add(face);
            flow.push({ type: 'screen', obj: face });
          }
          part(scr, { node: `head.${l}.${hh}`, trace: `L${l}.attn.probs`, title: `Head ${hh} · layer ${l}`, sub: 'its real attention pattern' });
        }

        // the MLP: the body with an inset tray, and its most active units standing as thin bars
        const yM = yA + RISE;
        const mlp = body('mlp', MW, MH, MD);
        mlp.position.set(MXR, yM, 0);
        part(mlp, { node: `mlp.${l}`, trace: `L${l}.mlp.out`, title: `MLP · layer ${l}`, sub: mlpDetail });
        flow.push({ type: 'glow', obj: mlp });
        const tray = body('mlp', MW - 0.34, 0.05, MD - 0.34, { r: 0.025, cast: false, mat: new THREE.MeshPhysicalMaterial({ color: new THREE.Color(dark() ? '#120f08' : '#fffaf0'), roughness: 0.55, envMapIntensity: 0.4 }) });
        tray.position.set(MXR, yM + MH / 2 + 0.012, 0);
        const act = tr[`L${l}.mlp.act`];
        if (act) units(act, q, V3(MXR, yM + MH / 2 + 0.04, 0), l);

        // each sub-layer writes back into the stream: an arrow into a ⊕ on the rod (x + f(x));
        // the norm it reads through sits on the rod just below (pre-LN), or just above the ⊕ (post-LN)
        arrowIn(V3(-GAP, yA, 0), V3(-0.34, yA, 0), bodyHue('attn'));
        arrowIn(V3(GAP, yM, 0), V3(0.34, yM, 0), bodyHue('mlp'));
        plus(yA, 'attn', { node: `residual.${l}.1`, trace: `L${l}.resid_mid`, title: res ? 'x + attention' : 'attention output', sub: pre ? 'the residual add' : `add & ${normName}` });
        plus(yM, 'mlp', { node: `residual.${l}.2`, trace: `L${l}.resid_out`, title: res ? 'x + MLP' : 'MLP output', sub: pre ? 'the residual add' : `add & ${normName}` });
        if (pre) {
          normRing(yA - 0.36, `norm.${l}.1`, `L${l}.ln1`, `${normName} before attention`, 'normalises what attention reads');
          normRing(yM - 0.36, `norm.${l}.2`, `L${l}.ln2`, `${normName} before the MLP`, 'normalises what the MLP reads');
        } else {
          normRing(yA + 0.36, `residual.${l}.1`, `L${l}.resid_mid`, `Add & ${normName}`, 'post-LN: after the add');
          normRing(yM + 0.36, `residual.${l}.2`, `L${l}.resid_out`, `Add & ${normName}`, 'post-LN: after the add');
        }
      }

      // ------------------------------------------------ the head: final norm, LM head, the distribution
      const yTopLayer = Y0 + (L - 1) * PITCH + PT + MH / 2 + 0.02 + RISE + MH / 2 + 0.45;
      let y = yTopLayer + 1.45;
      const ySide = yTopLayer + 0.95;
      if (pre) {
        const fn = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.08, 24, 72), new THREE.MeshPhysicalMaterial({ color: bodyHue('norm'), roughness: 0.18, clearcoat: 1, envMapIntensity: 1, emissive: bodyHue('norm'), emissiveIntensity: 0 }));
        fn.rotation.x = Math.PI / 2; fn.castShadow = true;
        root.add(fn);
        fn.position.set(0, y, 0);
        part(fn, { node: 'final-norm', trace: 'lnf', title: `Final ${normName}`, sub: 'before the LM head' });
        flow.push({ type: 'glow', obj: fn });
        railLabel(`Final ${normName}`, [], 'norm', y);
        y += 0.8;
      }
      // the stream's own label, where the rod runs free above the last layer
      if (!narrow) {
        line([V3(SPINE_R + 0.06, ySide, 0), V3(CX + PW / 2 + 0.15, ySide, 0)], hue('resid'), 1.2, 0.55);
        css(`<div><i></i><b>Residual stream</b></div><span>one ${d}-vector per token${res ? ', carried up' : ' · residual off'}</span>`, V3(CX + PW / 2 + 0.3, ySide, 0), { cls: 'ma-side', center: [0, 0.5] });
      }
      const lmH = 0.44, lmW = PW * 0.9;
      const lm = body('head', lmW, lmH, PD * 0.92, { r: 0.18 });
      lm.position.set(CX, y + lmH / 2, 0);
      part(lm, { node: 'lm-head', trace: 'logits', title: 'LM head', sub: `${d} → ${V.toLocaleString()}${cfg.tie_weights ? ' · tied to the embedding' : ''}` });
      flow.push({ type: 'glow', obj: lm });
      railLabel('LM head · softmax', [`${d} → ${V.toLocaleString()}${cfg.tie_weights ? ' · tied' : ''}`], 'head', y + lmH / 2);
      const ySpine1 = y;
      y += lmH;

      // the distribution: the most likely next tokens, ranked from the centre out (the winner on the stream)
      const pr = Array.from(tr.probs.data.subarray(q * V, (q + 1) * V));
      const order = [...pr.keys()].sort((a, b) => pr[b] - pr[a]);
      const ids = order.slice(0, Math.min(V, 13)).filter((v, r) => r === 0 || pr[v] >= 0.003);
      const step = 0.56;
      let topY = y;
      ids.forEach((v, r) => {
        const p = pr[v], bh = 0.06 + 3.0 * p, win = r === 0, bw = win ? 0.5 : 0.32;
        const geo = new RoundedBoxGeometry(bw, bh, bw, 3, Math.min(0.06, bh / 2.1));
        geo.translate(0, bh / 2, 0);
        const bar = new THREE.Mesh(geo, win
          ? new THREE.MeshPhysicalMaterial({ color: bodyHue('head'), emissive: hue('head'), emissiveIntensity: dark() ? 1.6 : 0.25, roughness: 0.25, clearcoat: 1, envMapIntensity: 0.8 })
          : bodyMat('head', { k: dark() ? 0.35 : 0.45 }));
        if (win) bar.userData.keepEmissive = true;
        bar.castShadow = true;
        bar.position.set((r % 2 ? -1 : 1) * Math.ceil(r / 2) * step, y + 0.01, 0);
        root.add(bar);
        part(bar, { node: 'sampling', trace: 'probs', title: `“${st.tokenizer.label(v)}”`, sub: `${pct(p)}% next` });
        flow.push({ type: 'bar', obj: bar, r });
        topY = Math.max(topY, y + bh);
      });
      winLab = css(`<div class="ma-win"><b>${spaceMark(st.tokenizer.label(ids[0]))}</b><span>${pct(pr[ids[0]])}%</span><em>next after “${esc(lab(q))}”</em></div>`, V3(0, topY + 0.35, 0), { center: [0.5, 1] });
      const rows = order.slice(0, 3).map((v, r) => `<div class="r${r ? '' : ' win'}"><span>${esc(st.tokenizer.label(v))}</span><div class="bar"><div style="width:${Math.max(2, pr[v] * 100).toFixed(1)}%"></div></div><em>${pct(pr[v])}%</em></div>`).join('');
      if (!narrow) {
        nextCard = css(`<div class="ma-next"><div class="t">most likely next</div>${rows}<div class="t" style="text-transform:none;letter-spacing:0">of ${V.toLocaleString()} tokens</div></div>`,
          V3(CX + lmW / 2 + 0.6, y + 0.9, -PD * 0.3), { center: [0, 0.5] });
      }
      [winLab, nextCard].forEach((lb) => lb?.userData.el.firstElementChild.classList.add('on'));

      // ------------------------------------------------ the residual stream: a thick, luminous rod from embedding to LM head
      const spineCol = new THREE.Color(dark() ? '#5c66ee' : BODY.resid);
      const spineMat = new THREE.MeshPhysicalMaterial({ color: spineCol, emissive: spineCol, emissiveIntensity: dark() ? 0.55 : 0.2, roughness: 0.25, clearcoat: 1, transparent: !res, opacity: res ? 1 : 0.35 });
      const spine = new THREE.Mesh(new THREE.CylinderGeometry(SPINE_R, SPINE_R, ySpine1 - ySpine0, 40, 1, true), spineMat);
      spine.position.set(0, (ySpine0 + ySpine1) / 2, 0);
      spine.castShadow = true;
      root.add(spine);
      part(spine, { node: `block.${selL}`, trace: `L${selL}.resid_out`, title: 'Residual stream', sub: `one ${d}-vector per token, layer to layer` });
      const bead = new THREE.Mesh(new THREE.SphereGeometry(SPINE_R * 1.9, 32, 16), new THREE.MeshBasicMaterial({ color: dark() ? new THREE.Color('#ffffff') : hue('resid'), toneMapped: false, transparent: true, opacity: 0 }));
      bead.position.set(0, ySpine0, 0);
      root.add(bead);
      flow.bead = { mesh: bead, y0: ySpine0, y1: ySpine1 };

      // ------------------------------------------------ hover card, floor shadow, framing
      hoverLab = css('<div class="ma-hover"><b></b><span></span></div>', V3(0, 0, 0), { center: [0.5, 1.15] });
      const cs = document.createElement('canvas'); cs.width = cs.height = 256;
      const cg = cs.getContext('2d'), gr = cg.createRadialGradient(128, 128, 6, 128, 128, 128);
      gr.addColorStop(0, 'rgba(0,0,0,1)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      cg.fillStyle = gr; cg.fillRect(0, 0, 256, 256);
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(PW + 6, PD + 6), new THREE.MeshBasicMaterial({ map: canvasTex(cs), transparent: true, opacity: dark() ? 0.55 : 0.16, depthWrite: false }));
      floor.rotation.x = -Math.PI / 2; floor.position.set(CX, 0.01, 0.6);
      root.add(floor);

      // the geometry only: room for the HTML labels is kept by the framing (fit), in screen space
      bounds = new THREE.Box3(V3(CX - PW / 2, -1.5, -PD / 2), V3(CX + PW / 2, topY + 1.45, cardZ + 1.6));
      key.target.position.set(0, topY / 2, 0);
      key.shadow.camera.left = CX - PW; key.shadow.camera.right = CX + PW; key.shadow.camera.top = topY + 4; key.shadow.camera.bottom = -4;
      key.shadow.camera.near = 1; key.shadow.camera.far = 90;
      key.shadow.camera.updateProjectionMatrix();
      engine.invalidate();

      // ---- builders (hoisted)
      function railLabel(name, sub, kind, yy, z = PD / 2, xEdge = CX - PW / 2 - 0.25, sel = false) {
        if (narrow) return;                                           // phones: the hover card names each part
        const lines = (Array.isArray(sub) ? sub : [sub]).filter(Boolean);
        css(`<div><i></i><b>${esc(name)}</b></div>${lines.map((t, k) => `<span${k ? ' class="d"' : ''}>${esc(t)}</span>`).join('')}`, V3(xEdge, yy, z), { cls: `ma-lab${sel ? ' sel' : ''}`, center: [1, 0.5], kind });
      }
      // a ⊕ badge on the rod, turned to the viewer
      function plus(yy, kind, data) {
        const g = new THREE.Group();
        const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.27, 0.27, 0.05, 48), new THREE.MeshStandardMaterial({ color: tok('--surface'), roughness: 0.45 }));
        disc.rotation.x = Math.PI / 2;
        disc.position.z = SPINE_R + 0.06;
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.27, 0.045, 16, 56), new THREE.MeshStandardMaterial({ color: bodyHue(kind), emissive: hue(kind), emissiveIntensity: dark() ? 0.9 : 0.1, roughness: 0.3 }));
        const bm = new THREE.MeshStandardMaterial({ color: bodyHue(kind), emissive: hue(kind), emissiveIntensity: dark() ? 0.9 : 0.1 });
        const b1 = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.055, 0.03), bm), b2 = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.3, 0.03), bm);
        ring.position.z = SPINE_R + 0.06;
        b1.position.z = b2.position.z = SPINE_R + 0.095;
        g.add(disc, ring, b1, b2);
        g.position.set(0, yy, 0);
        root.add(g);
        faceCam.push(g);
        const hit = new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 10), new THREE.MeshBasicMaterial({ visible: false }));
        g.add(hit);
        part(hit, { ...data, group: g });
        flow.push({ type: 'pulse', obj: g, y: yy });
      }
      function arrowIn(from, to, color) {
        line([from, to.clone().add(from.clone().sub(to).normalize().multiplyScalar(0.16))], color, 4.5);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.22, 24), new THREE.MeshStandardMaterial({ color, roughness: 0.35 }));
        cone.position.copy(to.clone().add(from.clone().sub(to).normalize().multiplyScalar(0.1)));
        cone.quaternion.setFromUnitVectors(V3(0, 1, 0), to.clone().sub(from).normalize());
        root.add(cone);
      }
      function normRing(yy, node, trace, title, sub) {
        const m = new THREE.Mesh(new THREE.TorusGeometry(SPINE_R + 0.07, 0.04, 14, 48), new THREE.MeshPhysicalMaterial({ color: bodyHue('norm'), roughness: 0.2, clearcoat: 1, envMapIntensity: 0.9, emissive: bodyHue('norm'), emissiveIntensity: 0 }));
        m.position.set(0, yy, 0);
        m.rotation.x = Math.PI / 2;
        root.add(m);
        part(m, { node, trace, title, sub });
      }
      // the most active hidden units for the selected token, as thin bars at their place in the layer
      // (sign-coloured: amber positive, blue negative), on the tray
      function units(act, t, top, l) {
        const F = act.shape[1], row = act.data.subarray(t * F, (t + 1) * F);
        const idx = [...row.keys()].sort((a, b) => Math.abs(row[b]) - Math.abs(row[a])).slice(0, Math.min(TOPK_UNITS, F));
        const mx = Math.abs(row[idx[0]]) || 1;
        const tw = MW - 0.6, td = MD - 0.6, cols = Math.ceil(Math.sqrt(F * tw / td)), rws = Math.ceil(F / cols);
        const cx = tw / cols, cz = td / rws, s = Math.max(0.075, Math.min(0.13, Math.min(cx, cz) * 1.4));
        const geo = new RoundedBoxGeometry(1, 1, 1, 2, 0.2);
        geo.translate(0, 0.5, 0);
        const mat = new THREE.MeshPhysicalMaterial({ roughness: 0.3, clearcoat: 0.8, envMapIntensity: 0.6, emissive: new THREE.Color('#ffffff'), emissiveIntensity: 0 });
        const im = new THREE.InstancedMesh(geo, mat, idx.length);
        const pos = new THREE.Color(dark() ? '#ffbf47' : '#e08a00'), neg = new THREE.Color(dark() ? '#9aa6b8' : '#5d6b80');   // negative: slate (blue is the stream's)
        const heights = new Float32Array(idx.length), places = [];
        idx.forEach((f, k) => {
          const v = Math.abs(row[f]) / mx;
          heights[k] = 0.08 + 0.72 * v;
          places.push(V3(top.x - tw / 2 + ((f % cols) + 0.5) * cx, top.y, top.z - td / 2 + (Math.floor(f / cols) + 0.5) * cz));
          im.setColorAt(k, (row[f] >= 0 ? pos.clone().multiplyScalar(dark() ? 1.12 : 1) : neg));   // dark: positive units glow softly
        });
        im.castShadow = true;
        root.add(im);
        const sgn = idx.filter((f) => row[f] < 0).length;
        part(im, { node: `neurons.${l}`, trace: `L${l}.mlp.act`, title: `Most active units · layer ${l}`, sub: `top ${idx.length} of ${F} for “${lab(t)}”${sgn ? ` · ${sgn} negative` : ''}` });
        const item = { type: 'pins', obj: im, heights, places, s, n: idx.length };
        flow.push(item);
        setPins(item, 1);
      }
    }

    // ---------------------------------------------------------------- camera: a fixed, framed three-quarter view
    function corners() { const b = bounds, out = []; for (const xx of [b.min.x, b.max.x]) for (const yy of [b.min.y, b.max.y]) for (const zz of [b.min.z, b.max.z]) out.push(V3(xx, yy, zz)); return out; }
    function fit(dir) {
      const target = bounds.getCenter(new THREE.Vector3()), saved = { p: cam.position.clone(), q: cam.quaternion.clone() };
      const cs = corners();
      // keep screen room for the HTML around the model: the label column (left) and the next-token card (right)
      const W = Math.max(1, host.clientWidth), narrowFit = W < 760;
      const xl = narrowFit ? -0.94 : -1 + 2 * (230 / W), xr = narrowFit ? 0.94 : 1 - 2 * (250 / W);
      let lo = 1, hi = 400;
      for (let it = 0; it < 30; it++) {
        const mid = (lo + hi) / 2;
        cam.position.copy(target).addScaledVector(dir, mid); cam.lookAt(target); cam.updateMatrixWorld();
        let ok = true;
        for (const c of cs) { const p = c.clone().project(cam); if (p.x < xl || p.x > xr || p.y > 0.9 || p.y < -0.88 || p.z > 1) { ok = false; break; } }
        if (ok) hi = mid; else lo = mid;
      }
      cam.position.copy(saved.p); cam.quaternion.copy(saved.q); cam.updateMatrixWorld();
      return { target, dist: hi };
    }
    function home(duration = 0) {
      const { target, dist } = fit(VIEW);
      const pos = target.clone().addScaledVector(VIEW, dist);
      if (!duration) { cam.position.copy(pos); engine.controls.target.copy(target); cam.lookAt(target); engine.invalidate(); return Promise.resolve(); }
      return engine.flyTo({ target, position: pos, duration });
    }

    // ---------------------------------------------------------------- the forward pass: one light up the stream
    const DUR = 3600;
    const smooth = (t) => t * t * (3 - 2 * t);
    const outBack = (t) => { const c = 1.4; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };
    function forwardPass() {
      if (reducedMotion() || !flow.bead) return;
      play = { start: performance.now() };
      winLab?.userData.el.firstElementChild.classList.remove('on');
      nextCard?.userData.el.firstElementChild.classList.remove('on');
      stepPlay(play.start);
    }
    // where the light is (0..1 of the stream's height) at time u, and when it reaches height y
    const yAt = (u) => flow.bead.y0 + (flow.bead.y1 - flow.bead.y0) * smooth(Math.min(1, u));
    const uAt = (y) => { const f = (y - flow.bead.y0) / (flow.bead.y1 - flow.bead.y0); let lo = 0, hi = 1; for (let i = 0; i < 24; i++) { const m = (lo + hi) / 2; if (smooth(m) < f) lo = m; else hi = m; } return lo; };
    function stepPlay(now) {
      const u = (play.frozen ?? (now - play.start)) / DUR, b = flow.bead, yNow = yAt(u);
      b.mesh.position.y = yNow;
      b.mesh.material.opacity = u < 1 ? Math.min(1, u * 8) : Math.max(0, 1 - (u - 1) * 6);
      for (const f of flow) {
        const yy = f.type === 'pins' ? f.places[0].y : f.type === 'pulse' ? f.y : f.type === 'bar' ? b.y1 : (f.obj.getWorldPosition?.(new THREE.Vector3()).y ?? 0);
        const t0 = uAt(Math.min(yy, b.y1)), d = (u - t0) * DUR;     // ms since the light passed this part
        if (f.type === 'glow') { if (f.obj !== hot && !f.obj.userData.keepEmissive) f.obj.material.emissiveIntensity = d > 0 ? 0.32 * Math.exp(-d / 420) : 0; }
        else if (f.type === 'screen') f.obj.material.color.setScalar(f.obj.userData.lit * (d > 0 ? 0.18 + 0.82 * Math.min(1, d / 450) : 0.18));
        else if (f.type === 'pulse') f.obj.scale.setScalar(d > 0 ? 1 + 0.35 * Math.sin(Math.min(1, d / 420) * Math.PI) : 1);
        else if (f.type === 'pins') setPins(f, d > 0 ? Math.min(1, d / 700) : 0);
        else if (f.type === 'bar') { const k = Math.max(0, Math.min(1, (d - 120 - f.r * 45) / 650)); f.obj.scale.y = Math.max(0.001, k > 0 ? outBack(k) : 0); }
      }
      if (u >= 1.05 && play.frozen === undefined) {
        for (const f of flow) { if (f.type === 'pins') setPins(f, 1); if (f.type === 'bar') f.obj.scale.y = 1; if (f.type === 'screen') f.obj.material.color.setScalar(f.obj.userData.lit); if (f.type === 'glow' && f.obj !== hot && !f.obj.userData.keepEmissive) f.obj.material.emissiveIntensity = 0; }
        b.mesh.material.opacity = 0;
        winLab?.userData.el.firstElementChild.classList.add('on');
        nextCard?.userData.el.firstElementChild.classList.add('on');
        play = null;
      }
      return true;
    }
    const m4 = new THREE.Matrix4(), qI = new THREE.Quaternion();
    function setPins(f, k) {
      const { obj, heights, places, s: w, n } = f;
      for (let i = 0; i < n; i++) {
        const kk = Math.max(0, Math.min(1, k * 1.5 - (i / n) * 0.5));             // the strongest rise first
        m4.compose(places[i], qI, V3(w, Math.max(0.001, heights[i] * smooth(kk)), w));
        obj.setMatrixAt(i, m4);
      }
      obj.instanceMatrix.needsUpdate = true;
    }
    const offFrame = engine.onFrame(() => {
      const k = `${cam.position.x.toFixed(3)},${cam.position.y.toFixed(3)},${cam.position.z.toFixed(3)}`;
      let moved = false;
      if (k !== camKey) { camKey = k; for (const g of faceCam) g.lookAt(cam.position); moved = true; }
      return play ? stepPlay(performance.now()) : moved;
    });
    // test hook: freeze the forward pass at a time (ms), so a slow headless renderer can capture any moment
    host.__scrub = (ms) => { forwardPass(); if (play) { play.frozen = ms; stepPlay(0); } engine.invalidate(); };

    // ---------------------------------------------------------------- hover and click (colour only: nothing moves)
    function setHover(info) {
      const m = info?.object && parts.includes(info.object) ? info.object : null;
      if (hot && hot !== m) highlight(hot, false);
      if (m && hot !== m) highlight(m, true);
      hot = m;
      const canvas = R.domElement;
      if (m) {
        canvas.dataset.node = m.userData.node;
        if (m.userData.trace && ctx.state.trace[m.userData.trace]) canvas.dataset.trace = m.userData.trace; else delete canvas.dataset.trace;
        const box = new THREE.Box3().setFromObject(m.userData.group || m);
        hoverLab.position.set((box.min.x + box.max.x) / 2, box.max.y + 0.08, (box.min.z + box.max.z) / 2);
        const el2 = hoverLab.userData.el.firstElementChild;
        el2.querySelector('b').textContent = m.userData.title || '';
        el2.querySelector('span').textContent = m.userData.sub || '';
        el2.classList.add('on');
      } else {
        delete canvas.dataset.node; delete canvas.dataset.trace;
        hoverLab?.userData.el.firstElementChild.classList.remove('on');
      }
      engine.invalidate();
    }
    function highlight(m, on) {
      const mat = m.material;
      if (mat && 'emissiveIntensity' in mat && !m.userData.keepEmissive) mat.emissiveIntensity = on ? (dark() ? 0.28 : 0.2) : 0;
    }
    function onClick(e) {
      const info = engine.pick(e);
      const m = info?.object && parts.includes(info.object) ? info.object : null;
      if (!m) return;
      if (m.userData.selectLayer !== undefined && m.userData.selectLayer !== ctx.state.sel.layer) { ctx.store.select({ layer: m.userData.selectLayer }); return; }
      if (m.userData.token !== undefined) {
        const token = m.userData.token, origin = engine.screenRect(m);
        e.preventDefault(); e.stopPropagation();
        ctx.store.select({ token });
        if (!e.shiftKey) ctx.go('tokens', origin);
        return;
      }
      ctx.go(m.userData.node, engine.screenRect(m.userData.group || m));
    }
    R.domElement.addEventListener('click', onClick);

    function onTokenContext(info, e) {
      const m = info?.object && parts.includes(info.object) ? info.object : null;
      if (m?.userData.token === undefined) return;
      const token = m.userData.token, origin = engine.screenRect(m);
      e.stopPropagation();
      openContextMenu(e.clientX, e.clientY, [
        { label: 'Focus token', hint: 'Stay in this view', run: () => ctx.store.select({ token }) },
        { label: 'Zoom into tokens', run: () => { ctx.store.select({ token }); ctx.go('tokens', origin); } },
      ], { title: `“${ctx.label(token)}” · position ${token}` });
    }

    look();
    build();
    engine.setInsets({ top: 60, bottom: 64, left: 0, right: 0 });
    home(0);
    if (!reducedMotion()) setTimeout(() => { if (!destroyed) forwardPass(); }, 500);
    let destroyed = false;
    // a rebuild keeps the view, unless the figure changed size (layers added or removed): then reframe smoothly
    const rebuild = () => {
      const before = bounds?.clone();
      build();
      if (before && (Math.abs(before.max.y - bounds.max.y) > 0.5 || Math.abs(before.min.y - bounds.min.y) > 0.5)) home(reducedMotion() ? 0 : 650);
    };
    const offs = [ctx.on('trace', rebuild), ctx.on('select', rebuild)];

    return {
      pause: () => engine.pause(),
      resume: () => { engine.resume(); home(600); },
      destroy() {
        destroyed = true;
        R.domElement.removeEventListener('click', onClick);
        offFrame(); offResize(); offLineRes(); offs.forEach((f) => f()); clear();
        bgTex?.dispose(); envTex.dispose(); pmrem.dispose(); composer?.dispose?.();
        engine.dispose(); host.remove();
      },
    };
  },
};

// probabilities as percentages that never round a near-certainty up to 100 or a small one down to 0
function pct(p) { return p >= 0.995 && p < 1 ? (p * 100).toFixed(1) : p < 0.001 ? '<0.1' : (p * 100).toFixed(p < 0.1 ? 1 : 0); }
// the leading-space mark drawn small and faint, so '␣saw' reads as a word
function spaceMark(label) { const t = esc(label); return t.startsWith('␣') ? `<i class="sp">␣</i>${t.slice(1)}` : t; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
