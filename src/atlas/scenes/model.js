// Root scene: the decoder-only Transformer as a 3D figure, built on the logic of the
// diagram in "Attention Is All You Need": inputs at the bottom, embedding ⊕ positions,
// one expanded block (its other layers stacked in the stream as thin wafers, the whole
// stack bracketed N×), Linear and Softmax, and the model's real output distribution.
//
// Everything drawn is live data for the selected token: the stream is a ribbon showing
// its residual vector at each depth, attention heads show their patterns on cards, the
// feed-forward band shows every hidden unit's activation. The view opens as a flat,
// paper-like front view and dolly-zooms into 3D. Every part zooms into its own level.
import { SceneEngine, THREE, label } from '../../scene/engine3d.js';
import { RoundedBoxGeometry } from '../../vendor/three/RoundedBoxGeometry.js';
import { RoomEnvironment } from '../../vendor/three/RoomEnvironment.js';
import { h, reducedMotion, cssVar } from '../../ui/index.js';
import { useStyle } from '../kit.js';

const W = 7.2;              // slab width (x)
const D = 2.4;              // slab depth (z)
const RX = W / 2 + 1.0;     // x of the residual loops (right side: things that bypass)
const RAIL = -W / 2 - 1.5;  // x of the label rail (left side: names, things that join)
const RIB = 1.1;            // width of the residual-stream ribbon
const FOV = 24;
let revealed = false;       // the paper-to-3D opening plays once per page load

useStyle('model3d', `
.m3 { position: absolute; inset: 0; }
.m3-rail { display: flex; align-items: baseline; gap: 8px; white-space: nowrap; pointer-events: none; transition: color .15s; }
.m3-rail b { font: 600 13px/1.2 var(--font-body); color: var(--fg); }
.m3-rail span { font: 500 11px/1.2 var(--font-mono); color: var(--fg-3); }
.m3-rail.hot b { color: var(--accent); }
.m3-rail.hot b::after { content: '  open →'; font: 600 11px var(--font-mono); color: var(--accent); }
.m3-cap { display: grid; justify-items: center; gap: 2px; white-space: nowrap; pointer-events: none; }
.m3-cap b { font: 650 14px/1.2 var(--font-display); color: var(--fg); }
.m3-cap span { font: 500 11.5px/1.2 var(--font-mono); color: var(--fg-3); }
.m3-mini { font: 600 11px/1 var(--font-mono); color: var(--fg-3); pointer-events: none; }
.m3-qkv { font: 700 12px/1 var(--font-mono); pointer-events: none; }
.m3-prob { font: 600 12px/1.2 var(--font-mono); color: var(--fg); white-space: nowrap; pointer-events: none; }
.m3-prob.top { font-size: 15px; padding: 3px 9px; border-radius: 6px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--c-head); }
.m3-nx { display: grid; gap: 8px; pointer-events: auto; user-select: none; }
.m3-nx b { font: 700 34px/1 var(--font-display); color: var(--fg-2); }
.m3-nx .pips { display: flex; gap: 5px; }
.m3-nx .pip { width: 22px; height: 22px; border-radius: 6px; border: 1px solid var(--line-strong); background: var(--surface); cursor: pointer; font: 600 10px/1 var(--font-mono); color: var(--fg-3); padding: 0; }
.m3-nx .pip.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.m3-nx span { font: 600 11.5px/1 var(--font-mono); color: var(--fg-3); }
`);

export default {
  mount(el, ctx) {
    const host = h('div', { class: 'm3' });
    el.append(host);
    const engine = new SceneEngine(host, { onHover: (info) => setHover(info), onHoverMove: () => {}, onPick: () => {}, onTheme: () => { applyLook(); build(); } });
    const R = engine.renderer;
    R.toneMapping = THREE.ACESFilmicToneMapping;
    R.shadowMap.enabled = true;
    R.shadowMap.type = THREE.VSMShadowMap;
    engine.camera.fov = FOV;
    engine.camera.updateProjectionMatrix();
    engine.controls.minDistance = 8;
    engine.controls.maxDistance = 140;
    engine.controls.maxPolarAngle = Math.PI * 0.62;
    engine.hemi.intensity = 0.35;
    engine.sun.intensity = 0;
    const pmrem = new THREE.PMREMGenerator(R);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    engine.scene.environment = envTex;

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(5, 26, 9);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 10;
    key.shadow.blurSamples = 16;
    key.shadow.bias = -0.0004;
    Object.assign(key.shadow.camera, { left: -12, right: 12, top: 30, bottom: -4, near: 1, far: 70 });
    const rim = new THREE.DirectionalLight(0xa9b8ff, 0.9);
    rim.position.set(-6, 10, -16);
    engine.scene.add(key, key.target, rim);

    const root = new THREE.Group();
    engine.scene.add(root);
    let parts = [], labels = [], disposables = [], billboards = [], faceCam = [], bounds = null, hot = null;
    let interacted = false, pointerIn = false, settleUntil = 0;
    engine.controls.addEventListener('start', () => { interacted = true; });
    R.domElement.addEventListener('pointerenter', () => { pointerIn = true; });
    R.domElement.addEventListener('pointerleave', () => { pointerIn = false; });

    // ---------------------------------------------------------------- look
    const dark = () => engine.isDark;
    const tok = (name, fb = '#888888') => { const c = new THREE.Color(); try { c.setStyle(cssVar(name) || fb); } catch { c.setStyle(fb); } return c; };
    const hue = (kind) => tok(`--c-${kind}`);
    const bodyTone = () => new THREE.Color(dark() ? '#1b212c' : '#f6f4f1');
    function applyLook() {
      R.toneMappingExposure = dark() ? 1.15 : 0.92;
      engine.scene.fog = null;
    }
    applyLook();
    const muted = (kind, k) => hue(kind).lerp(bodyTone(), k ?? (dark() ? 0.4 : 0.26));
    function bodyMat(kind, { k, opacity = 1, glass = false } = {}) {
      return new THREE.MeshPhysicalMaterial({
        color: muted(kind, k), roughness: glass ? 0.3 : 0.38, metalness: 0, clearcoat: 0.6, clearcoatRoughness: 0.2,
        envMapIntensity: 0.6, sheen: 0.25, sheenColor: hue(kind), emissive: hue(kind), emissiveIntensity: 0,
        transparent: opacity < 1, opacity,
      });
    }
    function outline(mesh, kind) {
      const c = hue(kind).lerp(tok('--bg'), 0.5);
      const mat = new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: dark() ? 0.35 : 0.22 });
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 30), mat);
      mesh.add(lines);
      mesh.userData.outline = lines;
      return lines;
    }
    function slab(kind, w, hgt, d, opts = {}) {
      const m = new THREE.Mesh(new RoundedBoxGeometry(w, hgt, d, 4, Math.min(opts.r ?? 0.12, hgt / 2.2)), bodyMat(kind, opts));
      m.castShadow = true;
      m.receiveShadow = true;
      outline(m, kind);
      return m;
    }
    function addPart(mesh, data) {
      Object.assign(mesh.userData, data);
      engine.addPickable(mesh);
      parts.push(mesh);
      return mesh;
    }
    const V3 = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    function css(html, pos, { cls = '', center = [0.5, 0.5], interactive = false } = {}) {
      const l = label(html, { className: cls, interactive });
      l.position.copy(pos);
      l.center.set(...center);
      root.add(l);
      labels.push(l);
      return l;
    }
    // name on the left rail, with a leader line to the part's front-left corner
    function rail(mesh, name, dims, y, x0, z = D / 2) {
      const pts = [V3(x0 - 0.05, y, z), V3(RAIL + 0.15, y, z)];
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: tok('--fg-3'), transparent: true, opacity: 0.45 }));
      root.add(line);
      const l = css(`<b>${name}</b>${dims ? `<span>${dims}</span>` : ''}`, V3(RAIL, y, z), { cls: 'm3-rail', center: [1, 0.5] });
      if (mesh) mesh.userData.railEl = l.userData.el;
      return l;
    }
    function tube(path, radius, color, segs = 64) {
      const m = new THREE.Mesh(new THREE.TubeGeometry(path, segs, radius, 10, false), new THREE.MeshStandardMaterial({ color, roughness: 0.4, envMapIntensity: 0.5 }));
      m.castShadow = true;
      root.add(m);
      return m;
    }
    function cone(pos, color, dir = 'up', s = 1) {
      const c = new THREE.Mesh(new THREE.ConeGeometry(0.12 * s, 0.24 * s, 20), new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
      c.position.copy(pos);
      if (dir === 'left') c.rotation.z = Math.PI / 2;
      root.add(c);
      return c;
    }
    function arrowUp(y0, y1, color, x = 0) {
      const path = new THREE.LineCurve3(V3(x, y0), V3(x, y1 - 0.2));
      tube(path, 0.05, color, 4);
      cone(V3(x, y1 - 0.12), color);
    }
    function plusNode(y, kind = 'resid', r = 0.3) {
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.05, 14, 48), new THREE.MeshStandardMaterial({ color: hue(kind), roughness: 0.35 }));
      const disc = new THREE.Mesh(new THREE.CircleGeometry(r, 48), new THREE.MeshStandardMaterial({ color: tok('--surface'), roughness: 0.7 }));
      const bm = new THREE.MeshStandardMaterial({ color: hue(kind) });
      g.add(disc, ring, new THREE.Mesh(new THREE.BoxGeometry(r * 1.15, 0.065, 0.02), bm), new THREE.Mesh(new THREE.BoxGeometry(0.065, r * 1.15, 0.02), bm));
      g.position.set(0, y, 0.02);
      root.add(g);
      faceCam.push(g);
      const hit = new THREE.Mesh(new THREE.SphereGeometry(r * 1.25, 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
      g.add(hit);
      return { g, hit };
    }
    function fork(y) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), new THREE.MeshStandardMaterial({ color: hue('resid'), roughness: 0.35 }));
      s.position.set(0, y, 0);
      root.add(s);
    }
    // residual loop: leaves the stream at y0, around the right, back into the ⊕ at y1
    function loop(y0, y1, xIn = 0.36) {
      const r = 0.35, P = (x, y) => V3(x, y);
      const path = new THREE.CurvePath();
      path.add(new THREE.LineCurve3(P(0.1, y0), P(RX - r, y0)));
      path.add(new THREE.QuadraticBezierCurve3(P(RX - r, y0), P(RX, y0), P(RX, y0 + r)));
      path.add(new THREE.LineCurve3(P(RX, y0 + r), P(RX, y1 - r)));
      path.add(new THREE.QuadraticBezierCurve3(P(RX, y1 - r), P(RX, y1), P(RX - r, y1)));
      path.add(new THREE.LineCurve3(P(RX - r, y1), P(xIn + 0.22, y1)));
      tube(path, 0.06, hue('resid'), 180);
      cone(V3(xIn + 0.12, y1, 0), hue('resid'), 'left');
    }

    // diverging ramp for vector values (theme tokens)
    function ramp() {
      const neg = tok('--heat-neg'), mid = tok('--heat-mid'), pos = tok('--heat-pos');
      return (v) => { const t = Math.max(-1, Math.min(1, v)), e = Math.pow(Math.abs(t), 0.75); return (t < 0 ? mid.clone().lerp(neg, e) : mid.clone().lerp(pos, e)); };
    }
    function vectorTexture(vals, range) {
      const f = ramp(), n = vals.length, px = new Uint8Array(n * 4);
      vals.forEach((v, i) => { const c = f(v / range); c.convertLinearToSRGB?.(); px[i * 4] = c.r * 255; px[i * 4 + 1] = c.g * 255; px[i * 4 + 2] = c.b * 255; px[i * 4 + 3] = 255; });
      const t = new THREE.DataTexture(px, n, 1, THREE.RGBAFormat);
      t.magFilter = THREE.NearestFilter;
      t.colorSpace = THREE.SRGBColorSpace;
      t.needsUpdate = true;
      disposables.push(t);
      return t;
    }
    // the stream between y0 and y1 as a camera-facing ribbon showing a real vector
    function ribbon(y0, y1, vals, range, edgeKind = 'resid', withArrow = true) {
      const len = y1 - y0 - (withArrow ? 0.2 : 0);
      if (len <= 0.05) return;
      const g = new THREE.Group();
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(RIB, len), new THREE.MeshBasicMaterial({ map: vectorTexture(vals, range), toneMapped: false, side: THREE.DoubleSide }));
      plane.position.y = len / 2;
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry), new THREE.LineBasicMaterial({ color: hue(edgeKind) }));
      edge.position.y = len / 2;
      g.add(plane, edge);
      if (withArrow) {
        const head = new THREE.Mesh(new THREE.ConeGeometry(RIB * 0.36, 0.22, 3), new THREE.MeshBasicMaterial({ color: hue(edgeKind), toneMapped: false }));
        head.position.y = len + 0.11;
        head.rotation.y = Math.PI / 6;
        g.add(head);
      }
      g.position.set(0, y0, 0);
      root.add(g);
      billboards.push(g);
    }

    // attention pattern drawn on a canvas: masked cells hatched, the selected query row outlined
    function patternTexture(probs, mask, hh, q) {
      const T = probs.shape[1], S = 256, c = document.createElement('canvas');
      c.width = c.height = S;
      const g = c.getContext('2d');
      const lo = tok('--heat-lo'), hi = tok('--c-attn');
      if (!dark()) { lo.multiplyScalar(0.95); hi.multiplyScalar(0.78); }
      const cell = S / T;
      g.fillStyle = `#${lo.getHexString()}`;
      g.fillRect(0, 0, S, S);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const allowed = mask ? mask.data[i * T + j] > 0 : true;
        if (!allowed) continue;
        const v = Math.pow(probs.data[(hh * T + i) * T + j], 0.42);
        g.fillStyle = `#${lo.clone().lerp(hi, v).getHexString()}`;
        g.fillRect(j * cell, i * cell, cell + 0.6, cell + 0.6);
      }
      g.strokeStyle = `#${tok('--c-mask').getHexString()}`;
      g.globalAlpha = 0.5;
      g.lineWidth = 1;
      g.beginPath();
      for (let k = -S; k < S; k += 7) { g.moveTo(k, 0); g.lineTo(k + S, S); }
      g.save();
      g.beginPath();
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if (mask && !(mask.data[i * T + j] > 0)) g.rect(j * cell, i * cell, cell, cell);
      g.clip();
      for (let k = -S; k < S; k += 7) { g.moveTo(k, 0); g.lineTo(k + S, S); }
      g.stroke();
      g.restore();
      g.globalAlpha = 1;
      g.strokeStyle = `#${tok('--accent').getHexString()}`;
      g.lineWidth = Math.max(2, cell * 0.25);
      g.strokeRect(1, q * cell, S - 2, cell);
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      disposables.push(t);
      return t;
    }
    // printed text on a card face
    function glyphTexture(text, w, hgt, fg, bg) {
      const c = document.createElement('canvas'), sc = 200;
      c.width = Math.round(w * sc); c.height = Math.round(hgt * sc);
      const g = c.getContext('2d');
      void bg;
      g.clearRect(0, 0, c.width, c.height);
      g.fillStyle = fg;
      g.font = `700 ${Math.round(hgt * sc * 0.52)}px "JetBrains Mono", ui-monospace, monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(text, c.width / 2, c.height / 2 + 2);
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      disposables.push(t);
      return t;
    }
    // a collapsed layer: its fingerprint (attention thumbnails + MLP strip) printed on the front
    function waferTexture(l, st) {
      const c = document.createElement('canvas'), Wp = 1400, Hp = 64;
      c.width = Wp; c.height = Hp;
      const g = c.getContext('2d');
      g.fillStyle = `#${muted('norm', dark() ? 0.7 : 0.8).getHexString()}`;
      g.fillRect(0, 0, Wp, Hp);
      g.fillStyle = `#${tok('--fg-2').getHexString()}`;
      g.font = '600 28px "Instrument Sans", system-ui, sans-serif';
      g.textBaseline = 'middle';
      g.fillText(`Layer ${l}`, 24, Hp / 2 + 1);
      const probs = st.trace[`L${l}.attn.probs`];
      if (probs) {
        const H = probs.shape[0], T = probs.shape[1], s = 48, lo = tok('--heat-lo'), hi = tok('--c-attn');
        for (let hh = 0; hh < H && hh < 8; hh++) {
          const x0 = 180 + hh * (s + 12);
          for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
            const v = Math.pow(probs.data[(hh * T + i) * T + j], 0.55);
            g.fillStyle = `#${lo.clone().lerp(hi, v).getHexString()}`;
            g.fillRect(x0 + (j * s) / T, 8 + (i * s) / T, s / T + 0.5, s / T + 0.5);
          }
        }
      }
      const act = st.trace[`L${l}.mlp.act`];
      if (act) {
        const F = act.shape[1], t = Math.min(st.sel.token, act.shape[0] - 1), x0 = 760, w = Wp - x0 - 30;
        let mx = 0;
        for (let f = 0; f < F; f++) mx = Math.max(mx, Math.abs(act.data[t * F + f]));
        const m = hue('mlp');
        for (let f = 0; f < F; f++) {
          const v = Math.abs(act.data[t * F + f]) / (mx || 1);
          g.fillStyle = `#${tok('--surface-3').lerp(m, v).getHexString()}`;
          g.fillRect(x0 + (f * w) / F, 22, w / F + 0.5, 20);
        }
      }
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      disposables.push(t);
      return t;
    }

    // ---------------------------------------------------------------- build
    function clear() {
      for (const p of parts) engine.removePickable(p);
      parts = []; billboards = []; faceCam = [];
      for (const l of labels) l.userData.el.remove();
      labels = [];
      disposables.forEach((t) => t.dispose());
      disposables = [];
      root.traverse((o) => { o.geometry?.dispose?.(); if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.dispose(); }); });
      root.clear();
      hot = null;
    }

    function build() {
      clear();
      const st = ctx.state, cfg = st.config, T = st.ids.length, L = cfg.n_layer, tr = st.trace;
      const layer = Math.min(st.sel.layer, L - 1), q = Math.min(st.sel.token, T - 1);
      const pre = cfg.norm_placement === 'pre';
      const normName = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
      const attnName = { causal: 'Masked Multi-Head Attention', none: 'Multi-Head Attention', sliding: 'Sliding-Window Attention', prefix: 'Prefix-LM Attention' }[cfg.mask] || 'Multi-Head Attention';
      const actName = cfg.activation === 'swiglu' ? 'SwiGLU' : cfg.activation === 'gelu' ? 'GELU' : 'ReLU';
      const row = (key) => { const t = tr[key]; if (!t || t.shape.length !== 2) return null; const n = t.shape[1]; return Array.from(t.data.subarray(q * n, (q + 1) * n)); };
      // one colour range for the whole stream, so a LayerNorm visibly evens out the contrast
      const streamKeys = ['emb.tok', 'emb.out', ...Array.from({ length: L }, (_, l) => [`L${l}.resid_mid`, `L${l}.resid_out`, `L${l}.ln1`, `L${l}.ln2`, `L${l}.attn.out`, `L${l}.mlp.out`]).flat(), 'lnf'];
      const all = streamKeys.flatMap((k) => row(k) || []).map(Math.abs).sort((a, b) => a - b);
      const range = all.length ? all[Math.floor(0.98 * (all.length - 1))] || 1 : 1;
      const streamBefore = (l) => (l === 0 ? 'emb.out' : `L${l - 1}.resid_out`);
      let y = 0;

      // ------------------------------------------------ inputs: upright cards with the token printed on
      const lab = (i) => ctx.label(i);
      const cw = (i) => Math.max(0.56, 0.17 * Array.from(lab(i)).length + 0.3);
      let show = [...Array(T).keys()];
      const budget = W + 1.2, total = () => show.reduce((a, i) => a + cw(i) + 0.12, 0);
      if (total() > budget) {
        const keep = new Set([q]);
        let a = 0, b = T - 1, side = 1;
        const width = () => [...keep].reduce((s, i) => s + cw(i) + 0.12, 0) + 0.6;
        while (width() < budget && (a <= b)) {
          const cand = side > 0 ? b-- : a++;
          keep.add(cand);
          if (width() > budget) { keep.delete(cand); break; }
          side = -side;
        }
        show = [...keep].sort((x, z) => x - z);
      }
      let x = -total() / 2, prev = -1;
      const cardY = 0.36, embY0 = 1.35;
      const fgHex = `#${tok('--fg').getHexString()}`;
      show.forEach((i) => {
        if (prev >= 0 && i !== prev + 1) {          // a gap in the sequence: three dots
          for (let k = 0; k < 3; k++) {
            const d = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), new THREE.MeshStandardMaterial({ color: tok('--fg-3') }));
            d.position.set(x + 0.08 + k * 0.12, cardY, 0.6);
            root.add(d);
          }
          x += 0.45;
        }
        const w = cw(i), sel = i === q, cx = x + w / 2;
        const card = slab('embed', w, 0.72, 0.07, { r: 0.05, k: dark() ? 0.5 : 0.55 });
        card.position.set(cx, cardY + (sel ? 0.18 : 0), 0.6);
        const face = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.92, 0.64), new THREE.MeshBasicMaterial({ map: glyphTexture(lab(i), w * 0.92, 0.64, sel ? `#${tok('--accent').getHexString()}` : fgHex, ''), toneMapped: false, transparent: true }));
        face.position.z = 0.04;
        card.add(face);
        if (sel) { card.userData.outline.material.color = tok('--accent'); card.userData.outline.material.opacity = 1; }
        root.add(card);
        addPart(card, { node: 'tokens', title: `“${lab(i)}” · position ${i}`, token: i });
        // each token is embedded on its own: a thread from its card into the embedding
        const xe = cx * 0.85;
        const thread = new THREE.Line(new THREE.BufferGeometry().setFromPoints([V3(cx, cardY + 0.36 + (sel ? 0.18 : 0), 0.6), V3(xe, embY0, 0.5)]),
          new THREE.LineBasicMaterial({ color: sel ? tok('--accent') : tok('--fg-3'), transparent: true, opacity: sel ? 0.95 : 0.25 }));
        root.add(thread);
        x += w + 0.12;
        prev = i;
      });
      css(`<b>Inputs</b><span>${T} tokens${show.length < T ? ` · ${show.length} shown` : ''} · ${st.tokenizer.kind === 'bpe' ? 'subword BPE' : 'characters'}</span>`, V3(0, cardY - 1.1, 0.6), { cls: 'm3-cap' });

      // ------------------------------------------------ embedding ⊕ positions
      y = embY0;
      const embH = 0.55, emb = slab('embed', W, embH, D);
      emb.position.set(0, y + embH / 2, 0);
      root.add(emb);
      addPart(emb, { node: 'embedding', trace: 'emb.out', title: 'Input Embedding' });
      rail(emb, 'Input Embedding', `${cfg.vocab_size} → ${cfg.d_model}`, y + embH / 2, -W / 2);
      y += embH;
      const absPos = cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal';
      if (absPos) {
        const yP = y + 0.65;
        ribbon(y, yP - 0.3, row('emb.tok'), range, 'embed', false);
        const pn = plusNode(yP, 'pos');
        addPart(pn.hit, { node: 'positional', trace: 'emb.pos', title: 'Positional Encoding', group: pn.g });
        // the circled sine joins from the left (left = things that join)
        const pts = [];
        for (let k = 0; k <= 48; k++) pts.push(V3(-2.18 + (k / 48) * 0.86, yP + Math.sin((k / 48) * Math.PI * 2) * 0.16));
        tube(new THREE.CatmullRomCurve3(pts), 0.035, hue('pos'), 80);
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.04, 12, 64), new THREE.MeshStandardMaterial({ color: hue('pos') }));
        ring.position.set(-1.75, yP, 0);
        root.add(ring);
        tube(new THREE.LineCurve3(V3(-1.2, yP), V3(-0.42, yP)), 0.035, hue('pos'), 2);
        cone(V3(-0.36, yP, 0), hue('pos'), 'left', 0.7).rotation.z = -Math.PI / 2;
        const hit = new THREE.Mesh(new THREE.SphereGeometry(0.62, 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
        hit.position.set(-1.75, yP, 0);
        root.add(hit);
        addPart(hit, { node: 'positional', trace: 'emb.pos', title: 'Positional Encoding' });
        rail(hit, 'Positional Encoding', cfg.pos_encoding, yP, -2.32);
        y = yP + 0.3;
      }

      // ------------------------------------------------ the layers: wafers + one expanded block
      const blockY0 = y + 0.25;
      y = blockY0 + 0.3;
      const waferH = 0.32, wafers = [];
      const putWafer = (l) => {
        ribbon(y - 0.55 + 0.05, y, row(streamBefore(l)), range);
        const m = new THREE.Mesh(new RoundedBoxGeometry(W * 0.92, waferH, D * 0.9, 3, 0.08), bodyMat('norm', { k: dark() ? 0.7 : 0.8 }));
        m.castShadow = true; m.receiveShadow = true;
        outline(m, 'norm');
        const face = new THREE.Mesh(new THREE.PlaneGeometry(W * 0.9, waferH * 0.9), new THREE.MeshBasicMaterial({ map: waferTexture(l, st), toneMapped: false }));
        face.position.z = D * 0.45 + 0.005;
        m.add(face);
        m.position.set(0, y + waferH / 2, 0);
        root.add(m);
        addPart(m, { node: `block.${l}`, title: `Layer ${l} · click to open it here`, selectLayer: l });
        wafers.push(m);
        y += waferH + 0.55;
      };
      for (let l = 0; l < layer; l++) putWafer(l);

      // expanded layer
      const yIn = y;
      const forkAndLoop = [];                        // [fork y, sub-layer name] to close with loops at the ⊕
      if (pre) {
        // attention sub-layer: fork, norm, Q/K/V, heads, ⊕
        ribbon(y - 0.55 + 0.05, y, row(streamBefore(layer)), range);
        const f1 = y + 0.12; fork(f1);
        y = f1 + 0.3;
        const ln1 = slab('norm', W * 0.9, 0.12, D * 0.9, { r: 0.04, glass: true, opacity: 0.6 });
        ln1.position.set(0, y + 0.06, 0); root.add(ln1);
        addPart(ln1, { node: `norm.${layer}.1`, trace: `L${layer}.ln1`, title: `${normName} before attention` });
        rail(ln1, normName, 'pre-attention', y + 0.06, -W * 0.45);
        y += 0.12;
        y = qkvFork(y, layer) ;
        y = attention(y, layer, q, attnName);
        const yAdd1 = y + 0.6;
        ribbon(y, yAdd1 - 0.3, row(`L${layer}.attn.out`), range, 'attn');
        const a1 = plusNode(yAdd1, 'resid');
        addPart(a1.hit, { node: `residual.${layer}.1`, trace: `L${layer}.resid_mid`, title: 'Add: x + attention', group: a1.g });
        if (cfg.residual) loop(f1, yAdd1);
        y = yAdd1 + 0.3;
        // MLP sub-layer
        ribbon(y, y + 0.5, row(`L${layer}.resid_mid`), range, 'resid', false);
        const f2 = y + 0.55; fork(f2);
        y = f2 + 0.3;
        const ln2 = slab('norm', W * 0.9, 0.12, D * 0.9, { r: 0.04, glass: true, opacity: 0.6 });
        ln2.position.set(0, y + 0.06, 0); root.add(ln2);
        addPart(ln2, { node: `norm.${layer}.2`, trace: `L${layer}.ln2`, title: `${normName} before the MLP` });
        rail(ln2, normName, 'pre-MLP', y + 0.06, -W * 0.45);
        y += 0.12;
        ribbon(y, y + 0.3, row(`L${layer}.ln2`), range, 'norm');
        y += 0.3;
        y = ffn(y, layer, q, actName);
        const yAdd2 = y + 0.6;
        ribbon(y, yAdd2 - 0.3, row(`L${layer}.mlp.out`), range, 'mlp');
        const a2 = plusNode(yAdd2, 'resid');
        addPart(a2.hit, { node: `residual.${layer}.2`, trace: `L${layer}.resid_out`, title: 'Add: x + MLP', group: a2.g });
        if (cfg.residual) loop(f2, yAdd2);
        y = yAdd2 + 0.3;
        void forkAndLoop;
      } else {
        // post-LN (2017): sub-layer, then Add & Norm
        ribbon(y - 0.55 + 0.05, y, row(streamBefore(layer)), range);
        const f1 = y + 0.12; fork(f1);
        y = qkvFork(f1 + 0.1, layer);
        y = attention(y, layer, q, attnName);
        ribbon(y, y + 0.45, row(`L${layer}.attn.out`), range, 'attn');
        y += 0.5;
        const an1 = slab('norm', W, 0.3, D, { r: 0.08 });
        an1.position.set(0, y + 0.15, 0); root.add(an1);
        addPart(an1, { node: `residual.${layer}.1`, trace: `L${layer}.resid_mid`, title: `Add & ${normName}` });
        rail(an1, `Add & ${normName}`, 'after attention', y + 0.15, -W / 2);
        if (cfg.residual) loop(f1, y + 0.15, W / 2);
        y += 0.3;
        ribbon(y, y + 0.5, row(`L${layer}.resid_mid`), range);
        const f2 = y + 0.55; fork(f2);
        y = f2 + 0.3;
        y = ffn(y, layer, q, actName);
        ribbon(y, y + 0.45, row(`L${layer}.mlp.out`), range, 'mlp');
        y += 0.5;
        const an2 = slab('norm', W, 0.3, D, { r: 0.08 });
        an2.position.set(0, y + 0.15, 0); root.add(an2);
        addPart(an2, { node: `residual.${layer}.2`, trace: `L${layer}.resid_out`, title: `Add & ${normName}` });
        rail(an2, `Add & ${normName}`, 'after the MLP', y + 0.15, -W / 2);
        if (cfg.residual) loop(f2, y + 0.15, W / 2);
        y += 0.3 + 0.25;
      }
      const yOut = y;
      y += 0.25;
      for (let l = layer + 1; l < L; l++) putWafer(l);
      const blockY1 = y - 0.2;

      // backplate tray + the N× bracket (the paper's grey frame, as an object you can see behind the block)
      const bx0 = -W / 2 - 0.6, bx1 = RX + 0.55, fhgt = blockY1 - blockY0 + 0.3;
      const plate = new THREE.Mesh(new RoundedBoxGeometry(bx1 - bx0, fhgt, 0.08, 4, 0.35),
        dark()
          ? new THREE.MeshStandardMaterial({ color: new THREE.Color('#141a24'), roughness: 0.9, envMapIntensity: 0.15, transparent: true, opacity: 0.9 })
          : new THREE.MeshLambertMaterial({ color: new THREE.Color('#e9edf4'), emissive: new THREE.Color('#e9edf4'), emissiveIntensity: 0.6, transparent: true, opacity: 0.6 }));
      plate.position.set((bx0 + bx1) / 2, (blockY0 + blockY1) / 2, -(D / 2 + 0.6));
      plate.receiveShadow = true;
      if (dark()) plate.add(new THREE.LineSegments(new THREE.EdgesGeometry(plate.geometry, 30), new THREE.LineBasicMaterial({ color: new THREE.Color('#2f3747') })));
      root.add(plate);
      const bxN = RX + 0.95;
      const bracket = new THREE.CurvePath();
      bracket.add(new THREE.LineCurve3(V3(bxN - 0.3, blockY0), V3(bxN, blockY0)));
      bracket.add(new THREE.LineCurve3(V3(bxN, blockY0), V3(bxN, blockY1)));
      bracket.add(new THREE.LineCurve3(V3(bxN, blockY1), V3(bxN - 0.3, blockY1)));
      tube(bracket, 0.03, tok('--fg-3'), 12);
      const nx = h('div', { class: 'm3-nx' },
        h('b', null, `${L}×`),
        h('div', { class: 'pips' }, Array.from({ length: L }, (_, l) => h('button', { type: 'button', class: ['pip', l === layer && 'on'], title: `Open layer ${l}`, onClick: (e) => { e.stopPropagation(); ctx.store.select({ layer: l }); } }, String(l)))),
        h('span', null, `layer ${layer} open · ${L} in the stack`));
      const nxl = label('', { interactive: true });
      nxl.userData.el.append(nx);
      nxl.position.set(bxN + 0.35, (blockY0 + blockY1) / 2, 0);
      nxl.center.set(0, 0.5);
      root.add(nxl);
      labels.push(nxl);
      const blockHit = new THREE.Mesh(new THREE.BoxGeometry(0.5, Math.max(0.5, yOut - yIn), 0.5), new THREE.MeshBasicMaterial({ visible: false }));
      blockHit.position.set(bxN, (yIn + yOut) / 2, 0);
      root.add(blockHit);
      addPart(blockHit, { node: `block.${layer}`, title: `Block ${layer}` });

      // ------------------------------------------------ head: final norm, Linear, Softmax, distribution
      y = blockY1 + 0.35;
      if (pre) {
        ribbon(y - 0.5, y + 0.25, row(`L${L - 1}.resid_out`), range);
        y += 0.3;
        const lnf = slab('norm', W * 0.9, 0.12, D * 0.9, { r: 0.04, glass: true, opacity: 0.6 });
        lnf.position.set(0, y + 0.06, 0); root.add(lnf);
        addPart(lnf, { node: 'final-norm', trace: 'lnf', title: `Final ${normName}` });
        rail(lnf, `Final ${normName}`, '', y + 0.06, -W * 0.45);
        y += 0.12;
        ribbon(y, y + 0.45, row('lnf'), range, 'norm');
        y += 0.45;
      } else {
        ribbon(y - 0.5, y + 0.4, row(`L${L - 1}.resid_out`), range);
        y += 0.45;
      }
      const lin = slab('head', W * 0.95, 0.42, D);
      lin.position.set(0, y + 0.21, 0); root.add(lin);
      addPart(lin, { node: 'lm-head', trace: 'logits', title: 'Linear (LM head)' });
      rail(lin, 'Linear', `${cfg.d_model} → ${cfg.vocab_size}${cfg.tie_weights ? ' · tied' : ''}`, y + 0.21, -W * 0.475);
      y += 0.42;
      arrowUp(y, y + 0.4, hue('head'));
      y += 0.4;
      const sm = slab('head', W * 0.95, 0.3, D, { r: 0.08, k: dark() ? 0.45 : 0.5 });
      sm.position.set(0, y + 0.15, 0); root.add(sm);
      addPart(sm, { node: 'sampling', trace: 'probs', title: 'Softmax' });
      rail(sm, 'Softmax', '', y + 0.15, -W * 0.475);
      y += 0.3;

      // distribution comb: every token when V is small, else the top 24
      const V = cfg.vocab_size, pr = Array.from(tr.probs.data.subarray(q * V, (q + 1) * V));
      const order = [...pr.keys()].sort((a, b) => pr[b] - pr[a]);
      const n = V <= 64 ? V : 24, ids = order.slice(0, n);
      const span = W * 0.9, step = span / n, bw = step * 0.7;
      let topY = y;
      ids.forEach((v, r) => {
        const p = pr[v], bh = 0.03 + 1.6 * p;
        const bar = new THREE.Mesh(new RoundedBoxGeometry(bw, bh, 0.5, 2, Math.min(0.04, bh / 2.2)), r === 0
          ? new THREE.MeshPhysicalMaterial({ color: hue('head'), emissive: hue('head'), emissiveIntensity: 0.4, roughness: 0.35, clearcoat: 0.6 })
          : bodyMat('head', { k: dark() ? 0.5 : 0.6 }));
        bar.castShadow = true;
        const bxp = -span / 2 + (r + 0.5) * step;
        bar.position.set(bxp, y + bh / 2 + 0.01, 0);
        root.add(bar);
        addPart(bar, { node: 'sampling', trace: 'probs', title: `“${st.tokenizer.label(v)}” ${(p * 100).toFixed(1)}%` });
        if (r === 0) css(`<span class="m3-prob top">${esc(st.tokenizer.label(v))} ${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%</span>`, V3(bxp, y + bh + 0.32, 0), { center: [0.5, 0.5] });
        topY = Math.max(topY, y + bh);
      });
      const runners = ids.slice(1, 3).map((v) => `${esc(st.tokenizer.label(v))} ${(pr[v] * 100).toFixed(pr[v] < 0.1 ? 1 : 0)}%`).join(' · ');
      css(`<b>Output probabilities</b><span>next token after “${esc(ctx.label(q))}” · then ${runners}${V > 64 ? ` · top ${n} of ${V.toLocaleString()}` : ''}</span>`, V3(0, topY + 1.35, 0), { cls: 'm3-cap' });

      // contact shadow under the inputs
      const cs = document.createElement('canvas');
      cs.width = cs.height = 128;
      const cg = cs.getContext('2d'), grad = cg.createRadialGradient(64, 64, 4, 64, 64, 64);
      grad.addColorStop(0, 'rgba(0,0,0,1)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
      cg.fillStyle = grad; cg.fillRect(0, 0, 128, 128);
      const ct = new THREE.CanvasTexture(cs);
      disposables.push(ct);
      const contact = new THREE.Mesh(new THREE.PlaneGeometry(W + 3, 3.2), new THREE.MeshBasicMaterial({ map: ct, transparent: true, opacity: dark() ? 0.5 : 0.18, depthWrite: false }));
      contact.rotation.x = -Math.PI / 2;
      contact.position.set(0, -0.02, 0.4);
      root.add(contact);

      bounds = new THREE.Box3(V3(RAIL - 2.6, -1.3, -D / 2), V3(bxN + 2.6, topY + 1.3, D / 2 + 0.9));
      key.target.position.set(0, (blockY0 + blockY1) / 2, 0);
      engine.invalidate();

      // ---- builders used above (hoisted function declarations close over this build)
      function qkvFork(y0, l) {
        const y1 = y0 + 0.62;
        const xs = [['q', -1.6], ['k', 0], ['v', 1.6]];
        for (const [kind, xx] of xs) {
          const path = new THREE.QuadraticBezierCurve3(V3(0, y0), V3(xx * 0.15, y0 + 0.4), V3(xx, y1 - 0.18));
          tube(path, 0.04, hue(kind), 24);
          cone(V3(xx, y1 - 0.1, 0), hue(kind), 'up', 0.75);
          css(`<span class="m3-qkv" style="color: var(--c-${kind})">${kind.toUpperCase()}</span>`, V3(xx * 0.55 + (xx >= 0 ? 0.22 : -0.22), y0 + 0.3, 0.2), { center: [xx >= 0 ? 0 : 1, 0.5] });
        }
        const hit = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.6, 0.6), new THREE.MeshBasicMaterial({ visible: false }));
        hit.position.set(0, y0 + 0.31, 0);
        root.add(hit);
        addPart(hit, { node: `qkv.${l}`, trace: `L${l}.attn.q`, title: 'Q, K, V projections' });
        return y1;
      }
      function attention(y0, l, qq, name) {
        const baseH = 0.22;
        const base = slab('attn', W, baseH, D);
        base.position.set(0, y0 + baseH / 2, 0);
        root.add(base);
        addPart(base, { node: `attention.${l}`, trace: `L${l}.attn.out`, title: name });
        const tags = [];
        if (cfg.pos_encoding === 'rope') tags.push('RoPE');
        if (cfg.pos_encoding === 'alibi') tags.push('ALiBi');
        if (cfg.n_kv_head < cfg.n_head) tags.push(`GQA ${cfg.n_head}/${cfg.n_kv_head} KV`);
        rail(base, name, `${cfg.n_head} heads · d_h ${cfg.d_model / cfg.n_head}${tags.length ? ' · ' + tags.join(' · ') : ''}`, y0 + baseH / 2, -W / 2);
        const probs = tr[`L${l}.attn.probs`], H = cfg.n_head;
        const pw = Math.min(1.6, (W - 0.4) / H - 0.18), chh = pw * 0.95;
        for (let hh = 0; hh < H; hh++) {
          const card = slab('attn', pw, chh, 0.08, { r: 0.06, k: dark() ? 0.6 : 0.7 });
          const cx = -W / 2 + 0.2 + (hh + 0.5) * ((W - 0.4) / H);
          card.position.set(cx, y0 + baseH + chh / 2 + 0.02, -0.1);
          card.rotation.x = -0.22;
          if (probs) {
            const face = new THREE.Mesh(new THREE.PlaneGeometry(pw * 0.9, chh * 0.9), new THREE.MeshBasicMaterial({ map: patternTexture(probs, tr.mask, hh, qq), toneMapped: false }));
            face.position.z = 0.045;
            card.add(face);
          }
          root.add(card);
          addPart(card, { node: `head.${l}.${hh}`, trace: `L${l}.attn.probs`, title: `Head ${hh}` });

        }
        return y0 + baseH + chh * Math.cos(0.22) + 0.08;
      }
      function ffn(y0, l, qq, act) {
        const fh = 1.1, wb = W * 0.68, wm = W * 1.02, sh = new THREE.Shape();
        sh.moveTo(-wb / 2, 0);
        sh.lineTo(wb / 2, 0);
        sh.quadraticCurveTo(wm / 2, fh * 0.12, wm / 2, fh / 2);
        sh.quadraticCurveTo(wm / 2, fh * 0.88, wb / 2, fh);
        sh.lineTo(-wb / 2, fh);
        sh.quadraticCurveTo(-wm / 2, fh * 0.88, -wm / 2, fh / 2);
        sh.quadraticCurveTo(-wm / 2, fh * 0.12, -wb / 2, 0);
        const geo = new THREE.ExtrudeGeometry(sh, { depth: D, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 6, curveSegments: 24 });
        geo.translate(0, 0, -D / 2);
        const m = new THREE.Mesh(geo, bodyMat('mlp'));
        m.castShadow = true; m.receiveShadow = true;
        outline(m, 'mlp');
        m.position.set(0, y0, 0);
        root.add(m);
        addPart(m, { node: `mlp.${l}`, trace: `L${l}.mlp.out`, title: `Feed Forward · ${act}` });
        rail(m, `Feed Forward · ${act}`, `${cfg.d_model} → ${cfg.d_ff} → ${cfg.d_model}`, y0 + fh / 2, -wm / 2);
        // the waist band: every hidden unit's |activation| for the selected token
        const a = tr[`L${l}.mlp.act`];
        if (a) {
          const F = a.shape[1], t = Math.min(qq, a.shape[0] - 1), px = new Uint8Array(F * 4);
          let mx = 0;
          for (let f = 0; f < F; f++) mx = Math.max(mx, Math.abs(a.data[t * F + f]));
          const lo = new THREE.Color(dark() ? '#120d03' : '#2a1f08'), hi = hue('mlp').lerp(new THREE.Color('#ffffff'), 0.25);
          for (let f = 0; f < F; f++) {
            const c = lo.clone().lerp(hi, Math.pow(Math.abs(a.data[t * F + f]) / (mx || 1), 0.7));
            px[f * 4] = c.r * 255; px[f * 4 + 1] = c.g * 255; px[f * 4 + 2] = c.b * 255; px[f * 4 + 3] = 255;
          }
          const tex = new THREE.DataTexture(px, F, 1, THREE.RGBAFormat);
          tex.magFilter = THREE.NearestFilter;
          tex.needsUpdate = true;
          disposables.push(tex);
          const band = new THREE.Mesh(new THREE.PlaneGeometry(wm * 0.86, 0.17), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
          band.position.set(0, y0 + fh / 2, D / 2 + 0.065);
          root.add(band);
          const bh = new THREE.Mesh(new THREE.PlaneGeometry(wm * 0.86, 0.17), new THREE.MeshBasicMaterial({ visible: false }));
          bh.position.copy(band.position);
          root.add(bh);
          addPart(bh, { node: `neurons.${l}`, trace: `L${l}.mlp.act`, title: `${F} hidden units` });
        }
        return y0 + fh;
      }
    }

    // ---------------------------------------------------------------- camera
    const homeDir = new THREE.Vector3(Math.sin(0.45), 0.24, Math.cos(0.45)).normalize();
    function corners() {
      const b = bounds, out = [];
      for (const xx of [b.min.x, b.max.x]) for (const yy of [b.min.y, b.max.y]) for (const zz of [b.min.z, b.max.z]) out.push(V3(xx, yy, zz));
      return out;
    }
    // smallest distance at which the whole figure fits (|ndc| within the margins), for a direction and fov
    function fit(dir, fov) {
      const cam = engine.camera, target = bounds.getCenter(new THREE.Vector3());
      const saved = { pos: cam.position.clone(), quat: cam.quaternion.clone(), fov: cam.fov };
      cam.fov = fov;
      cam.updateProjectionMatrix();
      const cs = corners();
      let lo = 1, hi = 2000;
      for (let it = 0; it < 32; it++) {
        const mid = (lo + hi) / 2;
        cam.position.copy(target).addScaledVector(dir, mid);
        cam.lookAt(target);
        cam.updateMatrixWorld();
        let ok = true;
        for (const c of cs) { const p = c.clone().project(cam); if (Math.abs(p.x) > 0.9 || p.y > 0.93 || p.y < -0.86 || p.z > 1) { ok = false; break; } }
        if (ok) hi = mid; else lo = mid;
      }
      cam.position.copy(saved.pos);
      cam.quaternion.copy(saved.quat);
      cam.fov = fov;                 // the caller wants this lens
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
      return { target, dist: hi };
    }
    function goHome(duration = 0) {
      const { target, dist } = fit(homeDir, FOV);
      return engine.flyTo({ target, position: target.clone().addScaledVector(homeDir, dist), duration });
    }
    // signature move: the paper's flat figure dolly-zooms into the 3D machine
    function reveal() {
      if (reducedMotion()) { root.scale.z = 1; goHome(0); return; }
      const front = new THREE.Vector3(0, 0, 1), cam = engine.camera;
      root.scale.z = 0.04;
      engine.controls.enabled = false;
      const sstep = (a, b, t) => { const u = Math.max(0, Math.min(1, (t - a) / (b - a))); return u * u * (3 - 2 * u); };
      const apply = (f) => {
        const fov = 4 + (FOV - 4) * f;
        const dir = front.clone().lerp(homeDir, sstep(0.3, 1, f)).normalize();
        const { target, dist } = fit(dir, fov);
        cam.position.copy(target).addScaledVector(dir, dist);
        engine.controls.target.copy(target);
        cam.lookAt(target);
        root.scale.z = 0.04 + 0.96 * sstep(0, 0.8, f);
      };
      apply(0);
      engine.tween({ duration: 400, update: () => {} })
        .then(() => engine.tween({ duration: 1800, easing: (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2), update: (e) => apply(e) }))
        .then(() => { engine.controls.enabled = true; settleUntil = performance.now() + 10000; });
    }

    // ---------------------------------------------------------------- per frame
    const offFrame = engine.onFrame((t) => {
      const cam = engine.camera.position;
      for (const b of billboards) b.rotation.y = Math.atan2(cam.x - b.position.x, cam.z - b.position.z);
      for (const g of faceCam) g.lookAt(cam);
      // a gentle sway for a few seconds after the opening, frozen while the pointer is on the canvas
      if (!interacted && !pointerIn && bounds && performance.now() < settleUntil && !reducedMotion()) {
        const { target, dist } = fitCache || (fitCache = fit(homeDir, FOV));
        const a = Math.sin(t / 3.2) * 0.07;
        const dir = new THREE.Vector3(Math.sin(0.45 + a), homeDir.y * Math.hypot(Math.sin(0.45), Math.cos(0.45)), Math.cos(0.45 + a)).normalize();
        engine.camera.position.copy(target).addScaledVector(dir, dist);
        engine.controls.target.copy(target);
        return true;
      }
      return false;
    });
    let fitCache = null;

    // ---------------------------------------------------------------- hover & click
    function setHover(info) {
      const m = info?.object && parts.includes(info.object) ? info.object : null;
      if (hot && hot !== m) lift(hot, false);
      if (m && hot !== m) lift(m, true);
      hot = m;
      const canvas = R.domElement;
      if (m) {
        canvas.dataset.node = m.userData.node;
        if (m.userData.trace && ctx.state.trace[m.userData.trace]) canvas.dataset.trace = m.userData.trace; else delete canvas.dataset.trace;
      } else { delete canvas.dataset.node; delete canvas.dataset.trace; }
      engine.invalidate();
    }
    function lift(m, on) {
      const obj = m.userData.group || m;
      const toCam = engine.camera.position.clone().sub(obj.getWorldPosition(new THREE.Vector3())).normalize().multiplyScalar(0.12);
      if (on) { obj.userData.home = obj.position.clone(); obj.position.add(toCam); } else if (obj.userData.home) obj.position.copy(obj.userData.home);
      const ol = m.userData.outline;
      if (ol) ol.material.opacity = on ? 1 : (dark() ? 0.35 : 0.22);
      if (m.material?.emissiveIntensity !== undefined && !m.userData.keepEmissive) m.material.emissiveIntensity = on ? 0.18 : 0;
      m.userData.railEl?.classList.toggle('hot', on);
    }
    R.domElement.addEventListener('click', (e) => {
      const info = engine.pick(e);
      const m = info?.object && parts.includes(info.object) ? info.object : null;
      e.preventDefault();
      if (!m) return;
      if (m.userData.selectLayer !== undefined) { ctx.store.select({ layer: m.userData.selectLayer }); return; }
      if (m.userData.token !== undefined) ctx.store.select({ token: m.userData.token });
      const target = m.getWorldPosition(new THREE.Vector3());
      const dir = engine.camera.position.clone().sub(engine.controls.target).normalize();
      const dist = engine.camera.position.distanceTo(engine.controls.target);
      engine.flyTo({ target, position: target.clone().add(dir.multiplyScalar(dist * 0.42)), duration: reducedMotion() ? 0 : 520 })
        .then(() => ctx.go(m.userData.node, engine.screenRect(m)));
    });

    build();
    engine.setInsets({ top: 60, bottom: 64, left: 0, right: 0 });
    if (!revealed) { revealed = true; reveal(); } else { goHome(0); settleUntil = performance.now() + 6000; }
    const rebuild = () => { build(); fitCache = null; };
    const offs = [ctx.on('trace', rebuild), ctx.on('select', rebuild)];

    return {
      pause: () => engine.pause(),
      resume: () => { engine.resume(); fitCache = null; goHome(700); },
      destroy() { offFrame(); offs.forEach((f) => f()); clear(); envTex.dispose(); pmrem.dispose(); engine.dispose(); host.remove(); },
    };
  },
};

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
