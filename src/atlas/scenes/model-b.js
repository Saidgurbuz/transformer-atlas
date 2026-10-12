// Transformer Atlas · root view, variant b: "the stem".
//
// The whole decoder-only model as one calm object in a soft studio. The residual stream is a
// polished indigo stem rising from the input tokens to the LM head. Every layer grows two leaves
// from it, attention to the left and the MLP to the right, alternating upward in the order they
// run, each joined to the stem by the ring where its output is added back. Data lives on neutral
// "screens": the heads' real attention patterns stand on the attention leaf, the hidden units'
// activations rise as a field of pillars from the MLP leaf, and the next-token distribution
// stands on the LM head. The stem's thickness follows the selected token's residual norm.
//
// Calm by construction: the camera never moves on its own, hover only changes colour, labels are
// few and fixed, and the one animation (a forward pass) is a single soft pulse up the stem.
import { SceneEngine, THREE, label as cssLabel, ease } from '../../scene/engine3d.js';
import { RoundedBoxGeometry } from '../../vendor/three/RoundedBoxGeometry.js';
import { h, reducedMotion } from '../../ui/index.js';
import { useStyle } from '../kit.js';
import { createLook } from './model-b/look.js';
import { keycapTexture, patternTexture, shadowTexture } from './model-b/paint.js';
import { tablet, disposeTablets } from './model-b/geo.js';

const NODE = 0.155;        // radius of an add node, a swelling of the stem
const G = NODE + 0.012;    // stem axis to a leaf's inner edge: leaves touch their node
const MW = 4.5;            // leaf width (x)
const MD = 1.7;            // leaf depth (z)
const MT = 0.24;           // leaf thickness
const WIDE = 2 * (G + MW); // embedding and LM head width
const KH = 0.48;           // keycap height
const KD = 0.44;           // keycap depth
const BAR_H = 2.0;         // height of a probability-1 bar
const FOV = 22;
const RAIL = 0.95;         // left rail captions sit this far beyond the leaves, outside the layer numbers
const HOME = { az: 0.42, el: 0.34 };
let opened = false;        // the opening forward pass plays once per page load

useStyle('model-b', `
.mb { position: absolute; inset: 0; }
.mb-cap { display: grid; gap: 3px; white-space: nowrap; pointer-events: none; user-select: none; }
.mb-cap b { font: 650 13.5px/1.15 var(--font-display); color: var(--fg); letter-spacing: -.005em; display: flex; align-items: center; gap: 7px; }
.mb-cap b::before { content: ''; width: 7px; height: 7px; border-radius: 2px; background: var(--kind, var(--fg-3)); flex: none; }
.mb-cap span { font: 500 11px/1.2 var(--font-mono); color: var(--fg-3); padding-left: 14px; }
.mb-cap.r { justify-items: end; text-align: right; }
.mb-cap.r b { flex-direction: row-reverse; }
.mb-cap.r span { padding: 0 14px 0 0; }
.mb-cap.c { justify-items: center; text-align: center; }
.mb-cap.c span { padding: 0; }
.mb-cap[data-node] { pointer-events: auto; cursor: zoom-in; }
.mb-cap[data-node]:hover b { color: var(--accent); }
.mb-num { pointer-events: auto; cursor: zoom-in; border: 0; background: none; padding: 3px 6px; border-radius: 6px; font: 600 11px/1 var(--font-mono); color: var(--fg-3); letter-spacing: .04em; transition: color .15s, background .15s; }
.mb-num:hover { color: var(--fg); background: color-mix(in srgb, var(--surface) 75%, transparent); }
.mb-num.on { color: var(--accent); }
.mb-win { display: grid; justify-items: center; gap: 4px; white-space: nowrap; pointer-events: none; transition: opacity .45s ease; }
.mb-win b { font: 650 19px/1 var(--font-mono); color: var(--fg); letter-spacing: -.01em; }
.mb-win b em { font-style: normal; color: var(--fg-3); font-weight: 500; }
.mb-win span { font: 600 12px/1 var(--font-mono); color: var(--c-head); }
.mb-win small { font: 500 10.5px/1.2 var(--font-mono); color: var(--fg-3); }
.mb-tip { font: 600 11.5px/1 var(--font-mono); color: var(--fg); white-space: nowrap; pointer-events: none; padding: 5px 8px; border-radius: 7px; background: color-mix(in srgb, var(--surface) 88%, transparent); border: 1px solid var(--line); box-shadow: var(--shadow-1); opacity: 0; transition: opacity .12s; }
.mb-tip.on { opacity: 1; }
.mb-tip em { font-style: normal; color: var(--fg-3); font-weight: 500; }
.mb-play { pointer-events: auto; margin-top: 7px; border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--line)); background: color-mix(in srgb, var(--surface) 80%, transparent); color: var(--accent); border-radius: 999px; padding: 5px 11px; font: 600 11px/1 var(--font-mono); cursor: pointer; justify-self: center; }
.mb-play:hover { background: color-mix(in srgb, var(--accent) 10%, var(--surface)); }
.mb-compact .mb-cap.side { display: none; }        /* a phone: the hover card names each part */
.mb-compact .mb-cap span { display: none; }
.mb-compact .mb-cap.c span { display: block; }
.mb-compact .mb-cap b { font-size: 12px; gap: 5px; }
.mb-compact .mb-win small { display: none; }
.mb-compact .mb-win b { font-size: 16px; }
.mb-compact .mb-num { padding: 2px 3px; font-size: 10px; }
`);

export default {
  mount(el, ctx) {
    const host = h('div', { class: 'mb' });
    el.append(host);
    let look = null;
    const engine = new SceneEngine(host, {
      onHover: (info, ev) => setHover(info, ev),
      onHoverMove: () => {},
      onPick: () => {},
      onTheme: () => { look.refresh(); build(); },
    });
    const R = engine.renderer, cam = engine.camera, canvas = R.domElement;
    cam.fov = FOV;
    cam.near = 0.5;
    cam.far = 400;
    cam.updateProjectionMatrix();
    Object.assign(engine.controls, { minDistance: 6, maxDistance: 140, maxPolarAngle: Math.PI / 2 - 0.04, rotateSpeed: 0.55, zoomSpeed: 0.8 });
    look = createLook(engine);
    look.refresh();

    const root = new THREE.Group();
    engine.scene.add(root);
    let builtWhileFontsLoading = false, fontsDirty = false, hoverKey = '';
    let parts = [], labels = [], textures = [], hot = null, anim = null, flow = null, bounds = null, framePts = [];
    let userMoved = false;
    engine.controls.addEventListener('start', () => { userMoved = true; });
    const tip = cssLabel('', { className: 'mb-tip' });
    tip.center.set(0.5, 1.15);
    engine.scene.add(tip);

    // ------------------------------------------------------------------ helpers
    const V3 = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const tokHTML = (s) => esc(s).replace(/^␣/, '<em>␣</em>');
    const box = (w, hh, d, r = 0.08, seg = 4) => new RoundedBoxGeometry(w, hh, d, seg, Math.min(r, hh / 2.05, w / 2.05, d / 2.05));
    function mesh(geo, mat, pos) {
      const m = new THREE.Mesh(geo, mat);
      if (pos) m.position.copy(pos);
      root.add(m);
      return m;
    }
    function tex(t) { textures.push(t); return t; }
    /** Register a pickable part. glow: materials whose emissive lights on hover / pulse. */
    function part(m, data, glow = [m.material]) {
      m.userData.part = true;
      Object.assign(m.userData, data);
      m.userData.g = { mats: glow.filter(Boolean), base: glow.map((mt) => mt?.emissiveIntensity ?? 0), hover: 0, pulse: 0, hoverK: data.hoverK ?? (look.dark ? 0.32 : 0.2) };
      engine.addPickable(m);
      parts.push(m);
      return m;
    }
    function applyGlow(m) {
      const g = m.userData.g;
      g.mats.forEach((mt, i) => {
        if (mt.isMeshBasicMaterial) mt.color.setScalar((mt.userData.k ?? 1) * (1 + 0.6 * (g.hover + g.pulse)));
        else mt.emissiveIntensity = g.base[i] + g.hover + g.pulse;
      });
    }
    function proxy(geo, pos) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ visible: false }));
      m.position.copy(pos);
      root.add(m);
      return m;
    }
    function caption(html, pos, { cls = '', center = [0.5, 0.5], kind = null, node = null } = {}) {
      const l = cssLabel(html, { className: `mb-cap ${cls}`, kind, interactive: !!node });
      if (node) l.userData.el.dataset.node = node;
      l.position.copy(pos);
      l.center.set(...center);
      root.add(l);
      labels.push(l);
      return l;
    }
    function tube(points, radius, mat, segs = 48) {
      const curve = points.length === 2 ? new THREE.LineCurve3(...points) : new THREE.CatmullRomCurve3(points);
      return mesh(new THREE.TubeGeometry(curve, segs, radius, 8, false), mat);
    }

    // ------------------------------------------------------------------ build
    function clear() {
      hoverKey = '\u0000';                                 // after a rebuild the old parts are gone
      for (const p of parts) engine.removePickable(p);
      parts = [];
      for (const l of labels) l.userData.el.remove();
      labels = [];
      textures.forEach((t) => t.dispose());
      textures = [];
      root.traverse((o) => {
        if (!o.geometry?.userData?.shared) o.geometry?.dispose?.();
        if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.alphaMap?.dispose?.(); m.dispose(); });
      });
      root.clear();
      hot = null;
      flow = null;
      setTip(null);
    }

    function build({ animate = false } = {}) {
      clear();
      builtWhileFontsLoading = document.fonts?.status === 'loading';
      const st = ctx.state, cfg = st.config, tr = st.trace, C = look.c, dark = look.dark;
      const T = st.ids.length, L = cfg.n_layer, H = cfg.n_head, Hkv = cfg.n_kv_head ?? H, V = cfg.vocab_size;
      const q = Math.max(0, Math.min(st.sel.token ?? T - 1, T - 1));
      const lsel = Math.max(0, Math.min(st.sel.layer ?? 0, L - 1));
      const pre = cfg.norm_placement === 'pre';
      const hasResid = cfg.residual !== false;
      const absPos = cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal';
      const normName = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
      const actName = { swiglu: 'SwiGLU', gelu: 'GELU', relu: 'ReLU' }[cfg.activation] || cfg.activation;
      const maskName = { causal: 'causal', none: 'no mask', sliding: `sliding window ${cfg.window}`, prefix: `prefix ${cfg.prefix_len}` }[cfg.mask] || cfg.mask;
      const S = L <= 4 ? 0.8 : L <= 6 ? 0.72 : 0.66;
      const row = (key) => { const t = tr[key]; if (!t || t.shape.length !== 2) return null; const n = t.shape[1]; return t.data.subarray(q * n, (q + 1) * n); };
      const norm2 = (v) => (v ? Math.sqrt(v.reduce((a, x) => a + x * x, 0)) : 0);
      flow = { mods: [], fields: [], bars: [], key: null, keyThread: null, head: null, win: null, rodMat: null, y0: 0, y1: 1 };

      // vertical layout
      const PT = 0.07, yE0 = KH + 0.62 + (absPos ? PT + 0.004 : 0), ET = 0.26, yE1 = yE0 + ET;
      const yP1 = yE1;
      const y0 = yP1 + 0.7;
      const ymod = (k) => y0 + k * S;
      const yTop = ymod(2 * L - 1) + MT;
      const yHs = yTop + 0.78, FT = pre ? 0.07 : 0, yH0 = yHs + (pre ? FT + 0.004 : 0), HT = 0.26, yH1 = yH0 + HT, yI = yH1 + 0.034;

      // ---------------------------------------------------------------- the stem (residual stream)
      // One continuous turned form. Its waist follows ‖x‖ of the selected token after each add, and it
      // swells into a round node at every add, where a leaf writes its output back onto the stream.
      const knots = [{ y: yP1, n: norm2(row('emb.out')) }];
      for (let k = 0; k < 2 * L; k++) {
        const l = k >> 1, key = k % 2 ? `L${l}.resid_out` : `L${l}.resid_mid`;
        knots.push({ y: ymod(k) + MT / 2, n: norm2(row(key)) });
      }
      const ns = knots.map((kn) => kn.n), nmin = Math.min(...ns), nmax = Math.max(...ns);
      const rOf = (n) => (nmax > nmin * 1.08 ? 0.07 + 0.05 * (n - nmin) / (nmax - nmin) : 0.095);
      const waist = (y) => {
        let r = rOf(knots[0].n);
        for (let i = 1; i < knots.length; i++) {
          const t = clamp01((y - knots[i].y + 0.2) / 0.4), sm = t * t * (3 - 2 * t);
          r += (rOf(knots[i].n) - rOf(knots[i - 1].n)) * sm;
        }
        return r;
      };
      const smax = (a, b, k = 0.025) => k * Math.log(Math.exp(a / k) + Math.exp(b / k));
      const radiusAt = (y) => {
        let r = waist(y);
        for (let i = 1; i < knots.length; i++) {
          const d = (y - knots[i].y) / (NODE * 1.05);
          if (Math.abs(d) < 1.6) r = smax(r, NODE * Math.sqrt(Math.max(0, 1 - d * d)), 0.035);
        }
        return r;
      };
      const ya = yP1 - 0.02, yb = yHs + 0.02, prof = [new THREE.Vector2(0, ya)];
      const nS = Math.ceil((yb - ya) / 0.014);
      for (let i = 0; i <= nS; i++) { const y = ya + ((yb - ya) * i) / nS; prof.push(new THREE.Vector2(radiusAt(y), y)); }
      prof.push(new THREE.Vector2(0, yb));
      const rodMat = look.bodyMat('resid', { color: dark ? look.body('resid') : C.resid.clone(), metalness: dark ? undefined : 0.35, roughness: 0.24, clearcoat: 1, emissive: dark ? 0.45 : 0.08, glow: C.resid, opacity: hasResid ? 1 : 0.3 });
      pulseShader(rodMat);
      const rod = mesh(new THREE.LatheGeometry(prof, 56), rodMat);
      flow.rodMat = rodMat;
      flow.y0 = yP1; flow.y1 = yHs;
      const junction = (y) => { for (let i = 1; i < knots.length; i++) if (Math.abs(y - knots[i].y) < NODE * 0.85) return i - 1; return -1; };
      const segment = (y) => { let k = -1; for (let i = 1; i < knots.length; i++) if (y >= knots[i].y) k = i - 1; return k; };
      const addId = (k) => ({ node: `residual.${k >> 1}.${k % 2 ? 2 : 1}`, trace: k % 2 ? `L${k >> 1}.resid_out` : `L${k >> 1}.resid_mid` });
      part(rod, {
        cellAt: (hit) => { const j = junction(hit.point.y); return { r: j >= 0 ? j : segment(hit.point.y), c: j >= 0 ? 1 : 0, fake: true }; },
        pickAt: (hit) => {
          const j = junction(hit.point.y);
          if (j >= 0) return addId(j);
          const k = segment(hit.point.y);
          return k < 0 ? { node: 'embedding', trace: 'emb.out' } : addId(k);
        },
        onHot: (hit) => {           // light only the node or the stretch of stream under the pointer
          const u = rodMat.userData.u;
          if (!hit) { u.uHi.value.z = 0; return; }
          const j = junction(hit.point.y), k = segment(hit.point.y);
          if (j >= 0) u.uHi.value.set(knots[j + 1].y - NODE, knots[j + 1].y + NODE, 1);
          else u.uHi.value.set(k < 0 ? yP1 - 0.1 : knots[k + 1].y + NODE * 0.6, k + 2 < knots.length ? knots[k + 2].y - NODE * 0.6 : yHs + 0.1, 1);
        },
      }, []);

      // ---------------------------------------------------------------- input tokens (keycaps)
      const lab = (i) => ctx.label(i);
      const kw = (i) => Math.max(0.46, Math.min(1.9, 0.26 + 0.15 * Array.from(lab(i)).length));
      const GAPK = 0.1, budget = WIDE - 0.4;
      let a = q, b = q, used = kw(q);
      for (let side = 0, stuck = 0; stuck < 2; side ^= 1) {
        const i = side === 0 ? a - 1 : b + 1;                 // grow left first: the context that leads to the prediction
        if (i < 0 || i >= T) { stuck++; continue; }
        const extra = kw(i) + GAPK + ((side === 0 ? i > 0 : i < T - 1) ? 0.38 : 0);
        if (used + extra > budget) { stuck++; continue; }
        stuck = 0;
        used += kw(i) + GAPK;
        if (side === 0) a = i; else b = i;
      }
      const shown = [];
      for (let i = a; i <= b; i++) shown.push(i);
      const ell = (a > 0 ? 0.38 : 0) + (b < T - 1 ? 0.38 : 0);
      let x = -(used + ell) / 2;
      const yK = KH / 2, zK = 0.3, dotMat = new THREE.MeshStandardMaterial({ color: C['fg-3'], roughness: 0.6 });
      const dots = (x0) => { for (let d = 0; d < 3; d++) mesh(new THREE.SphereGeometry(0.028, 12, 8), dotMat, V3(x0 + 0.1 + d * 0.09, 0.1, zK + KD / 2)); };
      if (a > 0) { dots(x); x += 0.38; }
      const threadMat = new THREE.MeshBasicMaterial({ color: C['fg-3'], transparent: true, opacity: dark ? 0.22 : 0.2, depthWrite: false });
      shown.forEach((i) => {
        const w = kw(i), cx = x + w / 2, sel = i === q, future = i > q;
        const bodyC = sel ? C.accent.clone() : dark ? C['surface-2'].clone() : new THREE.Color(1, 1, 1);
        const mat = look.bodyMat('resid', { color: bodyC, glow: C.accent, roughness: 0.42, metalness: 0, clearcoat: 0.5, env: dark ? 0.7 : 0.75 });
        const k = mesh(box(w, KH, KD, 0.075, 3), mat, V3(cx, yK, zK));
        const fg = sel ? new THREE.Color(1, 1, 1) : future ? C['fg-3'] : C.fg;
        const face = new THREE.MeshPhysicalMaterial({ map: tex(keycapTexture(R, lab(i), { w: w - 0.1, h: KH - 0.1, fg, dim: sel ? new THREE.Color(1, 1, 1).lerp(bodyC, 0.45) : C['fg-3'], bg: bodyC })), roughness: mat.roughness, clearcoat: mat.clearcoat, clearcoatRoughness: mat.clearcoatRoughness, envMapIntensity: mat.envMapIntensity, emissive: C.accent.clone(), emissiveIntensity: 0, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
        const fm = new THREE.Mesh(new THREE.PlaneGeometry(w - 0.1, KH - 0.1), face);
        fm.position.set(0, 0, KD / 2 + 0.002);
        k.add(fm);
        part(k, { token: i, title: lab(i) }, [mat, face]);
        if (sel) flow.key = k;
        // every token is embedded on its own: a hairline into the embedding; the selected one in accent
        void threadMat;
        x += w + GAPK;
      });
      if (b < T - 1) { dots(x - GAPK + 0.02); }
      {
        const sk = flow.key, kx = sk.position.x;
        const am = new THREE.MeshBasicMaterial({ color: C.accent, transparent: true, opacity: 0.95 });
        const yIn = yE0 - (absPos ? PT + 0.004 : 0);
        flow.keyThread = tube([V3(kx, KH + 0.01, zK), V3(kx * 0.55, (KH + yIn) / 2 + 0.04, zK * 0.6), V3(0, yIn + 0.01, 0.1)], 0.014, am, 32);
      }
      const inCap = caption(`<b>Input</b><span>${T} token${T === 1 ? '' : 's'}${shown.length < T ? ` · ${shown.length} shown` : ''} · ${st.tokenizer.kind === 'bpe' ? 'subword' : 'characters'}</span>`, V3(0, -0.22, zK + KD / 2), { cls: 'c', center: [0.5, 0], kind: 'embed', node: 'tokens' });
      if (!reducedMotion()) {
        const btn = h('button', { type: 'button', class: 'mb-play', title: 'Play one forward pass' }, '▶ forward pass');
        btn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); play(1); });
        inCap.userData.el.append(btn);
      }

      // ---------------------------------------------------------------- embedding (+ positions)
      const embMat = look.bodyMat('embed');
      const emb = mesh(tablet(WIDE, ET, MD, 0.5, 0.07), embMat, V3(0, yE0 + ET / 2, 0));
      part(emb, { node: 'embedding', trace: 'emb.out' });
      flow.mods.push({ m: emb, y: yE0 + ET / 2 });
      if (absPos) {        // the positions added to every token vector: a sky sole beneath the embedding
        const pm = look.bodyMat('pos');
        const p = mesh(tablet(WIDE - 0.24, PT, MD - 0.24, 0.42, 0.028), pm, V3(0, yE0 - 0.004 - PT / 2, 0));
        part(p, { node: 'positional', trace: 'emb.pos' });
        flow.mods.push({ m: p, y: yE0 });
      }
      caption(`<b>Embedding</b><span>${V.toLocaleString('en-US')} × ${cfg.d_model}${absPos ? ` + ${cfg.pos_encoding} positions` : ''}</span>`, V3(-WIDE / 2 - RAIL, yE0 + ET / 2, MD / 2), { cls: 'r side', center: [1, 0.5], kind: 'embed' });

      // ---------------------------------------------------------------- the layers: alternating leaves
      for (let k = 0; k < 2 * L; k++) {
        const l = k >> 1, isAttn = k % 2 === 0, side = isAttn ? -1 : 1, yk = ymod(k), xc = side * (G + MW / 2);
        const kind = isAttn ? 'attn' : 'mlp';
        const mat = look.bodyMat(kind);
        const leaf = mesh(tablet(MW, MT, MD, 0.42, 0.06), mat, V3(xc, yk + MT / 2, 0));
        const glowMats = [mat];
        if (isAttn) part(leaf, { node: `attention.${l}`, trace: `L${l}.attn.out` }, glowMats);
        else part(leaf, { node: `mlp.${l}`, trace: `L${l}.mlp.out` }, glowMats);
        const mod = { m: leaf, y: yk + MT / 2, extra: [] };
        flow.mods.push(mod);
        // pre-LN: the norm the leaf reads the stream through, as a slate sole beneath it
        if (pre) {
          const nm = look.bodyMat('norm');
          const sole = mesh(tablet(MW - 0.3, 0.06, MD - 0.3, 0.3, 0.025), nm, V3(xc, yk - 0.004 - 0.03, 0));
          part(sole, { node: `norm.${l}.${isAttn ? 1 : 2}`, trace: `L${l}.${isAttn ? 'ln1' : 'ln2'}` });
          mod.extra.push(sole);
        }
        // the add: a round node on the stem where the leaf writes back (post-LN: Add & Norm)
        const yr = yk + MT / 2;
        if (!pre) {             // post-LN: the add is followed by a norm, a slate collar on the node
          const cm = look.bodyMat('norm', { roughness: 0.3, metalness: 0.2, opacity: hasResid ? 1 : 0.4 });
          const col = mesh(new THREE.TorusGeometry(NODE + 0.012, 0.03, 12, 48), cm, V3(0, yr, 0));
          col.rotation.x = Math.PI / 2;
        }

        if (isAttn) buildHeads(l, xc, yk + MT, mod);
        else buildNeurons(l, xc, yk + MT, mod);
      }

      // layer numbers on the left rail: each opens its block
      for (let l = 0; l < L; l++) {
        const y = (ymod(2 * l) + ymod(2 * l + 1)) / 2 + MT / 2;
        const btn = h('button', { type: 'button', class: ['mb-num', l === lsel && 'on'], dataset: { node: `block.${l}` }, title: `Block ${l}` }, `L${l}`);
        const lbl = cssLabel('', { interactive: true });
        lbl.userData.el.append(btn);
        lbl.position.set(-WIDE / 2 - 0.12, y, MD / 2);
        lbl.center.set(1, 0.5);
        root.add(lbl);
        labels.push(lbl);
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const ms = parts.filter((p) => new RegExp(`^(attention|mlp)\\.${l}$`).test(p.userData.node));
          const grp = new THREE.Group();
          const rect = rectOf(ms);
          ctx.store.select({ layer: l });
          ctx.go(`block.${l}`, rect);
          void grp;
        });
        btn.addEventListener('pointerenter', () => { for (const p of parts) if (new RegExp(`^(attention|mlp)\\.${l}$`).test(p.userData.node)) { p.userData.g.hover = p.userData.g.hoverK; applyGlow(p); } engine.invalidate(); });
        btn.addEventListener('pointerleave', () => { for (const p of parts) if (new RegExp(`^(attention|mlp)\\.${l}$`).test(p.userData.node) && p !== hot) { p.userData.g.hover = 0; applyGlow(p); } engine.invalidate(); });
      }
      const tags = [];
      if (Hkv < H) tags.push(Hkv === 1 ? 'MQA' : `GQA ${H}/${Hkv}`);
      if (cfg.pos_encoding === 'rope' || cfg.pos_encoding === 'alibi') tags.push(cfg.pos_encoding === 'rope' ? 'RoPE' : 'ALiBi');
      // column captions sit on the rails, between two layer numbers
      const numY = (l) => (ymod(2 * l) + ymod(2 * l + 1)) / 2 + MT / 2;
      const colY = L >= 2 ? (numY((L >> 1) - 1) + numY(L >> 1)) / 2 : numY(0) + S * 0.75;
      caption(`<b>Attention</b><span>${H} head${H === 1 ? '' : 's'} · ${maskName}${tags.length ? ` · ${tags.join(' · ')}` : ''}</span>`, V3(-WIDE / 2 - RAIL, colY, MD / 2), { cls: 'r side', center: [1, 0.5], kind: 'attn' });
      const yRes = yHs - 0.42, colYR = Math.min(colY, yRes - 0.8);
      caption(`<b>MLP</b><span>${actName} · ${cfg.d_model} → ${cfg.d_ff} → ${cfg.d_model}</span>`, V3(WIDE / 2 + 0.3, colYR, MD / 2), { cls: 'side', center: [0, 0.5], kind: 'mlp' });
      caption(`<b>Residual stream</b><span>${hasResid ? `${pre ? 'pre' : 'post'}-norm · ${L} block${L === 1 ? '' : 's'}` : 'residual connections off'}</span>`, V3(WIDE / 2 + 0.3, yRes, MD / 2), { cls: 'side', center: [0, 0.5], kind: 'resid' });

      // ---------------------------------------------------------------- final norm, LM head
      if (pre) {
        const fm = look.bodyMat('norm');
        const f = mesh(tablet(WIDE - 0.24, FT, MD - 0.24, 0.42, 0.028), fm, V3(0, yHs + FT / 2, 0));
        part(f, { node: 'final-norm', trace: 'lnf' });
        flow.mods.push({ m: f, y: yHs });
      }
      const hm = look.bodyMat('head');
      const head = mesh(tablet(WIDE, HT, MD, 0.5, 0.07), hm, V3(0, yH0 + HT / 2, 0));
      part(head, { node: 'lm-head', trace: 'logits' });
      flow.mods.push({ m: head, y: yH0 + HT / 2 });
      flow.head = head;
      const scr = mesh(tablet(WIDE - 0.42, 0.03, MD - 0.42, 0.32, 0.012), look.screenMat({ color: look.tray('head') }), V3(0, yH1 + 0.015 + 0.002, 0));
      void scr;
      caption(`<b>LM head</b><span>${cfg.d_model} → ${V.toLocaleString('en-US')}${cfg.tie_weights ? ' · tied' : ''}${pre ? ` · after final ${normName}` : ''}</span>`, V3(-WIDE / 2 - RAIL, yH0 + HT / 2, MD / 2), { cls: 'r side', center: [1, 0.5], kind: 'head' });

      // ---------------------------------------------------------------- the prediction: top-k bars
      const pr = tr.probs.data.subarray(q * V, (q + 1) * V);
      const order = Array.from({ length: V }, (_, i) => i).sort((i, j) => pr[j] - pr[i]);
      const K = Math.min(V, 11), ids = order.slice(0, K).filter((v, r) => r < 3 || pr[v] >= 0.01);
      const pmax = pr[ids[0]] || 1, bw = 0.34, step = Math.min(0.56, (WIDE - 1.2) / Math.max(ids.length, 1));
      let barsTop = yI;
      ids.forEach((v, r) => {
        const p = pr[v], bh = Math.max(0.03, p * BAR_H);
        const bx = (r % 2 ? -1 : 1) * Math.ceil(r / 2) * step;
        const geo = box(bw, bh, bw, Math.min(0.06, bh / 2.1), 2);
        geo.translate(0, bh / 2, 0);
        const ink = look.ink('head');
        const c = r === 0 ? ink : look.tray('head').lerp(ink, 0.3 + 0.55 * Math.pow(p / pmax, 0.6));
        const bm = look.bodyMat('head', { color: c, emissive: dark ? (r === 0 ? 0.9 : 0.25 + 0.4 * (p / pmax)) : (r === 0 ? 0.08 : 0), glow: look.c.head, roughness: 0.36 });
        const bar = mesh(geo, bm, V3(bx, yI, 0));
        part(bar, { node: 'sampling', trace: 'probs', cell: { r: q, x: v, band: 0 }, tipHTML: `${tokHTML(st.tokenizer.label(v))} <em>${pct(p)}%</em>`, tipAt: V3(bx, yI + bh + 0.08, 0) });
        flow.bars.push({ m: bar, r });
        barsTop = Math.max(barsTop, yI + bh);
      });
      {
        const v = ids[0], p = pr[v], wl = st.tokenizer.label(v);
        const win = cssLabel(`<b>${tokHTML(wl)}</b><span>${pct(p)}%</span><small>next after “${esc(ctx.label(q))}”${V > K ? ` · top ${ids.length} of ${V.toLocaleString('en-US')}` : ''}</small>`, { className: 'mb-win' });
        win.position.set(0, yI + Math.max(p * BAR_H, 0.34) + 0.22, 0);
        win.center.set(0.5, 1);
        root.add(win);
        labels.push(win);
        flow.win = win;
        win.userData.fitAt = V3(0, yI + BAR_H + 0.22, 0);
      }

      // ---------------------------------------------------------------- floor: a soft contact shadow
      const sm = new THREE.MeshBasicMaterial({ color: dark ? new THREE.Color(0, 0, 0) : C.fg.clone().lerp(C.resid, 0.25), alphaMap: tex(shadowTexture(R)), transparent: true, opacity: dark ? 0.5 : 0.1, depthWrite: false });
      const sh = mesh(new THREE.PlaneGeometry(WIDE + 0.8, 2.2), sm, V3(0, -0.002, 0.25));
      sh.rotation.x = -Math.PI / 2;
      sh.renderOrder = -1;

      // framing: the object plus room for the rail labels
      const yWin = yI + pmax * BAR_H + 0.95;
      bounds = new THREE.Box3(V3(-WIDE / 2, 0, -MD / 2), V3(WIDE / 2, yI + BAR_H, MD / 2 + 0.3));
      void barsTop;
      void yWin;
      framePts = [];
      for (const xx of [bounds.min.x, bounds.max.x]) for (const yy of [bounds.min.y, bounds.max.y]) for (const zz of [bounds.min.z, bounds.max.z]) framePts.push(V3(xx, yy, zz));

      prime(animate);
      engine.invalidate();

      // ---- builders (hoisted; close over this build)
      function buildHeads(l, xc, ytop, mod) {
        const probs = tr[`L${l}.attn.probs`];
        const groups = Hkv < H && Hkv > 1 ? Hkv : 1, per = H / groups;
        const gapH = 0.08, gapG = groups > 1 ? 0.16 : 0;
        const avail = MW - 0.34;
        const pw = Math.min(0.92, 2 * S - MT - 0.36, (avail - (H - 1) * gapH - (groups - 1) * gapG) / H);
        const total = H * pw + (H - 1) * gapH + (groups - 1) * gapG;
        let px = xc - total / 2 + pw / 2;
        const zP = MD / 2 - 0.5, tilt = -0.14;
        const lo = dark ? look.screen().lerp(C.attn, 0.1) : new THREE.Color(1, 1, 1).lerp(C.attn, 0.13);
        const masked = dark ? look.screen() : new THREE.Color(1, 1, 1);
        const hi = dark ? C['heat-hi'].clone().lerp(new THREE.Color(1, 1, 1), 0.1) : C['heat-hi'].clone().multiplyScalar(0.8);
        for (let hh = 0; hh < H; hh++) {
          if (hh > 0 && hh % per === 0) px += gapG;
          const g = new THREE.Group();
          g.position.set(px, ytop + 0.004, zP);
          g.rotation.x = tilt;
          root.add(g);
          const bm = look.screenMat({ color: dark ? look.screen().lerp(C.attn, 0.12) : look.body('attn').lerp(new THREE.Color(1, 1, 1), 0.35) });
          const body = new THREE.Mesh(box(pw, pw, 0.036, 0.024, 2), bm);
          body.position.y = pw / 2;
          g.add(body);
          let face = null;
          if (probs) {
            face = new THREE.MeshBasicMaterial({ map: tex(patternTexture(R, probs, tr.mask, hh, { lo, hi, masked, gamma: dark ? 0.34 : 0.28 })), toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
            face.userData.k = dark ? 1.9 : 1;
            face.color.setScalar(face.userData.k);
            const fmesh = new THREE.Mesh(new THREE.PlaneGeometry(pw * 0.9, pw * 0.9), face);
            fmesh.position.set(0, pw / 2, 0.02);
            g.add(fmesh);
          }
          const T2 = probs ? probs.shape[1] : 1;
          part(body, {
            node: `head.${l}.${hh}`, trace: `L${l}.attn.probs`,
            cellAt: (hit) => {
              if (!probs) return null;
              const lp = body.worldToLocal(hit.point.clone());                       // centre of the panel = origin
              const u = (lp.x + 0.45 * pw) / (0.9 * pw), w = 1 - (lp.y + 0.45 * pw) / (0.9 * pw);
              if (u < 0 || u >= 1 || w < 0 || w >= 1) return null;
              const xx = Math.min(T2 - 1, Math.floor(u * T2));
              return { r: Math.min(T2 - 1, Math.floor(w * T2)), x: xx, c: xx, band: hh };
            },
          }, [bm, face]);
          mod.extra.push(body);
          px += pw + gapH;
        }
        // Q, K, V: three coloured inlays on the leaf's front edge, widths ∝ their projection widths
        const yS = ytop - MT / 2, zS = MD / 2 + 0.004;
        let sx = -G - 0.32;
        const qw = 0.46, kvw = Math.max(0.1, qw * Hkv / H), stripes = [['q', qw], ['k', kvw], ['v', kvw]];
        const sMats = [];
        const x0 = sx;
        for (const [kk, w] of stripes) {
          const sm = look.bodyMat(kk, { color: look.ink(kk), emissive: dark ? 0.6 : 0.05, glow: C[kk], roughness: 0.3 });
          sMats.push(sm);
          mesh(box(w, 0.075, 0.05, 0.02, 2), sm, V3(sx - w / 2, yS, zS));
          sx -= w + 0.05;
        }
        const qp = proxy(new THREE.BoxGeometry(x0 - sx + 0.1, MT + 0.04, 0.12), V3((x0 + sx) / 2, yS, MD / 2 + 0.03));
        part(qp, { node: `qkv.${l}`, trace: `L${l}.attn.q` }, sMats);
        if (cfg.pos_encoding === 'rope' || cfg.pos_encoding === 'alibi') {
          const pm = look.bodyMat('pos', { color: look.ink('pos'), emissive: dark ? 0.6 : 0.05, roughness: 0.3 });
          const w = 0.2, px2 = sx - 0.08 - w / 2;
          mesh(box(w, 0.075, 0.05, 0.02, 2), pm, V3(px2, yS, zS));
          const pp = proxy(new THREE.BoxGeometry(w + 0.1, MT + 0.04, 0.12), V3(px2, yS, MD / 2 + 0.03));
          part(pp, { node: 'positional', trace: cfg.pos_encoding === 'alibi' ? 'alibi' : `L${l}.attn.q_rot` }, [pm]);
        }
      }

      function buildNeurons(l, xc, ytop, mod) {
        const act = tr[`L${l}.mlp.act`];
        const sw = MW - 0.36, sd = MD - 0.36;
        const scr = look.screenMat({ color: look.tray('mlp') });
        mesh(tablet(sw, 0.03, sd, 0.3, 0.012), scr, V3(xc, ytop + 0.015 + 0.002, 0));
        if (!act) return;
        const F = act.shape[1], t = Math.min(q, act.shape[0] - 1);
        const aw = sw - 0.2, ad = sd - 0.2;
        const cols = Math.max(1, Math.ceil(Math.sqrt(F * aw / ad))), rows = Math.ceil(F / cols);
        const cell = Math.min(aw / cols, ad / rows), fp = cell * 0.86;
        const gx = cols * cell, gz = rows * cell;
        // GELU / ReLU units are off below zero (GELU dips to -0.17 at most): show how strongly each fires.
        // SwiGLU's gate can flip the sign, so there both signs rise, coloured by sign.
        const signed = cfg.activation === 'swiglu';
        const mag = (v) => (signed ? Math.abs(v) : Math.max(0, v));
        let amax = 0;
        for (let f = 0; f < F; f++) amax = Math.max(amax, mag(act.data[t * F + f]));
        amax = amax || 1;
        const geo = new THREE.CylinderGeometry(0.5, 0.5, 1, 10, 1);
        geo.translate(0, 0.5, 0);
        const pm = new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0, envMapIntensity: dark ? 0.6 : 0.6, emissive: new THREE.Color(1, 1, 1), emissiveIntensity: dark ? 1.15 : 0.42 });
        pm.onBeforeCompile = (shd) => {
          shd.fragmentShader = shd.fragmentShader.replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n#ifdef USE_COLOR\n\ttotalEmissiveRadiance *= vColor;\n#endif');
        };
        const im = new THREE.InstancedMesh(geo, pm, F);
        const mtx = new THREE.Matrix4(), col = new THREE.Color();
        const neutral = look.dark ? look.tray('mlp') : C.mlp.clone().lerp(new THREE.Color(1, 1, 1), 0.6);
        const pos = look.dark ? look.ink('mlp') : C.mlp.clone(), neg = C['heat-neg'].clone();
        for (let f = 0; f < F; f++) {
          const v = act.data[t * F + f], m = mag(v) / amax;
          const hgt = m < 0.15 ? 0 : 0.03 + 0.44 * m;               // near-silent units leave the screen clean
          const cx = (f % cols) * cell - gx / 2 + cell / 2, cz = Math.floor(f / cols) * cell - gz / 2 + cell / 2;
          mtx.makeScale(hgt ? fp : 1e-4, Math.max(1e-4, hgt), hgt ? fp : 1e-4).setPosition(cx, 0, cz);
          im.setMatrixAt(f, mtx);
          col.copy(neutral).lerp(v >= 0 ? pos : neg, Math.min(1, 0.72 + 0.4 * m));
          im.setColorAt(f, col);
        }
        im.instanceMatrix.needsUpdate = true;
        im.instanceColor.needsUpdate = true;
        im.position.set(xc, ytop + 0.032, 0);
        root.add(im);
        const fpx = proxy(new THREE.BoxGeometry(gx, 0.6, gz), V3(xc, ytop + 0.33, 0));
        part(fpx, {
          node: `neurons.${l}`, trace: `L${l}.mlp.act`,
          cellAt: (hit) => {
            const lx = hit.point.x - xc + gx / 2, lz = hit.point.z + gz / 2;
            const c = Math.floor(lx / cell), r = Math.floor(lz / cell), f = r * cols + c;
            return c >= 0 && c < cols && r >= 0 && f < F ? { r: t, x: f, c: f, band: 0 } : null;
          },
        }, [pm]);
        mod.extra.push(fpx);
        flow.fields.push({ m: im, y: ytop });
      }
    }

    // a soft luminous band that can travel up the stem (forward-pass pulse)
    function pulseShader(mat) {
      const u = { uPulseY: { value: -999 }, uPulseAmp: { value: 0 }, uPulseCol: { value: look.c.accent.clone().lerp(new THREE.Color(1, 1, 1), look.dark ? 0.35 : 0.1) },
        uHi: { value: new THREE.Vector3(0, 0, 0) }, uHiCol: { value: look.c.resid.clone().multiplyScalar(look.dark ? 0.5 : 0.3) } };
      mat.userData.u = u;
      mat.onBeforeCompile = (sh) => {
        Object.assign(sh.uniforms, u);
        sh.vertexShader = sh.vertexShader
          .replace('#include <common>', '#include <common>\nvarying float vWy;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWy = (modelMatrix * vec4(transformed, 1.0)).y;');
        sh.fragmentShader = sh.fragmentShader
          .replace('#include <common>', '#include <common>\nvarying float vWy;\nuniform float uPulseY;\nuniform float uPulseAmp;\nuniform vec3 uPulseCol;\nuniform vec3 uHi;\nuniform vec3 uHiCol;')
          .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n{ float d = (vWy - uPulseY) / 0.5; totalEmissiveRadiance += uPulseCol * uPulseAmp * exp(-d * d);\n  totalEmissiveRadiance += uHiCol * uHi.z * smoothstep(uHi.x - 0.03, uHi.x + 0.03, vWy) * (1.0 - smoothstep(uHi.y - 0.03, uHi.y + 0.03, vWy)); }');
      };
    }

    function rectOf(meshes) {
      const b = new THREE.Box3();
      meshes.forEach((m) => b.expandByObject(m));
      const r = canvas.getBoundingClientRect();
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const xx of [b.min.x, b.max.x]) for (const yy of [b.min.y, b.max.y]) for (const zz of [b.min.z, b.max.z]) {
        const p = V3(xx, yy, zz).project(cam);
        const sx = r.left + (p.x + 1) / 2 * r.width, sy = r.top + (1 - p.y) / 2 * r.height;
        x0 = Math.min(x0, sx); y0 = Math.min(y0, sy); x1 = Math.max(x1, sx); y1 = Math.max(y1, sy);
      }
      return new DOMRect(x0, y0, Math.max(8, x1 - x0), Math.max(8, y1 - y0));
    }

    // ------------------------------------------------------------------ forward pass
    // One luminous pulse climbs the stem; each part brightens softly as it passes, the MLP fields
    // rise as their layer is reached, then the prediction grows on the LM head.
    function prime(animate) {
      if (!animate || reducedMotion()) { anim = null; settle(); return; }
      for (const f of flow.fields) f.m.scale.y = 0.001;
      for (const b of flow.bars) b.m.scale.y = 0.001;
      flow.win.userData.el.style.opacity = '0';
    }
    function play(speed = 1) {
      if (reducedMotion() || !flow) return;
      prime(true);
      anim = { t0: performance.now() + 120, dur: 3700 * speed };
      engine.invalidate();
    }
    function settle() {
      if (!flow) return;
      for (const f of flow.fields) f.m.scale.y = 1;
      for (const b of flow.bars) b.m.scale.y = 1;
      flow.win.userData.el.style.opacity = '1';
      flow.rodMat.userData.u.uPulseAmp.value = 0;
      for (const p of parts) if (p.userData.g.pulse) { p.userData.g.pulse = 0; applyGlow(p); }
      engine.invalidate();
    }
    const sstep = (e0, e1, v) => { const t = clamp01((v - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
    function stepAnim(now) {
      const u = (now - anim.t0) / anim.dur;
      if (u < 0) return true;
      if (u >= 1) {
        anim = null;
        settle();
        if (fontsDirty) { fontsDirty = false; build(); if (!userMoved) goHome(0); }
        return true;
      }
      const rise = ease.inOut(clamp01((u - 0.06) / 0.7));
      const py = flow.y0 - 0.3 + (flow.y1 - flow.y0 + 0.6) * rise;
      const amp = sstep(0.02, 0.1, u) * (1 - sstep(0.74, 0.84, u));
      const uu = flow.rodMat.userData.u;
      uu.uPulseY.value = py;
      uu.uPulseAmp.value = amp * (look.dark ? 2.2 : 1.1);
      const glowAt = (y) => amp * Math.exp(-(((py - y) / 0.55) ** 2));
      for (const md of flow.mods) {
        const v = glowAt(md.y) * (look.dark ? 0.55 : 0.32);
        for (const m of [md.m, ...(md.extra || [])]) { if (m.userData.g) { m.userData.g.pulse = v; applyGlow(m); } }
      }
      if (flow.key) { flow.key.userData.g.pulse = (look.dark ? 0.5 : 0.35) * sstep(0, 0.04, u) * (1 - sstep(0.08, 0.16, u)); applyGlow(flow.key); }
      for (const f of flow.fields) f.m.scale.y = Math.max(0.001, ease.out(clamp01((py - f.y + 0.1) / 0.9)));
      for (const b of flow.bars) b.m.scale.y = Math.max(0.001, ease.out(clamp01((u - 0.74 - b.r * 0.012) / 0.2)));
      flow.win.userData.el.style.opacity = u > 0.9 ? '1' : '0';
      return true;
    }

    // ------------------------------------------------------------------ camera
    const homeDir = () => V3(Math.sin(HOME.az) * Math.cos(HOME.el), Math.sin(HOME.el), Math.cos(HOME.az) * Math.cos(HOME.el)).normalize();
    // Frame the object AND its labels: labels are fixed-size HTML, so on a phone they need relatively
    // more room than on a desktop. Solve the distance by bisection, then centre the whole composition
    // (geometry + label boxes, in pixels) in the area the atlas leaves free, and solve again.
    function fit(dir) {
      const W = Math.max(1, canvas.clientWidth), Hh = Math.max(1, canvas.clientHeight);
      host.classList.toggle('mb-compact', W < 640);
      engine.labels.render(engine.scene, cam);                       // attach labels so they can be measured
      const boxes = labels.filter((l) => l.visible && l.userData.el.isConnected && l.userData.el.offsetWidth > 0).map((l) => ({ p: l.userData.fitAt ?? l.position, w: l.userData.el.offsetWidth, h: l.userData.el.offsetHeight, cx: l.center.x, cy: l.center.y }));
      const sideM = Math.max(18, Math.round(W * 0.035)), M = { l: sideM, r: sideM, t: engine.insets.top + Math.round(Hh * 0.03), b: engine.insets.bottom + Math.round(Hh * 0.03) };
      const saved = { pos: cam.position.clone(), quat: cam.quaternion.clone() };
      const v = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3();
      const target = bounds.getCenter(V3(0, 0, 0));
      const place = (d) => { cam.position.copy(target).addScaledVector(dir, d); cam.lookAt(target); cam.updateMatrixWorld(); };
      const extent = () => {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, behind = false;
        const add = (px, py) => { x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py); };
        for (const c of framePts) { v.copy(c).project(cam); if (v.z > 1) behind = true; add((v.x + 1) / 2 * W, (1 - v.y) / 2 * Hh); }
        for (const b of boxes) {
          v.copy(b.p).project(cam);
          const px = (v.x + 1) / 2 * W, py = (1 - v.y) / 2 * Hh;
          add(px - b.cx * b.w, py - b.cy * b.h); add(px + (1 - b.cx) * b.w, py + (1 - b.cy) * b.h);
        }
        return { x0, y0, x1, y1, behind };
      };
      const fits = (e) => !e.behind && e.x0 >= M.l && e.x1 <= W - M.r && e.y0 >= M.t && e.y1 <= Hh - M.b;
      let d = 30;
      for (let pass = 0; pass < 3; pass++) {
        let lo = 2, hi = 220;
        for (let it = 0; it < 28; it++) { const mid = (lo + hi) / 2; place(mid); if (fits(extent())) hi = mid; else lo = mid; }
        d = hi;
        place(d);
        const e = extent(), wpp = (2 * d * Math.tan((cam.fov * Math.PI) / 360)) / Hh;
        const dx = (e.x0 + e.x1) / 2 - (M.l + W - M.r) / 2, dy = (e.y0 + e.y1) / 2 - (M.t + Hh - M.b) / 2;
        right.setFromMatrixColumn(cam.matrixWorld, 0);
        up.setFromMatrixColumn(cam.matrixWorld, 1);
        target.addScaledVector(right, dx * wpp).addScaledVector(up, -dy * wpp);
      }
      cam.position.copy(saved.pos);
      cam.quaternion.copy(saved.quat);
      cam.updateMatrixWorld();
      return { target, position: target.clone().addScaledVector(dir, d * 1.02) };
    }
    function goHome(duration = 0) {
      const f = fit(homeDir());
      if (duration <= 0) {
        engine.flight?.cancel();
        cam.position.copy(f.position);
        engine.controls.target.copy(f.target);
        cam.lookAt(f.target);
        engine.invalidate();
        return Promise.resolve();
      }
      return engine.flyTo({ ...f, duration });
    }
    // keep clear of the atlas' own top bar (it wraps to two rows on a phone)
    const topInset = () => Math.max(64, Math.round((document.querySelector('.atlas-top')?.getBoundingClientRect().bottom ?? 52) + 10));
    const offResize = engine.onResize(() => {
      const t = topInset();
      if (t !== engine.insets.top) { requestAnimationFrame(() => engine.setInsets({ top: t })); return; }
      if (!userMoved && bounds) goHome(0);
    });

    // ------------------------------------------------------------------ hover & click
    function setTip(html, at) {
      const e = tip.userData.el;
      if (!html) { e.classList.remove('on'); return; }
      e.innerHTML = html;
      tip.position.copy(at);
      e.classList.add('on');
    }
    // Hover changes colour only, never geometry. It re-renders only when the hovered part, cell or
    // stretch of stream changes; moving within one part costs nothing.
    function setHover(info, ev) {
      const m = info?.object?.userData.part ? info.object : null;
      const ud = m?.userData, at = m ? (ud.pickAt ? ud.pickAt(info) : ud) : null;
      const cell = m ? info.cell || ud.cell : null;
      const key = m ? `${m.id}|${at.node ?? ''}|${cell ? `${cell.r},${cell.x ?? cell.c},${cell.band ?? 0}` : ''}` : '';
      if (key === hoverKey) return;
      hoverKey = key;
      if (hot && hot !== m) { hot.userData.g.hover = 0; applyGlow(hot); hot.userData.onHot?.(null); }
      if (m && hot !== m) { m.userData.g.hover = m.userData.g.hoverK; applyGlow(m); }
      m?.userData.onHot?.(info);
      hot = m;
      const ds = canvas.dataset;
      delete ds.node; delete ds.trace; delete ds.cell;
      setTip(null);
      if (m) {
        if (ud.token !== undefined) {
          setTip(`${tokHTML(ctx.label(ud.token))} <em>· position ${ud.token}${ud.token === ctx.state.sel.token ? '' : ' · click to follow it'}</em>`, V3(m.position.x, KH + 0.18, m.position.z));
        } else {
          if (at.node) ds.node = at.node;
          if (at.trace && ctx.state.trace[at.trace]) ds.trace = at.trace;
          if (cell && !cell.fake && ds.trace) ds.cell = `${cell.r},${cell.x},${cell.band ?? 0}`;
          if (ud.tipHTML) setTip(ud.tipHTML, ud.tipAt);
        }
      }
      // the atlas reads data-node / data-trace synchronously on pointermove, but picking runs a frame
      // later: echo the move once so its hover card describes what is under the pointer now
      if (ev) canvas.dispatchEvent(new PointerEvent('pointermove', { clientX: ev.clientX, clientY: ev.clientY, bubbles: true }));
      engine.invalidate();
    }
    canvas.addEventListener('click', onClick);
    function onClick(e) {
      const info = engine.pick(e);
      const m = info?.object?.userData.part ? info.object : null;
      e.preventDefault();                      // the atlas must not also act on the canvas' data-node
      if (!m) return;
      const ud = m.userData;
      if (ud.token !== undefined) {
        if (ud.token !== ctx.state.sel.token) { pendingPass = true; ctx.store.select({ token: ud.token }); }
        return;
      }
      const at = ud.pickAt ? ud.pickAt(info) : ud;
      if (!at.node) return;
      const rect = rectOf([m]);
      const target = info.point.clone();
      const dir = cam.position.clone().sub(engine.controls.target).normalize();
      const dist = cam.position.distanceTo(engine.controls.target);
      engine.flyTo({ target, position: target.clone().addScaledVector(dir, dist * 0.55), duration: reducedMotion() ? 0 : 420 })
        .then(() => ctx.go(at.node, rectOf([m]) || rect));
    }

    // ------------------------------------------------------------------ per frame
    const offFrame = engine.onFrame((t) => {
      void t;
      if (!anim) return false;
      return stepAnim(performance.now());
    });

    // ------------------------------------------------------------------ lifecycle
    let pendingPass = false, lastTok = ctx.state.sel.token;
    const shapeSig = () => { const c = ctx.state.config; return [c.n_layer, c.n_head, c.norm_placement, c.pos_encoding, c.vocab_size].join(); };
    let lastSig = shapeSig();
    build({ animate: !opened && !reducedMotion() });
    engine.setInsets({ top: topInset(), bottom: 24, left: 0, right: 0 });
    goHome(0);
    if (!opened) {
      opened = true;
      if (!reducedMotion()) {
        host.style.opacity = '0';
        requestAnimationFrame(() => { host.style.transition = 'opacity .7s ease'; host.style.opacity = '1'; });
        play(1);
      }
    }
    // test hook (no globals): document.querySelector('.mb').__mb.play()
    host.__mb = { play: (k = 1) => play(k), state: () => ({ anim: !!anim, opened, parts: parts.length }) };
    const offs = [
      ctx.on('trace', () => {
        const sig = shapeSig();
        build();
        if (sig !== lastSig) { lastSig = sig; goHome(reducedMotion() ? 0 : 650); }
      }),
      ctx.on('select', () => {
        const tokChanged = ctx.state.sel.token !== lastTok;
        lastTok = ctx.state.sel.token;
        build();
        if (tokChanged && pendingPass) play(0.62);
        pendingPass = false;
      }),
    ];
    // if the web fonts were still loading at the first build, repaint the keycap faces once they arrive
    // (after the opening pass, never in the middle of it)
    document.fonts?.ready?.then(() => {
      if (!flow || !builtWhileFontsLoading) return;
      if (anim) { fontsDirty = true; return; }
      build();
      if (!userMoved) goHome(0);
    });

    return {
      pause: () => engine.pause(),
      resume: () => { engine.resume(); goHome(userMoved ? 600 : 0); userMoved = false; },
      destroy() {
        offFrame(); offResize(); offs.forEach((f) => f());
        canvas.removeEventListener('click', onClick);
        clear();
        tip.userData.el.remove();
        look.dispose();
        disposeTablets();
        engine.dispose();
        host.remove();
      },
    };
  },
};

// probabilities as percentages that never round a near-certainty up to 100 or a small one down to 0
function pct(p) { return p >= 0.995 && p < 1 ? (p * 100).toFixed(1) : p < 0.001 ? '<0.1' : (p * 100).toFixed(p < 0.1 ? 1 : 0); }
