// The forward pass as a tower of tensor sheets.
//   x = token position, y = computation order (bottom = input), z = features.
// Collapsed layers show the residual checkpoints plus the attention pattern and
// MLP activations; an expanded layer shows every traced tensor of that layer.
// sync(state) animates the tower to a new trace: same-shaped sheets blend their
// values, moved sheets glide, new ones grow in, removed ones fade out.
import { THREE, themeColor, label, ease } from './engine3d.js';
import { Sheet, sheetShape } from './sheet.js';
import { infoFor, parseKey } from '../core/describe.js';

export const CELL_X = 0.34;     // token spacing
export const CELL_Z = 0.085;    // feature spacing

const SEQUENTIAL = /(^|\.)probs$|^mask$/;

export function groupOf(key) {
  if (key.startsWith('emb.') || key === 'mask' || key === 'alibi') return 'embed';
  const m = /^L(\d+)\./.exec(key);
  if (m) return `L${m[1]}`;
  return 'head';
}

// Keys shown for a collapsed layer (the residual checkpoints + one view inside each sub-layer).
function collapsedKeys(i) {
  return [`L${i}.attn.probs`, `L${i}.resid_mid`, `L${i}.mlp.act`, `L${i}.resid_out`];
}

/** Which keys are visible, in execution order, and their heights. */
export function computeLayout(trace, cfg, expanded) {
  const keys = Object.keys(trace).filter((k) => trace[k].shape.length >= 2);
  const visible = [];
  for (const k of keys) {
    const g = groupOf(k);
    if (g === 'embed') {
      if (expanded.has('embed') || k === 'emb.out') visible.push(k);
    } else if (g === 'head') {
      visible.push(k);
    } else {
      const i = Number(g.slice(1));
      if (expanded.has(g) || collapsedKeys(i).includes(k)) visible.push(k);
    }
  }
  const items = [];
  let y = 0, prevGroup = null;
  for (const k of visible) {
    const g = groupOf(k);
    const exp = expanded.has(g);
    if (prevGroup !== null) {
      y += g !== prevGroup ? 1.5 : exp ? 0.85 : 1.25;
    }
    items.push({ key: k, y, group: g });
    prevGroup = g;
  }
  // sub-layer zones (plates) for each layer
  const plates = [];
  for (let i = 0; i < cfg.n_layer; i++) {
    const g = `L${i}`, its = items.filter((it) => it.group === g);
    if (!its.length) continue;
    const mid = its.findIndex((it) => it.key === `L${i}.resid_mid`);
    const attnItems = its.slice(0, mid + 1);
    const mlpItems = its.slice(mid + 1);
    const span = (arr) => [arr[0].y - 0.45, arr[arr.length - 1].y + 0.35];
    if (attnItems.length) plates.push({ id: `${g}.attn`, layer: i, sub: 'attn', kind: 'attn', label: `Layer ${i} · attention`, range: span(attnItems), keys: attnItems.map((x) => x.key) });
    if (mlpItems.length) plates.push({ id: `${g}.mlp`, layer: i, sub: 'mlp', kind: 'mlp', label: `Layer ${i} · MLP`, range: span(mlpItems), keys: mlpItems.map((x) => x.key) });
  }
  return { items, plates, top: y };
}

export class Tower {
  constructor(engine) {
    this.engine = engine;
    this.root = new THREE.Group();
    this.root.name = 'tower';
    engine.scene.add(this.root);
    this.sheets = new Map();
    this.plates = new Map();
    this.expanded = new Set();
    this.dim = new Map();          // key -> target opacity multiplier (focus fading)
    this.layout = null;
    this.tokenLabels = [];
    this.railGroup = new THREE.Group();
    this.root.add(this.railGroup);
    this.selCol = null;
    this.arcGroup = new THREE.Group();
    this.root.add(this.arcGroup);
    this.barGroup = new THREE.Group();
    this.root.add(this.barGroup);
  }

  // ------------------------------------------------------------------ sync

  /** Rebuild or animate to match the store state. */
  sync(st, { animate = true } = {}) {
    this.state = st;
    const { trace, config: cfg, ids, tokenizer } = st;
    const T = ids.length;
    const lay = computeLayout(trace, cfg, this.expanded);
    const oldLayout = this.layout;
    this.layout = lay;
    const keep = new Set(lay.items.map((it) => it.key));
    const eng = this.engine;
    const tokenChange = this.T !== T;
    this.T = T;

    // remove sheets no longer visible (or whose shape changed)
    for (const [k, sh] of this.sheets) {
      const t = trace[k];
      if (!keep.has(k) || !t || !sh.sameShape(t)) {
        this.sheets.delete(k);
        if (animate) {
          const g = sh.group, y0 = g.position.y;
          eng.tween({ duration: 380, update: (e) => { sh.setOpacity((1 - e) * sh.opacity); g.position.y = y0 - e * 0.3; } }).then(() => sh.dispose());
        } else sh.dispose();
      }
    }
    // add / move / update
    lay.items.forEach((it, n) => {
      const t = trace[it.key];
      let sh = this.sheets.get(it.key);
      const info = infoFor(it.key);
      if (!sh) {
        sh = new Sheet({
          key: it.key, tensor: t, kind: info.kind, mode: SEQUENTIAL.test(it.key) ? 'sequential' : 'diverging',
          fixedRange: /attn\.probs$|^mask$/.test(it.key) ? [0, 1] : null,
          cellX: CELL_X, cellZ: CELL_Z, labelText: this.tagText(it.key, info),
          focus: it.group, info,
        });
        this.sheets.set(it.key, sh);
        this.root.add(sh.group);
        this.engine.addPickable(sh.top);
        sh.group.position.y = it.y;
        if (animate) {
          sh.setOpacity(0);
          sh.group.scale.set(1, 1, 0.001);
          eng.tween({ duration: 520, delay: Math.min(n * 18, 400), easing: ease.out, update: (e) => { sh.setOpacity(e * this.opacityFor(it.key)); sh.group.scale.z = Math.max(0.001, e); } });
        } else sh.setOpacity(this.opacityFor(it.key));
      } else {
        const y0 = sh.group.position.y, y1 = it.y;
        if (animate && Math.abs(y0 - y1) > 1e-3) eng.tween({ duration: 650, update: (e) => { sh.group.position.y = y0 + (y1 - y0) * e; } });
        else sh.group.position.y = y1;
        if (sh.tensor !== t) sh.setData(t, { animate, engine: eng });
        sh.top.userData.info = info;
      }
      sh.setSelectedRow(st.sel.token);
      sh.tag.visible = this.tagVisible(it.key);
    });
    this.syncPlates(animate);
    this.syncTokens(tokenizer, ids, animate && tokenChange);
    this.syncRails();
    this.syncArcs();
    this.syncBars();
    this.syncSelectedColumn();
    void oldLayout;
    eng.invalidate();
  }

  tagText(key, info) {
    const { layer } = parseKey(key);
    const short = key.replace(/^L\d+\./, '');
    return `<b>${short}</b>${layer != null ? `<span>L${layer}</span>` : ''}<i>${info.title}</i>`;
  }

  opacityFor(key) { return this.dim.get(groupOf(key)) ?? 1; }

  // Labels: every sheet of an open (or focused) group, otherwise only the checkpoints.
  tagVisible(key) {
    const g = groupOf(key);
    if (this.expanded.has(g) || this.focusGroup === g) return true;
    return ['emb.out', 'lnf', 'logits', 'probs'].includes(key) || /\.resid_out$/.test(key);
  }

  /** Fade every group except `focusGroup` (null = everything at full opacity). */
  setFocusGroup(focusGroup) {
    this.focusGroup = focusGroup;
    for (const g of ['embed', 'head', ...Array.from({ length: this.state.config.n_layer }, (_, i) => `L${i}`)]) {
      this.dim.set(g, !focusGroup || g === focusGroup ? 1 : 0.07);
    }
    for (const [k, sh] of this.sheets) {
      sh.tag.visible = this.tagVisible(k);
      const a0 = sh.opacity, a1 = this.opacityFor(k);
      this.engine.tween({ duration: 450, update: (e) => sh.setOpacity(a0 + (a1 - a0) * e) });
    }
    for (const [id, p] of this.plates) {
      const a1 = !focusGroup || id.startsWith(focusGroup + '.') ? 1 : 0.2;
      p.mat.opacity = 0.05 * a1;
      p.edges.material.opacity = 0.35 * a1;
      p.tag.userData.el.style.opacity = String(a1);
    }
  }

  // ---------------------------------------------------------------- plates

  syncPlates(animate) {
    const want = new Map(this.layout.plates.map((p) => [p.id, p]));
    for (const [id, p] of this.plates) if (!want.has(id)) { this.root.remove(p.group); p.tag.userData.el.remove(); p.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); }); this.plates.delete(id); }
    for (const [id, spec] of want) {
      const sheets = spec.keys.map((k) => this.sheets.get(k)).filter(Boolean);
      const w = Math.max(...sheets.map((s) => s.width)) + 0.9;
      const d = Math.max(...sheets.map((s) => s.depth)) + 0.9;
      const h = spec.range[1] - spec.range[0];
      const yc = (spec.range[0] + spec.range[1]) / 2;
      let p = this.plates.get(id);
      if (!p || Math.abs(p.w - w) > 1e-3 || Math.abs(p.d - d) > 1e-3 || Math.abs(p.h - h) > 1e-3) {
        if (p) { this.root.remove(p.group); p.tag.userData.el.remove(); }
        const group = new THREE.Group();
        const mat = new THREE.MeshBasicMaterial({ color: themeColor(`--c-${spec.kind}`), transparent: true, opacity: 0.05, depthWrite: false, side: THREE.BackSide });
        const box = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
        box.userData = { traceKey: null, kind: spec.kind, focus: `L${spec.layer}`, plate: spec, label: spec.label };
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({ color: themeColor(`--c-${spec.kind}`), transparent: true, opacity: 0.35 }));
        const tag = label(`${spec.label}`, { className: 'plate-tag', kind: spec.kind });
        tag.position.set(w / 2 + 0.2, h / 2 - 0.15, d / 2);
        tag.center.set(0, 0.5);
        group.add(box, edges, tag);
        group.position.y = yc;
        this.root.add(group);
        p = { group, box, mat, edges, tag, w, d, h, spec };
        this.plates.set(id, p);
        if (animate) { group.scale.y = 0.01; this.engine.tween({ duration: 500, easing: ease.out, update: (e) => { group.scale.y = Math.max(0.01, e); } }); }
      } else {
        const y0 = p.group.position.y;
        if (animate) this.engine.tween({ duration: 650, update: (e) => { p.group.position.y = y0 + (yc - y0) * e; } });
        else p.group.position.y = yc;
        p.spec = spec;
      }
    }
  }

  // ---------------------------------------------------------------- tokens

  xOf(t) { return (t - (this.T - 1) / 2) * CELL_X; }

  frontZ() { let d = 0; for (const s of this.sheets.values()) d = Math.max(d, s.depth); return d / 2; }

  syncTokens(tokenizer, ids, animate) {
    for (const l of this.tokenLabels) { l.removeFromParent(); l.userData.el.remove(); }
    this.tokenLabels = [];
    const z = (this.sheets.get('emb.out')?.depth ?? 4) / 2 + 0.45;
    const y0 = this.layout.items[0]?.y ?? 0;
    ids.forEach((id, t) => {
      const l = label(`<span>${escapeHtml(tokenizer.label(id))}</span><small>${t}</small>`, { className: `token-tag${ids.length > 24 ? ' compact' : ''}`, interactive: true });
      l.position.set(this.xOf(t), y0 - 0.35, z);
      l.userData.el.addEventListener('click', () => this.onTokenClick?.(t));
      l.userData.el.classList.toggle('sel', t === this.state.sel.token);
      if (animate) { l.userData.el.style.opacity = '0'; setTimeout(() => { l.userData.el.style.opacity = ''; }, 30 + t * 12); }
      this.root.add(l);
      this.tokenLabels.push(l);
    });
  }

  // ----------------------------------------------------------------- rails

  syncRails() {
    this.railGroup.clear();
    const items = this.layout.items;
    if (!items.length) return;
    const y0 = items[0].y, y1 = items[items.length - 1].y;
    const pos = [];
    for (let t = 0; t < this.T; t++) { const x = this.xOf(t); pos.push(x, y0, 0, x, y1, 0); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({ color: themeColor('--c-resid'), transparent: true, opacity: 0.12 });
    this.railGroup.add(new THREE.LineSegments(g, m));
  }

  syncSelectedColumn() {
    if (this.selCol) { this.root.remove(this.selCol); this.selCol.geometry.dispose(); this.selCol.material.dispose(); }
    const items = this.layout.items;
    if (!items.length) return;
    const t = this.state.sel.token;
    const h = items[items.length - 1].y - items[0].y + 1.2;
    let d = 0; for (const s of this.sheets.values()) d = Math.max(d, s.depth);
    const geo = new THREE.BoxGeometry(CELL_X * 0.96, h, d + 0.6);
    const mat = new THREE.MeshBasicMaterial({ color: themeColor('--accent'), transparent: true, opacity: 0.06, depthWrite: false });
    this.selCol = new THREE.Mesh(geo, mat);
    this.selCol.position.set(this.xOf(t), (items[0].y + items[items.length - 1].y) / 2, 0);
    this.root.add(this.selCol);
  }

  /** Selection changed without a new trace. */
  select(sel) {
    for (const sh of this.sheets.values()) sh.setSelectedRow(sel.token);
    for (const l of this.tokenLabels) l.userData.el.classList.toggle('sel', this.tokenLabels.indexOf(l) === sel.token);
    this.syncArcs();
    this.syncBars();
    this.syncSelectedColumn();
    this.engine.invalidate();
  }

  // ------------------------------------------------------------------ arcs

  /** Attention arcs into the selected token, for the selected head, in every layer. */
  syncArcs() {
    for (const c of [...this.arcGroup.children]) { c.geometry?.dispose(); c.material?.dispose(); }
    this.arcGroup.clear();
    if (this.showArcs === false) return;
    const st = this.state, { trace, sel, config: cfg } = st;
    const q = sel.token;
    for (let i = 0; i < cfg.n_layer; i++) {
      const probs = trace[`L${i}.attn.probs`];
      const target = this.sheets.get(`L${i}.resid_mid`);
      const plate = this.layout.plates.find((p) => p.id === `L${i}.attn`);
      if (!probs || !target || !plate) continue;
      const T = probs.shape[1], h = Math.min(sel.head, probs.shape[0] - 1);
      const yFrom = plate.range[0] + 0.1, yTo = target.group.position.y;
      const color = themeColor('--c-attn');
      for (let j = 0; j < T; j++) {
        const w = probs.data[h * T * T + q * T + j];
        if (!(w > 0.02)) continue;
        const a = new THREE.Vector3(this.xOf(j), yFrom, 0.25);
        const b = new THREE.Vector3(this.xOf(q), yTo, 0.25);
        const mid = a.clone().lerp(b, 0.5);
        mid.y += 0.5 + Math.abs(q - j) * CELL_X * 0.18;
        mid.z += 0.9;
        const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
        const geo = new THREE.TubeGeometry(curve, 28, 0.012 + 0.05 * w, 6, false);
        const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.25 + 0.7 * w, depthWrite: false });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.userData = { traceKey: `L${i}.attn.probs`, kind: 'attn', focus: `L${i}`, arc: { layer: i, head: h, q, k: j, w } };
        const total = geo.index.count;
        geo.setDrawRange(0, 0);
        this.engine.tween({ duration: 650, delay: i * 120, easing: ease.out, update: (e) => geo.setDrawRange(0, Math.floor(total * e / 3) * 3) });
        this.arcGroup.add(mesh);
      }
    }
  }

  // ------------------------------------------------------------------ bars

  /** Next-token distribution of the selected token as bars above the probs sheet. */
  syncBars() {
    for (const c of [...this.barGroup.children]) { c.geometry?.dispose(); c.material?.dispose(); if (c.userData.el) c.userData.el.remove(); }
    this.barGroup.clear();
    const sh = this.sheets.get('probs');
    if (!sh) return;
    const st = this.state, t = st.sel.token, V = sh.cols;
    const row = sh.tensor.data.subarray(t * V, (t + 1) * V);
    const order = [...row.keys()].sort((a, b) => row[b] - row[a]);
    const color = themeColor('--c-head');
    const y = sh.group.position.y + sh.thickness / 2;
    for (let v = 0; v < V; v++) {
      const p = row[v];
      const hgt = Math.max(0.01, p * 3);
      const geo = new THREE.BoxGeometry(CELL_X * 0.7, hgt, CELL_Z * 0.8);
      const mat = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.35 + 0.65 * Math.sqrt(p) });
      const bar = new THREE.Mesh(geo, mat);
      bar.position.set(this.xOf(t), y + hgt / 2, sh.zOf(0, v));
      bar.userData = { traceKey: 'probs', kind: 'head', focus: 'head', bar: { token: v, p } };
      bar.scale.y = 0.001;
      this.engine.tween({ duration: 600, delay: order.indexOf(v) * 8, easing: ease.outBack, update: (e) => { bar.scale.y = Math.max(0.001, e); bar.position.y = y + (hgt * e) / 2; } });
      this.barGroup.add(bar);
      this.engine.addPickable(bar);
    }
    order.slice(0, 3).forEach((v, rank) => {
      const l = label(`<b>${escapeHtml(st.tokenizer.label(v))}</b> ${(row[v] * 100).toFixed(1)}%`, { className: `bar-tag rank${rank}`, kind: 'head' });
      l.position.set(this.xOf(t) + 0.25, y + row[v] * 3 + 0.25, sh.zOf(0, v));
      l.center.set(0, 0.5);
      this.barGroup.add(l);
    });
  }

  // ---------------------------------------------------------------- bounds

  boxOf(keys) {
    const box = new THREE.Box3();
    for (const k of keys) {
      const sh = this.sheets.get(k);
      if (!sh) continue;
      const y = sh.group.position.y;
      box.expandByPoint(new THREE.Vector3(-sh.width / 2, y - 0.3, -sh.depth / 2));
      box.expandByPoint(new THREE.Vector3(sh.width / 2, y + 0.3, sh.depth / 2));
    }
    return box;
  }

  boxOfGroup(g) { return this.boxOf(this.layout.items.filter((it) => it.group === g).map((it) => it.key)); }

  boxAll() { return this.boxOf(this.layout.items.map((it) => it.key)); }

  dispose() {
    for (const s of this.sheets.values()) s.dispose();
    for (const l of this.tokenLabels) l.userData.el.remove();
    this.root.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); if (o.userData?.el) o.userData.el.remove(); });
    this.root.removeFromParent();
  }
}

export function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
export { sheetShape };
