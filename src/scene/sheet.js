// A traced tensor drawn as a horizontal slab: x = token position, z = feature
// (or key position / head band), colour = value. Values live in a float texture
// and a shader does the colouring, so value tweens, reveal sweeps, hover and
// selection highlights are uniform changes with no CPU redraw.
import { THREE, themeColor, label } from './engine3d.js';

const GAP = 3.0e38;          // texel sentinel for the gap between head bands
const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FRAG = /* glsl */`
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform float mixT;
uniform vec2 grid;          // (tokens, feature columns incl. gaps)
uniform int mode;           // 0 diverging, 1 sequential
uniform float rangeFrom, rangeTo, loFrom, loTo, hiFrom, hiTo;
uniform vec3 cNeg, cMid, cPos, cLo, cHi, cMask, cAccent;
uniform float reveal;       // 0..1, fraction of tokens shown (left to right)
uniform float opacity;
uniform vec2 hover;         // (token, column) or (-1, -1)
uniform float selRow;       // selected token or -1
uniform float glow;         // 0..1 pulse
varying vec2 vUv;

void main() {
  vec2 g = vUv * grid;
  vec2 cell = floor(g);
  vec2 f = fract(g);
  vec2 tc = (cell + 0.5) / grid;
  float b = texture2D(tTo, tc).r;
  if (b > 1.0e38) discard;
  float rowFrac = (cell.x + 0.5) / grid.x;
  float lead = reveal - rowFrac;
  if (lead < 0.0) discard;
  float a = texture2D(tFrom, tc).r;
  float range = mix(rangeFrom, rangeTo, mixT);
  float lo = mix(loFrom, loTo, mixT), hi = mix(hiFrom, hiTo, mixT);
  vec3 col;
  float strength = 1.0;     // small values are drawn more transparent so the tower stays see-through
  bool masked = b < -1.0e37 && (mixT > 0.5 || a < -1.0e37);
  if (masked) {
    float hatch = step(0.5, fract((g.x + g.y) * 2.0));
    col = mix(cMid, cMask, 0.25 + 0.35 * hatch);
  } else {
    float va = a < -1.0e37 ? b : a;
    float vb = b < -1.0e37 ? va : b;
    float v = mix(va, vb, mixT);
    if (mode == 0) {
      float t = clamp(v / max(range, 1e-9), -1.0, 1.0);
      float e = pow(abs(t), 0.75);
      col = t < 0.0 ? mix(cMid, cNeg, e) : mix(cMid, cPos, e);
      strength = mix(0.28, 1.0, e);
    } else {
      float t = clamp((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0);
      col = mix(cLo, cHi, pow(t, 0.75));
      strength = mix(0.22, 1.0, pow(t, 0.6));
    }
  }
  // thin gaps between cells once a cell spans enough pixels
  vec2 px = fwidth(g);
  float gridOn = 1.0 - smoothstep(0.12, 0.3, max(px.x, px.y));
  vec2 inner = smoothstep(vec2(0.0), px * 1.2, f) * smoothstep(vec2(0.0), px * 1.2, 1.0 - f);
  float cellMask = mix(1.0, min(inner.x, inner.y), gridOn * 0.85);
  // leading edge of the reveal sweep glows
  float edge = 1.0 - smoothstep(0.0, 2.5 / grid.x, lead);
  col = mix(col, cAccent, edge * 0.55);
  // selected token: soft band
  if (abs(cell.x - selRow) < 0.5) col = mix(col, cAccent, 0.18);
  // hovered cell / token
  if (abs(cell.x - hover.x) < 0.5) {
    col = mix(col, cAccent, abs(cell.y - hover.y) < 0.5 ? 0.45 : 0.08);
  }
  col += glow * 0.25;
  float alpha = opacity * cellMask * max(strength, edge);
  if (alpha < 0.02) discard;
  gl_FragColor = vec4(col, alpha);
  #include <colorspace_fragment>
}`;

function quantileAbs(data, q) {
  const vals = [];
  for (let i = 0; i < data.length; i += Math.max(1, Math.floor(data.length / 4000))) {
    const v = data[i];
    if (Number.isFinite(v)) vals.push(Math.abs(v));
  }
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.floor(q * (vals.length - 1)))] || vals[vals.length - 1] || 1;
}

/**
 * Describe how a tensor maps onto the sheet.
 *   2-D [T, X]      -> bands 1, cols X
 *   3-D [B, T, X]   -> B bands of X columns with 1-column gaps
 *   1-D / scalar    -> not drawable
 */
export function sheetShape(shape) {
  if (shape.length === 2) return { rows: shape[0], bands: 1, cols: shape[1] };
  if (shape.length === 3) return { rows: shape[1], bands: shape[0], cols: shape[2] };
  return null;
}

export class Sheet {
  /**
   * @param {object} o  { key, tensor, kind, mode: 'diverging'|'sequential', fixedRange?: [lo, hi],
   *                      cellX, cellZ, thickness, labelText, focus, info }
   */
  constructor(o) {
    this.key = o.key;
    this.kind = o.kind;
    this.mode = o.mode;
    this.fixedRange = o.fixedRange || null;
    this.cellX = o.cellX;
    this.cellZ = o.cellZ;
    this.thickness = o.thickness ?? 0.06;
    const sh = sheetShape(o.tensor.shape);
    Object.assign(this, sh);
    this.gap = this.bands > 1 ? 1 : 0;
    this.totalCols = this.bands * this.cols + (this.bands - 1) * this.gap;
    this.width = this.rows * this.cellX;
    this.depth = this.totalCols * this.cellZ;
    this.group = new THREE.Group();
    this.group.name = `sheet:${this.key}`;

    this.texFrom = this.makeTexture();
    this.texTo = this.makeTexture();
    this.uniforms = {
      tFrom: { value: this.texFrom }, tTo: { value: this.texTo }, mixT: { value: 1 },
      grid: { value: new THREE.Vector2(this.rows, this.totalCols) },
      mode: { value: this.mode === 'sequential' ? 1 : 0 },
      rangeFrom: { value: 1 }, rangeTo: { value: 1 }, loFrom: { value: 0 }, loTo: { value: 0 }, hiFrom: { value: 1 }, hiTo: { value: 1 },
      cNeg: { value: new THREE.Color() }, cMid: { value: new THREE.Color() }, cPos: { value: new THREE.Color() },
      cLo: { value: new THREE.Color() }, cHi: { value: new THREE.Color() }, cMask: { value: new THREE.Color() }, cAccent: { value: new THREE.Color() },
      reveal: { value: 1 }, opacity: { value: 1 }, hover: { value: new THREE.Vector2(-1, -1) }, selRow: { value: -1 }, glow: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms, transparent: true, side: THREE.DoubleSide, extensions: { derivatives: true } });
    const top = new THREE.Mesh(new THREE.PlaneGeometry(this.width, this.depth), mat);
    top.rotation.x = -Math.PI / 2;
    top.position.y = this.thickness / 2 + 0.002;
    top.userData = { traceKey: this.key, kind: this.kind, focus: o.focus, info: o.info, sheet: this, cellAt: (hit) => this.cellAt(hit) };
    this.top = top;

    this.bodyMat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.22, depthWrite: false });
    const body = new THREE.Mesh(new THREE.BoxGeometry(this.width + 0.04, this.thickness, this.depth + 0.04), this.bodyMat);
    this.edgeMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55 });
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry), this.edgeMat);
    this.group.add(body, edges, top);

    this.tag = label(o.labelText || this.key, { className: 'sheet-tag', kind: this.kind });
    this.tag.position.set(-this.width / 2 - 0.35, 0, this.depth / 2);
    this.tag.center.set(1, 0.5);
    this.group.add(this.tag);
    this.opacity = 1;
    this.applyTheme();
    this.setData(o.tensor, { animate: false });
  }

  makeTexture() {
    const tex = new THREE.DataTexture(new Float32Array(this.rows * this.totalCols), this.rows, this.totalCols, THREE.RedFormat, THREE.FloatType);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    return tex;
  }

  applyTheme() {
    const u = this.uniforms;
    u.cNeg.value.copy(themeColor('--heat-neg'));
    u.cMid.value.copy(themeColor('--heat-mid'));
    u.cPos.value.copy(themeColor('--heat-pos'));
    u.cLo.value.copy(themeColor('--heat-lo'));
    u.cHi.value.copy(themeColor(this.kind === 'head' || this.kind === 'attn' || this.kind === 'mask' ? `--c-${this.kind === 'mask' ? 'attn' : this.kind}` : '--heat-hi'));
    u.cMask.value.copy(themeColor('--c-mask'));
    u.cAccent.value.copy(themeColor('--accent'));
    const kc = themeColor(`--c-${this.kind}`);
    this.bodyMat.color.copy(kc);
    this.edgeMat.color.copy(kc);
  }

  // Copy tensor values into the texture layout (feature 0 at the front).
  fill(tex, tensor) {
    const d = tex.image.data, { rows, bands, cols, gap, totalCols } = this;
    const src = tensor.data;
    for (let zc = 0; zc < totalCols; zc++) {
      const b = Math.floor(zc / (cols + gap)), x = zc % (cols + gap);
      for (let t = 0; t < rows; t++) {
        let v;
        if (x >= cols) v = GAP;
        else v = bands > 1 ? src[b * rows * cols + t * cols + x] : src[t * cols + x];
        d[zc * rows + t] = v === undefined ? 0 : v;
      }
    }
    tex.needsUpdate = true;
  }

  computeRange(tensor) {
    if (this.mode === 'sequential') {
      if (this.fixedRange) return { lo: this.fixedRange[0], hi: this.fixedRange[1], range: 1 };
      let hi = 0;
      for (const v of tensor.data) if (Number.isFinite(v)) hi = Math.max(hi, v);
      return { lo: 0, hi: hi || 1, range: 1 };
    }
    return { lo: 0, hi: 1, range: quantileAbs(tensor.data, 0.99) };
  }

  /** New values for the same shape; animates a colour blend when `engine` is given. */
  setData(tensor, { animate = true, engine = null, duration = 650 } = {}) {
    this.tensor = tensor;
    const u = this.uniforms;
    const r = this.computeRange(tensor);
    if (!animate || !engine) {
      this.fill(this.texTo, tensor);
      u.mixT.value = 1;
      u.rangeFrom.value = u.rangeTo.value = r.range;
      u.loFrom.value = u.loTo.value = r.lo;
      u.hiFrom.value = u.hiTo.value = r.hi;
      return Promise.resolve();
    }
    // current target becomes the start; new values become the target
    [this.texFrom, this.texTo] = [this.texTo, this.texFrom];
    u.tFrom.value = this.texFrom;
    u.tTo.value = this.texTo;
    this.fill(this.texTo, tensor);
    u.rangeFrom.value = u.rangeTo.value; u.loFrom.value = u.loTo.value; u.hiFrom.value = u.hiTo.value;
    u.rangeTo.value = r.range; u.loTo.value = r.lo; u.hiTo.value = r.hi;
    u.mixT.value = 0;
    this.blend?.cancel();
    this.blend = engine.tween({ duration, update: (e) => { u.mixT.value = e; } });
    return this.blend;
  }

  sameShape(tensor) {
    const sh = sheetShape(tensor.shape);
    return sh && sh.rows === this.rows && sh.bands === this.bands && sh.cols === this.cols;
  }

  setReveal(p) { this.uniforms.reveal.value = p; }

  setOpacity(a) {
    this.opacity = a;
    this.uniforms.opacity.value = a;
    this.bodyMat.opacity = 0.22 * a;
    this.edgeMat.opacity = 0.55 * a;
    this.top.userData.opacity = a;
    this.tag.userData.el.style.opacity = String(Math.min(1, a * 1.2));
    this.group.visible = a > 0.01;
  }

  setHover(t, zc) { this.uniforms.hover.value.set(t ?? -1, zc ?? -1); }
  setSelectedRow(t) { this.uniforms.selRow.value = t ?? -1; }

  pulse(engine) {
    const u = this.uniforms;
    return engine.tween({ duration: 700, update: (e) => { u.glow.value = Math.sin(e * Math.PI) * 0.9; } });
  }

  /** Map a raycast hit on the top face to {r (token), band, x (feature), zc, value}. */
  cellAt(hit) {
    if (!hit.uv) return null;
    const t = Math.min(this.rows - 1, Math.floor(hit.uv.x * this.rows));
    const zc = Math.min(this.totalCols - 1, Math.floor(hit.uv.y * this.totalCols));
    const band = Math.floor(zc / (this.cols + this.gap)), x = zc % (this.cols + this.gap);
    if (x >= this.cols) return null;
    const src = this.tensor.data;
    const value = this.bands > 1 ? src[band * this.rows * this.cols + t * this.cols + x] : src[t * this.cols + x];
    return { r: t, band, x, zc, value };
  }

  /** World-space z of feature column (band, x) centre. */
  zOf(band, x) {
    const zc = band * (this.cols + this.gap) + x;
    return this.depth / 2 - (zc + 0.5) * this.cellZ;
  }

  dispose() {
    this.blend?.cancel();
    this.texFrom.dispose();
    this.texTo.dispose();
    this.group.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    this.tag.userData.el.remove();
    this.group.removeFromParent();
  }
}
