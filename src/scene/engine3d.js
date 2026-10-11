// WebGL scene host for the 3D explorer: renderer, camera + orbit controls,
// HTML labels, picking (hover / click / double-click / right-click), a tween
// clock, camera fly-to, and theme-aware colours.
import * as THREE from '../vendor/three/three.module.js';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from '../vendor/three/CSS2DRenderer.js';
import { cssVar, onThemeChange, reducedMotion } from '../ui/index.js';

export { THREE, CSS2DObject };

export const ease = {
  linear: (t) => t,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t) => 1 - Math.pow(1 - t, 3),
  outBack: (t) => { const c1 = 1.4, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
};

export function themeColor(name, fallback = '#888888') {
  const v = cssVar(name);
  const c = new THREE.Color();
  try { c.setStyle(v || fallback); } catch { c.setStyle(fallback); }
  return c;
}

export class SceneEngine {
  /**
   * @param {HTMLElement} container  element that receives the canvas (position: relative)
   * @param {{onHover?, onPick?, onDouble?, onContext?}} handlers  receive (info|null, event)
   */
  constructor(container, handlers = {}) {
    this.container = container;
    this.handlers = handlers;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.className = 'scene-canvas';
    container.append(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'scene-labels';
    container.append(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 4000);
    this.camera.position.set(18, 14, 26);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.9;
    this.controls.addEventListener('change', () => this.invalidate());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(10, 30, 20);
    this.scene.add(sun);

    this.pickables = new Set();
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.hovered = null;
    this.tweens = new Set();
    this.frameHooks = new Set();
    this.dirty = true;
    this.running = true;
    this.insets = { left: 0, right: 0, top: 0, bottom: 0 };
    this.renderFn = null;          // optional replacement for renderer.render (post-processing)
    this.resizeHooks = new Set();

    this.applyTheme();
    this.offTheme = onThemeChange(() => { this.applyTheme(); this.handlers.onTheme?.(); });

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.resize();
    this.bindPointer();
    const loop = (now) => {
      if (!this.running) return;
      this.raf = requestAnimationFrame(loop);
      this.tick(now);
    };
    this.raf = requestAnimationFrame(loop);
  }

  applyTheme() {
    const bg = themeColor('--bg', '#0d1016');
    this.scene.background = bg;
    this.scene.fog = new THREE.Fog(bg, 60, 260);
    this.isDark = (bg.r + bg.g + bg.b) / 3 < 0.5;
    this.invalidate();
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    const { left, right, top, bottom } = this.insets;
    // shift the projection so the scene centres in the area not covered by HUD panels
    this.camera.setViewOffset(w, h, -(left - right) / 2, -(top - bottom) / 2, w, h);
    this.camera.updateProjectionMatrix();
    for (const fn of this.resizeHooks ?? []) fn(w, h);
    this.invalidate();
  }

  /** Replace the default render call, e.g. with an EffectComposer. null restores it. */
  setRenderFn(fn) { this.renderFn = fn; this.invalidate(); }

  /** fn(width, height) after every resize. Returns remover. */
  onResize(fn) { this.resizeHooks.add(fn); return () => this.resizeHooks.delete(fn); }

  /** Pixels covered by overlay panels on each side; framing centres in the rest. */
  setInsets(insets) { this.insets = { ...this.insets, ...insets }; this.resize(); }

  invalidate() { this.dirty = true; }

  tick(now) {
    const t = now / 1000;
    if (this.tweens.size) {
      for (const tw of [...this.tweens]) {
        tw.start ??= now;
        const p = tw.duration <= 0 ? 1 : Math.min(1, (now - tw.start - (tw.delay || 0)) / tw.duration);
        if (p < 0) continue;
        tw.update(tw.ease(p), p);
        if (p >= 1) { this.tweens.delete(tw); tw.resolve(); }
      }
      this.dirty = true;
    }
    for (const fn of this.frameHooks) if (fn(t) !== false) this.dirty = true;
    const moved = this.controls.update();
    if (this.dirty || moved) {
      if (this.renderFn) this.renderFn(this.scene, this.camera);
      else this.renderer.render(this.scene, this.camera);
      this.labels.render(this.scene, this.camera);
      this.dirty = false;
    }
  }

  /** Tween over `duration` ms. Returns a promise with .cancel(). update(easedT, rawT) */
  tween({ duration = 600, delay = 0, easing = ease.inOut, update }) {
    let resolve;
    const p = new Promise((r) => { resolve = r; });
    if (reducedMotion() || duration <= 0) { update(1, 1); resolve(); this.invalidate(); p.cancel = () => {}; return p; }
    const tw = { duration, delay, ease: easing, update, resolve };
    this.tweens.add(tw);
    p.cancel = () => { this.tweens.delete(tw); resolve(); };
    return p;
  }

  /** Per-frame callback fn(timeSeconds); return false when nothing changed. Returns remover. */
  onFrame(fn) { this.frameHooks.add(fn); return () => this.frameHooks.delete(fn); }

  /** Smoothly move camera and orbit target. */
  flyTo({ position, target, duration = 900 }) {
    const p0 = this.camera.position.clone(), t0 = this.controls.target.clone();
    const p1 = position.clone(), t1 = target.clone();
    this.flight?.cancel();
    this.flight = this.tween({
      duration,
      update: (e) => {
        this.camera.position.lerpVectors(p0, p1, e);
        this.controls.target.lerpVectors(t0, t1, e);
      },
    });
    return this.flight;
  }

  /** Camera position that frames `box` from direction `dir` (unit vector from target to camera). */
  framing(box, dir = new THREE.Vector3(0.55, 0.45, 0.85), padding = 1.15) {
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(0.5, size.length() / 2) * padding;
    const fov = (this.camera.fov * Math.PI) / 180;
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    const { left, right, top, bottom } = this.insets;
    const freeW = Math.max(120, w - left - right), freeH = Math.max(120, h - top - bottom);
    const fitH = (radius / Math.sin(fov / 2)) * (h / freeH);
    const fitW = (radius / Math.sin(fov / 2)) * (h / freeH) * Math.max(1, (freeH / freeW));
    const dist = Math.max(fitH, fitW);
    return { target: center, position: center.clone().add(dir.clone().normalize().multiplyScalar(dist)) };
  }

  frame(box, opts = {}) {
    const dir = opts.dir ?? this.camera.position.clone().sub(this.controls.target).normalize();
    return this.flyTo({ ...this.framing(box, dir, opts.padding), duration: opts.duration ?? 900 });
  }

  // ------------------------------------------------------------------ picking

  addPickable(obj) { this.pickables.add(obj); }
  removePickable(obj) { this.pickables.delete(obj); if (this.hovered?.object === obj) this.hovered = null; }

  pick(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const objs = [...this.pickables].filter((o) => o.visible && isVisible(o));
    const hits = this.raycaster.intersectObjects(objs, false);
    for (const hit of hits) {
      const ud = hit.object.userData;
      if (ud.opacity !== undefined && ud.opacity < 0.2) continue;
      const cell = ud.cellAt ? ud.cellAt(hit) : null;
      return { object: hit.object, point: hit.point, traceKey: ud.traceKey, kind: ud.kind, focus: ud.focus, label: ud.label, cell, data: ud };
    }
    return null;
  }

  bindPointer() {
    const el = this.renderer.domElement;
    let down = null, pending = null;
    const onMove = (ev) => {
      pending = ev;
      if (this._moveQueued) return;
      this._moveQueued = true;
      requestAnimationFrame(() => {
        this._moveQueued = false;
        if (!pending) return;
        const info = this.pick(pending);
        const same = info && this.hovered && info.object === this.hovered.object && info.cell?.r === this.hovered.cell?.r && info.cell?.c === this.hovered.cell?.c;
        if (!same) { this.hovered = info; this.handlers.onHover?.(info, pending); }
        else this.handlers.onHoverMove?.(info, pending);
        el.style.cursor = info ? 'pointer' : 'grab';
      });
    };
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerleave', () => { pending = null; this.hovered = null; this.handlers.onHover?.(null, null); });
    el.addEventListener('pointerdown', (ev) => { down = { x: ev.clientX, y: ev.clientY, b: ev.button }; });
    el.addEventListener('pointerup', (ev) => {
      if (!down || ev.button !== 0) return;
      const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y);
      down = null;
      if (moved > 5) return;
      this.handlers.onPick?.(this.pick(ev), ev);
    });
    el.addEventListener('dblclick', (ev) => this.handlers.onDouble?.(this.pick(ev), ev));
    el.addEventListener('contextmenu', (ev) => {
      ev.preventDefault();
      this.handlers.onContext?.(this.pick(ev), ev);
    });
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.offTheme();
    this.controls.dispose();
    this.scene.traverse((o) => {
      o.geometry?.dispose?.();
      const m = o.material;
      (Array.isArray(m) ? m : m ? [m] : []).forEach((mm) => { mm.map?.dispose?.(); mm.dispose?.(); });
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labels.domElement.remove();
  }
}

function isVisible(o) {
  for (let p = o; p; p = p.parent) if (!p.visible) return false;
  return true;
}

/** HTML label attached to a 3D position. */
export function label(html, { className = '', kind = null, interactive = false } = {}) {
  const div = document.createElement('div');
  div.className = `scene-label ${className}`;
  if (kind) div.style.setProperty('--kind', `var(--c-${kind})`);
  if (interactive) div.style.pointerEvents = 'auto';
  div.innerHTML = html;
  const obj = new CSS2DObject(div);
  obj.userData.el = div;
  return obj;
}
