// Animation helpers that respect prefers-reduced-motion.
import { h } from './dom.js';

export const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export const ease = {
  linear: (t) => t,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t) => 1 - Math.pow(1 - t, 3),
};

// tween({duration, onUpdate(t), onDone}) -> cancel()
export function tween({ duration = 400, easing = ease.inOut, onUpdate, onDone }) {
  if (reducedMotion()) { onUpdate(1); onDone?.(); return () => {}; }
  let raf, start;
  const step = (now) => {
    start ??= now;
    const t = Math.min(1, (now - start) / duration);
    onUpdate(easing(t));
    if (t < 1) raf = requestAnimationFrame(step); else onDone?.();
  };
  raf = requestAnimationFrame(step);
  return () => cancelAnimationFrame(raf);
}

// Continuous animation loop: loop((dt, time) => ...) -> stop()
export function loop(fn) {
  let raf, last;
  const tick = (now) => { const dt = last ? (now - last) / 1000 : 0; last = now; fn(dt, now / 1000); raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

/**
 * Step-through control used by every staged explanation.
 *   const pb = playbar({ steps: ['project', 'score', 'mask'], onStep: (i) => ..., interval: 1400 });
 * Includes previous / play-pause / next buttons and clickable step pills.
 */
export function playbar({ steps, onStep, interval = 1600, initial = 0, loop: loops = false }) {
  let i = initial, timer = null;
  const prev = h('button', { type: 'button', class: 'btn icon', title: 'Previous step', 'aria-label': 'Previous step', onClick: () => go(i - 1) }, '‹');
  const play = h('button', { type: 'button', class: 'btn', title: 'Play all steps', onClick: () => (timer ? stop() : start()) }, '▶ Play');
  const next = h('button', { type: 'button', class: 'btn icon', title: 'Next step', 'aria-label': 'Next step', onClick: () => go(i + 1) }, '›');
  const pills = steps.map((s, k) => h('button', { type: 'button', class: 'step-dot', onClick: () => { stop(); go(k); } }, `${k + 1}. ${s}`));
  const el = h('div', { class: 'playbar' }, prev, play, next, h('div', { class: 'steps' }, pills));

  function paint() {
    pills.forEach((p, k) => { p.toggleAttribute('aria-current', k === i); if (k === i) p.setAttribute('aria-current', 'step'); p.classList.toggle('done', k < i); });
    prev.disabled = i === 0 && !loops;
    next.disabled = i === steps.length - 1 && !loops;
  }
  function go(k) {
    if (loops) k = (k + steps.length) % steps.length;
    k = Math.max(0, Math.min(steps.length - 1, k));
    i = k;
    paint();
    onStep?.(i);
  }
  function start() {
    play.textContent = '❚❚ Pause';
    if (i >= steps.length - 1 && !loops) go(0);
    timer = setInterval(() => {
      if (i >= steps.length - 1 && !loops) { stop(); return; }
      go(i + 1);
    }, reducedMotion() ? Math.max(interval, 2000) : interval);
  }
  function stop() { clearInterval(timer); timer = null; play.textContent = '▶ Play'; }
  paint();
  return { el, go, stop, start, get index() { return i; }, destroy: stop };
}
