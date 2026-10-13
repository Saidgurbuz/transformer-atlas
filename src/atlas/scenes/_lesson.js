// Shared teaching components. Keep the atlas's existing type, palette and motion.
import { stage2d, sceneHeader, sceneCaption, useStyle, MOTION } from '../kit.js';
import { h, texHTML, reducedMotion } from '../../ui/index.js';

useStyle('lessons', `
.ls-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; width: 1380px; }
.ls-button { border: 1px solid var(--line); border-radius: 999px; background: var(--surface); color: var(--fg-2); padding: 9px 15px; font: 600 16px/1.1 var(--font-body); cursor: pointer; transition: background .3s, border-color .3s; }
.ls-button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: var(--fg); }
.ls-button:hover { border-color: var(--line-strong); color: var(--fg); }
.ls-button:focus-visible, .ls-slot:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.ls-panel { box-sizing: border-box; border: 1px solid var(--line); border-radius: 14px; padding: 22px; background: color-mix(in srgb, var(--surface) 90%, transparent); box-shadow: var(--shadow-1); min-width: 0; }
.ls-title { font: 650 25px/1.2 var(--font-display); color: var(--fg); margin-bottom: 8px; }
.ls-meta { font: 500 16px/1.45 var(--font-mono); color: var(--fg-3); }
.ls-copy { font: 400 18px/1.5 var(--font-body); color: var(--fg-2); }
.ls-side { display: grid; gap: 18px; width: 340px; min-width: 0; }
.ls-equation { font-size: 24px; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
.ls-code { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 500 16px/1.55 var(--font-mono); color: var(--fg-2); }
.ls-source { font: 500 15px/1.4 var(--font-body); color: var(--fg-3); }
.ls-source a { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; }
.ls-pictures { display: grid; gap: 16px; align-content: start; margin-top: 22px; }
.ls-tensor { display: grid; justify-items: start; gap: 9px; min-width: 0; }
.ls-grid { display: grid; gap: 3px; }
.ls-slot { width: 27px; height: 27px; border: 1px solid color-mix(in srgb, var(--kind) 30%, var(--line)); border-radius: 4px; background: color-mix(in srgb, var(--kind) 22%, var(--surface)); display: grid; place-items: center; color: var(--fg-2); padding: 0; font: 500 10px/1 var(--font-mono); cursor: pointer; }
.ls-slot::after { content: ''; width: 5px; height: 5px; border-radius: 2px; background: var(--kind); opacity: .55; }
.ls-slot.labelled::after { display: none; }
.ls-slot.selected { outline: 2px solid var(--accent); outline-offset: 2px; background: var(--accent-soft); }
.ls-buffer { display: flex; gap: 3px; margin-top: 12px; }
.ls-buffer .ls-slot { width: 23px; height: 27px; }
.ls-check { display: inline-block; font: 600 15px/1.4 var(--font-body); color: var(--good); }
.ls-flow-box { box-sizing: border-box; border: 1px solid color-mix(in srgb, var(--kind) 35%, var(--line)); border-radius: 12px; background: var(--surface); padding: 16px 20px; transition: opacity .5s, border-color .5s, box-shadow .5s; box-shadow: var(--shadow-1); }
.ls-flow-box.on { border-color: var(--kind); box-shadow: 0 0 0 4px color-mix(in srgb, var(--kind) 12%, transparent), var(--shadow-1); }
.ls-flow-box.dim { opacity: .3; }
.ls-flow-box .ls-title { color: var(--kind); font-size: 21px; }
`);

export function lessonFrame(el, { title, sub, chips = [], captionWidth = 1050 } = {}) {
  const stage = stage2d(el), header = sceneHeader(stage, { title, sub, chips });
  const controls = h('div', { class: 'ls-controls' });
  stage.add(controls, { x: 110, y: 196 });
  const caption = sceneCaption(stage, { w: captionWidth });
  const side = h('div', { class: 'ls-side' });
  stage.add(side, { x: 1180, y: 270 });
  return { stage, header, controls, caption, side, destroy: () => stage.destroy() };
}
export function lessonButton(label, onClick, extra = {}) { return h('button', { class: 'ls-button', type: 'button', onClick, ...extra }, label); }
export function equation(tex) { return h('div', { class: 'ls-equation', html: texHTML(tex, { display: true }) }); }
export function sourceLink(title, url, locator = '') { return h('div', { class: 'ls-source' }, h('a', { href: url, target: '_blank', rel: 'noopener' }, title), locator && ` · ${locator}`); }
export function flowBox(title, sub, kind, width = 180) { return h('div', { class: 'ls-flow-box', style: { width: `${width}px`, '--kind': `var(--c-${kind})` } }, h('div', { class: 'ls-title' }, title), h('div', { class: 'ls-meta' }, sub)); }

/** Animate the same logical elements between layouts without animating numbers. */
export function morphContent(container, build, stage, { track = true } = {}) {
  const before = new Map([...container.querySelectorAll('[data-identity]')].map((e) => [e.dataset.identity, e.getBoundingClientRect()]));
  container.replaceChildren(build());
  if (reducedMotion()) return;
  for (const cell of container.querySelectorAll('[data-identity]')) {
    const a = track && before.get(cell.dataset.identity), b = cell.getBoundingClientRect();
    if (a) cell.animate([{ transform: `translate(${(a.left - b.left) / stage.scale}px,${(a.top - b.top) / stage.scale}px)` }, { transform: 'none' }], { duration: MOTION.move, easing: MOTION.ease });
    else cell.animate([{ opacity: 0 }, { opacity: 1 }], { duration: MOTION.appear, easing: MOTION.easeOut });
  }
}
