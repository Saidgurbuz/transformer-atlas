import { h, reducedMotion } from '../../ui/index.js';
import { useStyle, MOTION } from '../kit.js';
import { EXERCISES } from '../../lessons/exercises.js';
import { runExercise } from '../../lessons/exercise-runner.js';
import { lessonButton, equation } from './_lesson.js';
useStyle('exercises', `
.ex-doc .k-doc-inner { min-width: 0; gap: 24px; }
.ex-doc { padding-top: 100px; }
.ex-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.ex-layout { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(0, 1fr); gap: 24px; }
.ex-panel { padding: 26px; min-width: 0; }
.ex-panel h2 { margin: 0 0 16px; }
.ex-prompt { font: 400 18px/1.6 var(--font-body); color: var(--fg-2); margin-bottom: 24px; }
.ex-options { display: grid; gap: 12px; margin: 24px 0; }
.ex-option { display: flex; align-items: baseline; gap: 12px; padding: 16px; border-radius: 10px; border: 1px solid var(--line); background: var(--surface); cursor: pointer; font: 500 16px/1.5 var(--font-mono); overflow-wrap: anywhere; }
.ex-option:has(:checked) { border-color: var(--accent); background: var(--accent-soft); }
.ex-editor { width: 100%; min-height: 260px; resize: vertical; box-sizing: border-box; padding: 18px; border: 1px solid var(--line-strong); border-radius: 10px; background: var(--surface-2); color: var(--fg); font: 500 15px/1.6 var(--font-mono); tab-size: 2; }
.ex-editor:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ex-feedback { display: grid; gap: 18px; font: 400 17px/1.6 var(--font-body); color: var(--fg-2); }
.ex-status { font-weight: 650; color: var(--fg); }
.ex-status.passed { color: var(--good); }
.ex-results { margin: 0; padding-left: 24px; overflow-wrap: anywhere; }
.ex-foot { font: 400 14px/1.6 var(--font-body); color: var(--fg-3); }
.ex-progress { color: var(--fg-3); font: 500 15px/1.5 var(--font-mono); }
.ex-doc .ls-equation { font-size: 21px; }
@media(max-width: 760px) { .ex-layout { grid-template-columns: minmax(0, 1fr); } .ex-doc { padding-top: 125px; } .ex-panel { padding: 20px; } .ex-doc h1 { font-size: 34px; } .ex-controls .ls-select { max-width: 100%; width: 100%; } }
`);
export default {
  mount(el, ctx) {
    let category = 'All', index = 0, selected = null, active = null, destroyed = false, storageOK = true;
    const drafts = new Map(), revealed = new Set();
    let progress = {};
    try { const stored = JSON.parse(localStorage.getItem('atlas-exercises-v1') || '{}'); if (stored && typeof stored === 'object' && !Array.isArray(stored)) progress = stored; } catch { storageOK = false; }
    const doc = h('div', { class: 'k-doc ex-doc', role: 'region', tabindex: 0, 'aria-label': 'Transformer exercises, scrollable practice' });
    const inner = h('div', { class: 'k-doc-inner' }); doc.append(inner); el.append(doc);
    const progressEl = h('div', { class: 'ex-progress', id: 'exercise-progress' });
    inner.append(h('header', null, h('div', { class: 'k-eyebrow' }, h('span', { class: 'k-chip', style: { '--kind': 'var(--c-head)' } }, 'Derive · debug · code')), h('h1', null, 'Transformer practice'), h('p', { class: 'lede' }, 'Explain the logic, repair a mistake, or implement the operation. One task at a time.')));
    const filters = h('select', { class: 'ls-select', id: 'exercise-category', 'aria-label': 'Exercise category', onChange: e => { category = e.target.value; index = 0; paint(); } },
      ['All', 'Shapes', 'Debugging', 'Derivation', 'Coding'].map(s => h('option', { value: s }, s)));
    const picker = h('select', { class: 'ls-select', id: 'exercise-picker', 'aria-label': 'Exercise', onChange: e => { index = Number(e.target.value); paint(); } });
    const previous = lessonButton('← Previous', () => { index = (index + list().length - 1) % list().length; paint(); });
    const next = lessonButton('Next →', () => { index = (index + 1) % list().length; paint(); });
    inner.append(h('div', { class: 'ex-controls' }, filters, picker, previous, next), progressEl);
    const layout = h('div', { class: 'ex-layout' }), work = h('div', { class: 'ls-panel ex-panel' }), side = h('div', { class: 'ls-panel ex-panel' }); layout.append(work, side); inner.append(layout);
    inner.append(h('div', { class: 'ex-foot' }, 'Coding tasks run JavaScript in a browser worker, with a two-second time limit and small deterministic tests. PyTorch source links show the corresponding model operation. Passing these tests checks the supplied cases; it does not prove every possible input correct. Progress is stored only in this browser.'));
    function list() { return EXERCISES.filter(e => category === 'All' || e.type === category); }
    function record(e) {
      progress[e.id] = { passed: true, revealed: revealed.has(e.id), at: Date.now() };
      try { localStorage.setItem('atlas-exercises-v1', JSON.stringify(progress)); } catch { storageOK = false; }
      updateProgress();
    }
    function updateProgress() {
      const completed = EXERCISES.filter(e => progress[e.id]?.passed).length;
      progressEl.textContent = `${completed} / ${EXERCISES.length} practiced${storageOK ? ' · saved in this browser' : ' · storage unavailable; this session only'}`;
    }
    function paint() {
      active?.cancel(); active = null; selected = null;
      const items = list(), e = items[index];
      ctx.node.reference = e.reference;
      ctx.node.keys = [e.math];
      picker.replaceChildren(...items.map((item, i) => h('option', { value: i, selected: index === i }, `${i + 1}. ${item.title}`)));
      work.dataset.exercise = e.id;
      const feedback = h('div', { class: 'ex-feedback', id: 'exercise-feedback', role: 'status', 'aria-live': 'polite' }, 'Try the task before revealing the worked solution.');
      const controls = h('div', { class: 'ex-controls', style: { marginTop: '20px' } });
      let editor;
      async function check() {
        if (active) return;
        if (e.type !== 'Coding') {
          if (selected === null) { feedback.textContent = 'Select an answer first.'; return; }
          const passed = selected === e.correct;
          feedback.replaceChildren(h('div', { class: ['ex-status', passed && 'passed'] }, passed ? 'Correct' : 'Try again'), equation(e.tex), h('div', null, e.explanation));
          feedback.dataset.passed = String(passed); if (passed) record(e); return;
        }
        run.disabled = true; feedback.textContent = 'Running test cases…';
        const job = runExercise(editor.value, e.tests); active = job;
        const result = await job.promise;
        if (destroyed || active !== job) return;
        active = null; run.disabled = false;
        feedback.dataset.passed = String(!!result.passed);
        feedback.replaceChildren(h('div', { class: ['ex-status', result.passed && 'passed'] }, result.error || (result.passed ? 'All supplied tests passed' : 'Some tests failed')),
          result.results?.length ? h('ul', { class: 'ex-results' }, result.results.map(r => h('li', null, `${r.passed ? '✓' : '×'} ${r.name}: ${r.message}`))) : null);
        if (result.passed) { feedback.append(equation(e.tex), h('div', null, e.explanation)); record(e); }
      }
      const run = lessonButton(e.type === 'Coding' ? 'Run tests' : 'Check answer', check, { id: 'exercise-run' });
      const solution = lessonButton('Worked solution', () => {
        active?.cancel(); active = null; run.disabled = false;
        revealed.add(e.id);
        if (editor) { editor.value = e.solution; drafts.set(e.id, editor.value); }
        feedback.replaceChildren(h('div', { class: 'ex-status' }, 'Worked solution'), e.choices ? h('div', null, e.choices[e.correct]) : null, equation(e.tex), h('div', null, e.explanation));
      }, { id: 'exercise-solution' });
      controls.append(run, lessonButton('Hint', () => { feedback.textContent = e.hint; }), solution);
      if (e.type === 'Coding') editor = h('textarea', { id: 'exercise-editor', class: 'ex-editor', 'aria-label': 'JavaScript solution', spellcheck: 'false', value: drafts.get(e.id) ?? e.starter, onInput: event => drafts.set(e.id, event.target.value) });
      const options = e.choices && h('div', { class: 'ex-options' }, e.choices.map((text, i) => h('label', { class: 'ex-option' }, h('input', { type: 'radio', name: 'exercise-answer', value: i, onChange: () => { selected = i; } }), text)));
      work.replaceChildren(h('div', { class: 'ls-meta', style: { color: `var(--c-${e.type === 'Coding' ? 'q' : 'head'})`, marginBottom: '12px' } }, e.type), h('h2', null, e.title), h('div', { class: 'ex-prompt' }, e.prompt), editor || options, controls);
      side.replaceChildren(h('h2', null, 'Reasoning & feedback'), feedback, h('div', { class: 'ex-controls', style: { marginTop: '26px' } },
        lessonButton('Relevant maths', () => ctx.showMath(e.math)), lessonButton('PyTorch operation', () => e.reference ? ctx.showReference(e.reference.file, e.reference.region) : ctx.showCode(e.id === 'code-gqa' ? 'L0.attn.k_heads' : e.id === 'rope-offset' ? 'L0.attn.q_rot' : 'mask'))));
      updateProgress();
      if (!reducedMotion()) layout.animate([{ opacity: .35 }, { opacity: 1 }], { duration: MOTION.appear, easing: MOTION.easeOut });
    }
    paint(); return { pause() { active?.cancel(); }, destroy() { destroyed = true; active?.cancel(); } };
  },
};
