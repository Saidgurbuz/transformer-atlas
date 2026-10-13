(async () => {
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const change = (id, value) => { const e = document.getElementById(id); e.value = String(value); e.dispatchEvent(new Event('change')); };
  const result = () => document.getElementById('exercise-feedback');
  const wait = async () => { const end = Date.now() + 4000; while (document.getElementById('exercise-run')?.disabled && Date.now() < end) await new Promise(r => setTimeout(r, 30)); assert(!document.getElementById('exercise-run').disabled, 'run returns and re-enables button'); };
  const main = window.__te.store.state.model;
  const answers = [1, 0, 1, 1, 1, 2, 1, 2, 0, 1, 2, 0];
  for (let i = 0; i < 12; i++) {
    change('exercise-picker', i);
    const options = document.querySelectorAll('input[name="exercise-answer"]');
    options[(answers[i] + 1) % options.length].click(); document.getElementById('exercise-run').click();
    assert(result().dataset.passed === 'false', 'wrong answer rejected');
    options[answers[i]].click(); document.getElementById('exercise-run').click();
    assert(result().dataset.passed === 'true', 'correct answer accepted');
    assert(!document.querySelector('.katex-error'), 'worked derivation compiles');
  }
  change('exercise-category', 'Coding');
  for (let i = 0; i < 4; i++) {
    change('exercise-picker', i); document.getElementById('exercise-run').click(); await wait();
    assert(result().dataset.passed === 'false', 'unfinished code fails');
    document.getElementById('exercise-solution').click(); document.getElementById('exercise-run').click(); await wait();
    assert(result().dataset.passed === 'true', 'worked code passes real worker tests');
  }
  change('exercise-picker', 0);
  const editor = () => document.getElementById('exercise-editor');
  editor().value = 'function broken('; document.getElementById('exercise-run').click(); await wait();
  assert(result().dataset.passed === 'false', 'syntax errors handled');
  editor().value = 'function solve() { while (true) {} }'; document.getElementById('exercise-run').click();
  let responsive = false; setTimeout(() => { responsive = true; }, 50);
  await wait(); assert(responsive && result().textContent.includes('Time limit'), 'infinite loop times out without blocking the UI');
  editor().value = 'function solve() { while (true) {} }'; document.getElementById('exercise-run').click(); change('exercise-picker', 1);
  await new Promise(r => setTimeout(r, 100)); assert(!document.getElementById('exercise-run').disabled && !result().textContent.includes('Time limit'), 'changing exercises cancels the old worker');
  change('exercise-picker', 0); editor().value = 'function solve() { while (true) {} }'; document.getElementById('exercise-run').click(); document.getElementById('exercise-solution').click();
  await new Promise(r => setTimeout(r, 100)); assert(!document.getElementById('exercise-run').disabled && result().textContent.includes('Worked solution'), 'revealing a solution cancels the outdated run');
  assert(window.__te.store.state.model === main, 'coding leaves main model unchanged');
  const saved = JSON.parse(localStorage.getItem('atlas-exercises-v1'));
  assert(Object.values(saved).filter(v => v.passed).length === 16, 'practice progress saved');
  change('exercise-picker', 0); document.getElementById('exercise-solution').click(); document.getElementById('exercise-run').click(); await wait();
  document.querySelector('.atlas-tools [aria-label="Code for this view"]').click(); await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/flash_attention.py'), 'toolbar follows the selected exercise reference');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  return { checks };
})()
