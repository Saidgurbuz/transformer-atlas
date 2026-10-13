(async () => {
  const baseline = window.__te.store.state.model;
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const selectStep = (i) => document.querySelectorAll('.atlas-story .story-dot')[i].click();
  for (const profile of ['gpt', 'llama']) {
    document.getElementById(`training-${profile}`).click();
    for (const optimizer of ['adamw', 'sgd']) {
      const opt = document.getElementById('training-optimizer'); opt.value = optimizer; opt.dispatchEvent(new Event('change'));
      for (let i = 0; i < 5; i++) {
        selectStep(i);
        assert(!document.querySelector('.katex-error'), `${profile} ${optimizer} step ${i} maths`);
        const data = document.querySelector('[data-training-result]');
        assert(Number.isFinite(Number(data.dataset.before)) && Number.isFinite(Number(data.dataset.after)), 'real losses are finite');
      }
      document.getElementById('training-next').click();
      assert(document.querySelector('[data-training-result]').dataset.round === '2', 'optimizer state continues across updates');
    }
  }
  selectStep(2);
  const parameter = document.getElementById('training-parameter');
  for (const value of [...parameter.options].map(o => o.value)) {
    parameter.value = value; parameter.dispatchEvent(new Event('change'));
    assert(!document.querySelector('.atlas-error'), `gradient for ${value}`);
  }
  assert(window.__te.store.state.model === baseline, 'training lesson leaves the main model intact');
  document.querySelector('.atlas-tools [aria-label="Code for this view"]').click();
  await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/training_lesson.py'), 'training has its actual reference source');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  document.getElementById('training-gpt').click(); selectStep(2);
  const p = document.getElementById('training-parameter'); p.value = 'blocks.0.attn.q_proj.weight'; p.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 900));
  for (const card of document.querySelectorAll('.tr-card')) assert([...card.children].every(e => e.getBoundingClientRect().bottom <= card.getBoundingClientRect().bottom), 'training cards fit their content');
  return { checks, before: document.querySelector('[data-training-result]').dataset.before };
})()
