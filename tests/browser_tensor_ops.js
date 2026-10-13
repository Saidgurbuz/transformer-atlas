(async () => {
  const { atlas } = window.__te;
  const top = () => atlas.stack().at(-1);
  const steps = top().steps;
  const expected = ['36', '2,3,2,3', '2,2,3,3', '2,6,3', '2,2,3,3', '2,3,6', '2,2,3,3', '1,3', '3'];
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const selectStep = (i) => document.querySelectorAll('.atlas-story .story-dot')[i].click();
  for (let i = 0; i < steps.labels.length; i++) {
    selectStep(i);
    const result = document.querySelector('[data-lesson-result="tensor-ops"]');
    assert(result.dataset.shape === expected[i], `step ${i} shape`);
    assert([...result.querySelectorAll('.ls-slot')].every(e => e.getBoundingClientRect().bottom <= result.getBoundingClientRect().bottom), `step ${i} grids fit`);
    assert(!document.querySelector('.katex-error'), `step ${i} maths`);
  }
  selectStep(2);
  const resultPanel = document.querySelector('[data-lesson-result="tensor-ops"]');
  assert([...resultPanel.querySelectorAll('.ls-slot')].every(e => e.getBoundingClientRect().bottom <= resultPanel.getBoundingClientRect().bottom), 'all head grids fit their panel');
  assert(document.querySelector('[data-lesson-result="tensor-ops"]').dataset.contiguous === 'false', 'transpose is a strided view');
  assert(document.querySelectorAll('.ls-slot.labelled').length === 0, 'offset numbers are hidden by default');
  document.getElementById('tensor-ops-labels').click();
  assert(document.querySelectorAll('.ls-slot.labelled').length > 0, 'offsets can be revealed');
  document.getElementById('tensor-ops-labels').click();
  document.querySelector('.atlas-tools [aria-label="Code for this view"]').click();
  await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/tensor_ops.py'), 'code is the correct executable reference');
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('torch.einsum'), 'reference contains the tested tensor operations');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  await new Promise(r => setTimeout(r, 700));
  return { checks, operation: steps.labels[2] };
})()
