(async () => {
  let checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  const baseline = window.__te.store.state.model;
  const select = (id, value) => { const e = document.getElementById(id); e.value = String(value); e.dispatchEvent(new Event('change')); };
  for (let example = 0; example < 4; example++) {
    select('encdec-example', example);
    for (let head = 0; head < 2; head++) {
      select('encdec-head', head);
      for (let step = 0; step < 5; step++) {
        document.querySelectorAll('.atlas-story .story-dot')[step].click();
        assert(document.querySelectorAll('.ec-panel').length === 3, 'three real panels rendered');
        assert(document.querySelector('[data-encdec-result]').dataset.paddedZero === 'true', 'padded source keys have zero cross-attention');
        assert(!document.querySelector('.katex-error'), 'all scene equations compile');
      }
    }
  }
  select('encdec-example', 0); document.querySelectorAll('.atlas-story .story-dot')[2].click();
  assert(document.querySelector('[data-encdec-result]').dataset.shape === '3,4', 'cross-attention is rectangular');
  document.getElementById('encdec-numbers').click(); await new Promise(r => setTimeout(r, 900));
  for (const card of document.querySelectorAll('.ec-panel')) {
    const b = card.getBoundingClientRect();
    assert([...card.children].every(c => { const r = c.getBoundingClientRect(); return r.bottom <= b.bottom && r.right <= b.right; }), 'card content fits');
  }
  assert(window.__te.store.state.model === baseline, 'main model remains intact');
  document.querySelector('.atlas-tools [aria-label="Code for this view"]').click(); await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/encoder_decoder.py'), 'actual PyTorch source');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  return { checks };
})()
