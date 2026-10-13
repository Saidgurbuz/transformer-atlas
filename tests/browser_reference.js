(async () => {
  const store = window.__te.store;
  store.focusCell('L0.attn.probs', { band: 0, r: 0, x: 0 });
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const open = label => document.querySelector(`.atlas-tools [aria-label="${label}"]`).click();
  const close = () => document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  open('Code for this view'); await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/tensor_ops.py'), 'lesson source wins over an earlier main-model cell selection'); close();
  open('Maths for this view'); await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('Head layout, strides and contraction'), 'lesson maths follows its own operation'); close();
  [...document.querySelectorAll('.ls-button')].find(b => b.textContent === 'PyTorch code').click(); await new Promise(r => setTimeout(r, 100));
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/tensor_ops.py'), 'explicit source request preserves its reference'); close();
  assert(store.state.cellFocus.key === 'L0.attn.probs', 'original cell selection is preserved');
  return { checks };
})()
