(async () => {
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const main = document.querySelector('.atlas-layer[data-id="training"]').textContent;
  const click = text => [...document.querySelectorAll('.atlas-drawer.right.open button')].find(b => b.textContent === text).click();
  const settle = () => new Promise(r => setTimeout(r, 100));
  document.querySelector('.atlas-tools [aria-label="Maths for this view"]').click(); await settle();
  assert([...document.querySelectorAll('.math-study')].every(d => !d.open), 'deeper study starts collapsed');
  const link = document.querySelector('[data-math-concept="concept:attention-backward"]');
  assert(!!link, 'backward chain is reachable from the training equations');
  link.closest('details').open = true; link.click(); await settle();
  assert(document.querySelector('.math-title').textContent === 'Attention backward, end to end', 'study link opens the chosen equation');
  assert(!document.querySelector('.katex-error'), 'backward maths compiles in the drawer');
  const source = document.querySelector('[data-math-reference="attention_backward"]'); source.closest('details').open = true; source.click(); await settle();
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/attention_backward.py'), 'derivative links its executable source');
  click('Return to explanation'); await settle();
  assert(document.querySelector('.math-title').textContent === 'Attention backward, end to end', 'source returns to its explanation');
  const body = document.querySelector('.atlas-drawer.right.open .atlas-drawer-body');
  assert(body.scrollWidth <= body.clientWidth + 1, 'deep content stays within drawer width');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  assert(document.querySelector('.atlas-layer[data-id="training"]').textContent === main, 'main lesson text stays concise and unchanged');
  return { checks };
})()
