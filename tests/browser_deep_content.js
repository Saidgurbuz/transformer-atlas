(async () => {
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const settle = () => new Promise(r => setTimeout(r, 100));
  const cases = [
    ['training', ['attention-backward', 'rmsnorm-backward']],
    ['norm.0.1', ['layernorm-backward', 'norm-epsilon']],
    ['softmax.0.0', ['attention-invariants', 'head-basis', 'attention-circuits', 'attention-numerics']],
    ['positional', ['permutation']], ['generation', ['cache-equivalence']],
    ['mlp.0', ['mlp-interpretation', 'gelu']],
  ];
  for (const [route, concepts] of cases) {
    await window.__te.atlas.go(route); await settle();
    for (const id of concepts) {
      document.querySelector('.atlas-tools [aria-label="Maths for this view"]').click(); await settle();
      const link = document.querySelector(`[data-math-concept="concept:${id}"]`);
      assert(!!link, `${id} reachable from ${route}`);
      assert(!link.closest('details').open, 'further study starts collapsed');
      link.closest('details').open = true; link.click(); await settle();
      assert([...document.querySelectorAll('.math-deriv')].every(d => !d.open), 'derivations start collapsed');
      document.querySelectorAll('.math-deriv').forEach(d => d.open = true);
      assert(!document.querySelector('.katex-error'), `${id} all equations compile`);
      const body = document.querySelector('.atlas-drawer.right.open .atlas-drawer-body');
      assert(body.scrollWidth <= body.clientWidth + 1, `${id} fits drawer width`);
      const regions = [...document.querySelectorAll('[data-math-reference]')].map(b => b.dataset.mathReference);
      for (const region of regions) {
        const button = document.querySelector(`[data-math-reference="${region}"]`); button.closest('details').open = true; button.click(); await settle();
        assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('reference/'), `${region} executable source displayed`);
        [...document.querySelectorAll('.atlas-drawer.right.open button')].find(b => b.textContent === 'Return to explanation').click(); await settle();
        assert(!!document.querySelector(`[data-math-reference="${region}"]`), 'source returns to its own equation');
      }
      document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
    }
  }
  await window.__te.atlas.go('training');
  document.querySelector('.atlas-tools [aria-label="Maths for this view"]').click(); await settle();
  const link = document.querySelector('[data-math-concept="concept:attention-backward"]'); link.closest('details').open = true; link.click();
  return { checks };
})()
