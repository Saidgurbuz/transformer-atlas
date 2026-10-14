(async () => {
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  const settle = () => new Promise(r => setTimeout(r, 100));
  const cases = [
    ['training', ['attention-backward', 'layernorm-backward'], 'tinystories'],
    ['norm.0.1', ['layernorm-backward', 'norm-epsilon'], 'tinystories'],
    ['softmax.0.0', ['attention-invariants', 'online-attention', 'attention-backward'], 'tinystories'],
    ['scores.0.0', ['head-basis', 'qk-normalization'], 'tinystories'],
    ['mask.0.0', ['attention-numerics', 'permutation'], 'tinystories'],
    ['positional', ['permutation'], 'tinystories'],
    ['positional', ['rope-relative', 'rope-context'], 'llama'],
    ['norm.0.1', ['rmsnorm-backward'], 'llama'],
    ['generation', ['cache-equivalence', 'mla'], 'llama'],
    ['mlp.0', ['mlp-interpretation', 'swiglu'], 'llama'],
    ['mlp.0', ['mlp-interpretation', 'gelu'], 'tinystories'],
  ];
  for (const [route, concepts, preset] of cases) {
    await window.__te.store.loadPreset(preset);
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
  // A drilled-down explanation must not follow us into an unrelated view.
  const tool = () => document.querySelector('.atlas-tools [aria-label="Maths for this view"]');
  await window.__te.atlas.go('softmax.0.0'); tool().click(); await settle();
  const invariant = document.querySelector('[data-math-concept="concept:attention-invariants"]');
  invariant.closest('details').open = true; invariant.click(); await settle();
  await window.__te.atlas.go('positional'); await settle();
  assert(document.querySelector('.math-title')?.textContent === 'Positional encoding', 'open maths follows the new view');
  assert(!document.querySelector('[data-math-concept="concept:online-attention"]'), 'previous softmax topics disappear');
  assert(!document.querySelector('[data-math-concept="concept:rope-context"]'), 'learned positions omit rotary extensions');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  window.__te.store.focusCell('emb.pos', { r: 0, x: 0 }); tool().click(); await settle();
  await window.__te.store.loadPreset('llama'); await settle();
  assert(!!document.querySelector('[data-math-concept="concept:rope-context"]'), 'a vanished focused tensor falls back to the current rotary equations');
  assert(document.querySelector('.math-title')?.textContent === 'Queries after RoPE', 'preset changes refresh the operation as well as study links');
  await window.__te.store.loadPreset('tinystories'); await settle();
  await window.__te.atlas.go('tokens'); await settle();
  assert(!document.querySelector('.math-study'), 'views without equations have no further study section');
  await window.__te.atlas.go('norm.0.1'); await settle();
  window.__te.store.setConfig({ norm: 'rmsnorm' }); await settle();
  assert(!!document.querySelector('[data-math-concept="concept:rmsnorm-backward"]'), 'open links follow a norm configuration change');
  assert(!document.querySelector('[data-math-concept="concept:layernorm-backward"]'), 'inactive norm derivative is absent');
  await window.__te.atlas.go('mlp.0'); await settle();
  window.__te.store.setConfig({ activation: 'relu' }); await settle();
  assert(!document.querySelector('[data-math-concept="concept:gelu"], [data-math-concept="concept:swiglu"]'), 'ReLU omits unrelated activation formulas');
  document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  await window.__te.atlas.go('training'); tool().click(); await settle();
  const link = document.querySelector('[data-math-concept="concept:attention-backward"]'); link.closest('details').open = true; link.click();
  return { checks };
})()
