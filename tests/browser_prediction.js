// Run with tools/smoke.sh --view model --preset tinystories --eval-file tests/browser_prediction.js.
(async () => {
  const { store, atlas } = window.__te;
  let checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  const settle = async () => {
    await new Promise(r => setTimeout(r, 250));
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
  };
  const current = () => atlas.stack().at(-1).el;
  const prediction = () => {
    const st = store.state, q = st.sel.token, V = st.config.vocab_size;
    const values = st.trace.probs.data.subarray(q * V, (q + 1) * V);
    let best = 0;
    for (let v = 1; v < V; v++) if (values[v] > values[best]) best = v;
    const shown = current().querySelector('.ma-win b')?.textContent;
    assert(shown === st.tokenizer.label(best), `overview winner matches row ${q}: ${shown} versus ${st.tokenizer.label(best)}`);
    assert(current().querySelector('.ma-win em')?.textContent === `next after “${st.tokenizer.label(st.ids[q])}”`, 'overview names the token being continued');
  };
  assert(store.state.sel.token === store.state.ids.length - 1, 'preset starts at its final token');
  prediction();
  for (const text of ['Lily had a red ball. One day, she will', 'Lily had a blue ball. One day, she can', 'Lily will', 'Lily will play']) {
    store.select({ token: 0 });
    document.querySelector('.atlas-tools [aria-label="Model, weights and input"]').click();
    const input = document.getElementById('settings-text');
    input.value = text;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    assert(store.state.text === text, 'input field updates the live model');
    assert(store.state.sel.token === store.state.ids.length - 1, 'input edits follow the final retained token');
    prediction();
    document.querySelector('.atlas-drawer.right.open [aria-label="Close"]').click();
  }
  store.select({ token: 0 }); await settle();
  prediction();
  await atlas.go('lm-head'); await settle();
  assert(current().querySelector('[data-cell]')?.dataset.cell.startsWith('0,'), 'LM-head details follow an explicit token selection');
  store.setText('Lily will play'); await settle();
  assert(current().querySelector('[data-cell]')?.dataset.cell.startsWith(`${store.state.ids.length - 1},`), 'LM-head details resume the final position after editing');
  assert(current().querySelector('.k-pick.sel')?.title === `position ${store.state.ids.length - 1}`, 'token picker agrees with the prediction position');
  await atlas.go('model'); await settle();
  prediction();
  const selected = store.state.sel.token;
  store.setConfig({ attn_scale: !store.state.config.attn_scale }); await settle();
  assert(store.state.sel.token === selected, 'settings keep the explored position');
  store.setText('Lily '.repeat(100), { quiet: true }); await settle();
  assert(store.state.sel.token === store.state.config.block_size - 1, 'long input follows the retained context boundary');
  prediction();
  store.setText(''); await settle();
  assert(store.state.ids.length === 1 && store.state.sel.token === 0, 'empty input keeps the safe fallback');
  store.setConfig({ attn_scale: !store.state.config.attn_scale });
  store.setText('Lily had a red ball. One day, she will'); await settle();
  prediction();
  return { checks, token: store.state.sel.token, tokens: store.state.ids.length };
})()
