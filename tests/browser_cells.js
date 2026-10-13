// Run with tools/smoke.sh --view stepper --preset llama --eval-file tests/browser_cells.js.
(async () => {
  const { store, atlas } = window.__te;
  const checks = [];
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks.push(message); };
  const settle = () => new Promise((r) => setTimeout(r, 100));
  const button = (label) => [...document.querySelectorAll('.atlas-drawer.right.open button')].find((b) => b.textContent === label);
  const tool = (label) => document.querySelector(`.atlas-tools [aria-label="${label}"]`);
  const close = () => document.querySelector('.atlas-drawer.right.open [aria-label="Close"]')?.click();
  const coords = () => [...document.querySelectorAll('.atlas-drawer.right.open .cell-coordinates input')].map((i) => Number(i.value));
  const openCode = async () => { tool('Code for this view').click(); await settle(); };
  const key = 'L1.attn.probs';
  store.focusCell(key, { band: 3, r: 4, x: 2 });
  const timelineCell = document.querySelector('.sp-timeline .sp-cell');
  assert(document.querySelector('.sp-side .cell-focus')?.dataset.focusKey === key, 'focus jumps the stepper to the exact tensor');
  assert(document.querySelector('.sp-side .cell-focus')?.dataset.focusCell === '4,2,3', 'stepper preserves head, row and column');
  const source = document.querySelector('.sp-code .code');
  assert(source.scrollWidth <= source.clientWidth + 1, 'source lines wrap without hiding trace links off screen');

  // Hit a real canvas coordinate. The plot includes 16 px of column labels.
  const canvas = document.querySelector('.sp-side .heatmap canvas'), rect = canvas.getBoundingClientRect();
  const T = store.state.ids.length, dpr = devicePixelRatio;
  const cw = (canvas.height / dpr - 16) / T, labelW = canvas.width / dpr - T * cw;
  const point = { clientX: rect.left + (labelW + 2.5 * cw) * rect.width / (canvas.width / dpr),
    clientY: rect.top + (16 + 4.5 * cw) * rect.height / (canvas.height / dpr), bubbles: true };
  canvas.dispatchEvent(new MouseEvent('contextmenu', { ...point, button: 2 }));
  assert(canvas.dataset.cell === '4,2,3', 'canvas right-click publishes the original head and coordinates');
  const showCode = [...document.querySelectorAll('.ctx-item')].find((b) => b.textContent.startsWith('Show the code'));
  assert(!!showCode, 'canvas cell has an explicit code action');
  showCode.click(); await settle();
  assert(coords().join(',') === '3,4,2', 'code drawer carries the clicked cell');
  const line = Number(document.querySelector('.sp-code .ln.hl').dataset.line);
  assert(!!document.querySelector(`.atlas-drawer.right.open .ln.hl[data-line="${line}"]`), 'correct producer line is highlighted');
  assert(!!button('Inspect this cell'), 'code offers a touch/keyboard route back to the cell');
  button('Inspect this cell').click(); await settle();
  assert(coords().join(',') === '3,4,2', 'code-to-inspector round trip keeps all coordinates');
  const sourceBox = source.getBoundingClientRect(), highlightBox = source.querySelector('.ln.hl').getBoundingClientRect();
  assert(highlightBox.top >= sourceBox.top && highlightBox.bottom <= sourceBox.bottom, 'highlighted source line stays visible when the inspector narrows the pane');
  assert(document.querySelector('.sp-timeline .sp-cell') === timelineCell, 'selection changes reuse the stepper timeline');
  assert(document.querySelector('.atlas-drawer.right .math-ok')?.textContent.includes('matches'), 'cell equation independently matches the traced value');
  const expected = store.state.trace[key].data[(3 * T + 4) * T + 2];
  const printed = Number(document.querySelector('.atlas-drawer.right .cell-value').textContent.replace('=', '').trim().replace('−', '-'));
  assert(Math.abs(printed - expected) < 1e-6, 'inspector shows the clicked value, not column zero');

  const col = document.querySelector('.atlas-drawer.right input[data-axis="x"]');
  col.value = 1; col.dispatchEvent(new Event('change', { bubbles: true })); await settle();
  assert(store.state.cellFocus.cell.x === 1, 'coordinate input updates the selected cell');
  button('Show in full source').click(); await settle();
  const actChips = [...document.querySelectorAll('.atlas-drawer.right.open .trace-chip[data-trace-key="L1.mlp.act"]')];
  assert(actChips.length === 3 && actChips.filter((b) => !b.disabled).length === 1, 'only the active SwiGLU branch is clickable');
  const kvChip = document.querySelector('.atlas-drawer.right.open .trace-chip[data-trace-key="L1.attn.k_rot"]:not(:disabled)');
  kvChip.click(); await settle();
  assert(coords()[0] === 1, 'query head 3 maps to shared KV head 1');
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('Inside apply_rope'), 'RoPE exposes the actual rotation implementation');
  assert(store.state.sel.head === 3, 'selecting its KV head preserves the corresponding query head');
  button('Follow in code stepper').click(); await atlas.go('stepper'); await settle();
  assert(document.querySelector('.sp-side .cell-focus')?.dataset.focusKey === 'L1.attn.k_rot', 'inspector opens the matching stepper tensor');
  assert(document.querySelector('.sp-side .cell-focus')?.dataset.focusCell.endsWith(',1'), 'stepper uses the KV slice, not the query-head number');

  // Reducing the input safely clamps the focused cell, preserving the source key.
  store.focusCell('L1.attn.probs', { band: 3, r: 8, x: 8 });
  store.setText('the'); await settle();
  assert(store.state.cellFocus.cell.r < store.state.ids.length && store.state.cellFocus.cell.x < store.state.ids.length, 'input changes clamp stale coordinates');
  assert(!document.querySelector('.math-bad, .katex-error'), 'live maths remains valid after shortening the input');

  await store.loadPreset('charlm');
  store.setConfig({ norm_placement: 'post', norm: 'rmsnorm', activation: 'relu', residual: false, pos_encoding: 'alibi', attn_scale: false });
  await atlas.go('stepper'); await settle();
  store.focusCell('L1.resid_mid', { r: 2, x: 5 });
  assert(document.querySelector('.sp-code .ln.hl .src').textContent.includes('self.ln1(s)'), 'post-LN correspondence selects normalization after the add');
  await openCode();
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('Inside RMSNorm'), 'post-LN exposes RMSNorm internals');
  const reluChips = [...document.querySelectorAll('.atlas-drawer.right.open .trace-chip[data-trace-key="L1.mlp.act"]')];
  assert(reluChips.filter((b) => !b.disabled).length === 1 && reluChips.find((b) => !b.disabled).closest('.ln').querySelector('.src').textContent.includes('F.relu'), 'ReLU branch availability follows configuration changes');
  close();

  await atlas.go('generation'); await settle();
  tool('Maths for this view').click(); await settle();
  assert(document.querySelector('.atlas-drawer.right.open').textContent.includes('Incremental decoding and KV-cache storage'), 'generation exposes the deeper cache equations');
  assert(!document.querySelector('.katex-error'), 'cache equations typeset correctly');
  close();
  await atlas.go('embedding');
  await atlas.go('generation');
  const waitRoute = async (id) => {
    for (let n = 0; n < 50 && atlas.stack().at(-1)?.id !== id; n++) await settle();
    assert(atlas.stack().at(-1)?.id === id, `browser history reaches ${id}`);
  };
  history.back(); await waitRoute('embedding');
  history.forward(); await waitRoute('generation');
  await store.loadPreset('llama');
  await atlas.go('stepper');
  store.focusCell(key, { band: 3, r: 4, x: 2 });
  await openCode();
  button('Inspect this cell').click(); await settle();
  const drawer = document.querySelector('.atlas-drawer.right.open .atlas-drawer-body');
  assert(drawer.scrollWidth <= drawer.clientWidth + 1, 'inspector does not overflow the drawer');
  assert(!document.querySelector('.math-bad, .katex-error, .atlas-error'), 'final screen has no math or scene errors');
  await new Promise((r) => setTimeout(r, 3600)); // let transient configuration notices clear before the screenshot
  return { checks: checks.length, passed: checks, finalCell: store.state.cellFocus };
})()
