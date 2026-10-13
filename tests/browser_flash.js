(async () => {
  let checks = 0;
  const assert = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  const store = window.__te.store;
  for (const config of [{}, { n_head: 4, n_kv_head: 1, pos_encoding: 'rope' }, { pos_encoding: 'alibi', attn_scale: false }, { mask: 'prefix', prefix_len: 3 }, { norm_placement: 'post', mask: 'none' }]) {
    assert(!store.setConfig(config)?.length, 'valid test configuration'); await new Promise(r => setTimeout(r, 350));
    store.select({ token: store.state.ids.length - 1, head: store.state.config.n_head - 1 });
    for (let i = 0; i < 5; i++) {
      document.querySelectorAll('.atlas-story .story-dot')[i].click();
      assert(document.querySelectorAll('.fl-panel').length === 3, 'three actual tile panels are rendered');
      assert(document.querySelector('[data-flash-result]').dataset.matched === 'true', 'stream matches actual dense trace');
      assert(!document.querySelector('.katex-error'), 'step equations compile');
    }
    const size = document.getElementById('flash-size'); size.value = '8'; size.dispatchEvent(new Event('change'));
    for (let i = 0; i < Number(document.querySelector('[data-flash-result]').dataset.tiles); i++) document.getElementById('flash-next').click();
  }
  document.getElementById('flash-numbers').click();
  await new Promise(r => setTimeout(r, 900));
  for (const card of document.querySelectorAll('.fl-panel')) assert([...card.children].every(c => c.getBoundingClientRect().bottom <= card.getBoundingClientRect().bottom), 'tile panels fit content');
  store.setText('a'); await new Promise(r => setTimeout(r, 350));
  assert(document.querySelector('[data-flash-result]').dataset.matched === 'true', 'one-token attention works');
  store.setText('the cat sat on the mat'); await new Promise(r => setTimeout(r, 350));
  return { checks };
})()
