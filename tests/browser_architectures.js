(async () => {
  let checks = 0;
  const assert = (ok, msg) => { if (!ok) throw new Error(msg); checks++; };
  document.getElementById('architectures-counts').click();
  for (let step = 0; step < 3; step++) {
    document.querySelectorAll('.atlas-story .story-dot')[step].click();
    assert(document.querySelectorAll('.ac-row.on').length === 3, 'one feature group per architecture highlighted');
    assert(!document.querySelector('.katex-error'), 'comparison equations compile');
  }
  assert(document.querySelectorAll('.ac-count').length === 3 && !document.querySelector('.ac-count').textContent.includes('NaN'), 'actual unique parameter counts');
  for (const card of document.querySelectorAll('.ac-card')) assert([...card.children].every(c => c.getBoundingClientRect().bottom <= card.getBoundingClientRect().bottom), 'comparison cards fit');
  return { checks };
})()
