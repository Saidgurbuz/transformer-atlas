// Each coding exercise is synchronous and receives only its small test arguments.
// A Worker keeps infinite loops off the UI thread; this is not a security sandbox.
export function evaluateExercise(code, tests) {
  function equal(a, b) {
    if (typeof a === 'number' && typeof b === 'number') return Number.isFinite(a) && Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(b));
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => equal(v, b[i]));
    return a === b;
  }
  let solve;
  try { solve = new Function('"use strict"; return (' + code + '\n);')(); if (typeof solve !== 'function') throw new Error('Submit a function.'); }
  catch (e) { return { error: e.message, results: [] }; }
  const results = tests.map(t => {
    try {
      const value = solve(...structuredClone(t.args));
      return { name: t.name, passed: !t.error && equal(value, t.expected), message: t.error ? 'Expected an error for invalid input.' : equal(value, t.expected) ? 'Passed' : 'Output differs from the expected value or shape.' };
    } catch (e) { return { name: t.name, passed: !!t.error, message: t.error ? 'Invalid input rejected' : e.message }; }
  });
  return { results, passed: results.length > 0 && results.every(r => r.passed) };
}
export function runExercise(code, tests, { timeoutMs = 2000 } = {}) {
  const source = `const evaluate = ${evaluateExercise.toString()}; onmessage = e => postMessage(evaluate(e.data.code, e.data.tests));`;
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  let worker, timer, finish;
  const promise = new Promise(resolve => {
    let done = false;
    finish = result => { if (done) return; done = true; clearTimeout(timer); worker?.terminate(); URL.revokeObjectURL(url); resolve(result); };
    try {
      worker = new Worker(url);
      worker.onmessage = e => finish(e.data);
      worker.onerror = e => { e.preventDefault(); finish({ error: e.message || 'Worker failed.', results: [] }); };
      timer = setTimeout(() => finish({ error: 'Time limit reached. Check for an infinite loop.', results: [] }), timeoutMs);
      worker.postMessage({ code, tests });
    } catch (e) { finish({ error: e.message, results: [] }); }
  });
  return { promise, cancel: () => finish({ cancelled: true, error: 'Run cancelled.', results: [] }) };
}
