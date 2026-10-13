// Online softmax with an unnormalised output accumulator. This is a CPU teaching
// implementation of the FlashAttention recurrence, not a GPU performance benchmark.
export function streamAttentionRow(scores, values, width, tileSize = 4) {
  if (!Number.isInteger(width) || width < 1 || values.length !== scores.length * width || !Number.isInteger(tileSize) || tileSize < 1) throw new Error('Invalid streamed attention dimensions');
  if (Array.from(scores).some((v) => v !== -Infinity && !Number.isFinite(v))) throw new Error('Scores must be finite or masked with −Infinity');
  let max = -Infinity, sum = 0;
  const accumulator = new Float64Array(width), tiles = [];
  for (let start = 0; start < scores.length; start += tileSize) {
    const end = Math.min(start + tileSize, scores.length), tileMax = Math.max(...scores.slice(start, end));
    const oldMax = max, oldSum = sum, oldAccumulator = accumulator.slice();
    const nextMax = Math.max(max, tileMax), weights = new Float64Array(end - start);
    const correction = max === -Infinity ? 0 : Math.exp(max - nextMax);
    if (nextMax !== -Infinity) {
      for (let d = 0; d < width; d++) accumulator[d] *= correction;
      sum *= correction;
      for (let j = start; j < end; j++) {
        const p = scores[j] === -Infinity ? 0 : Math.exp(scores[j] - nextMax);
        weights[j - start] = p; sum += p;
        for (let d = 0; d < width; d++) accumulator[d] += p * values[j * width + d];
      }
    }
    max = nextMax;
    tiles.push({ start, end, oldMax, oldSum, oldAccumulator, tileMax, max, sum, correction, weights,
      accumulator: accumulator.slice(), output: Float64Array.from(accumulator, (v) => sum ? v / sum : 0) });
  }
  if (!sum) throw new Error('Attention needs at least one visible key');
  return { tiles, output: Float64Array.from(accumulator, (v) => v / sum), max, sum };
}
