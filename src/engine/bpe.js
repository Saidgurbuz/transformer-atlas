// Byte-level BPE tokenizer, the GPT-2 scheme (also what the HF `tokenizers`
// ByteLevel BPE trained by tools/train_bpe_preset.py produces):
//   1. split text with the GPT-2 regex into words (a leading space stays with its word),
//   2. map each word's UTF-8 bytes to printable unicode characters,
//   3. repeatedly merge the adjacent pair with the lowest merge rank,
//   4. look the resulting pieces up in the vocabulary.
// tests/bpe.mjs checks it against the Python tokenizer on the training corpus.

const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256+n.
function byteMaps() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const enc = new Array(256), dec = new Map();
  bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec.set(String.fromCharCode(cs[i]), b); });
  return { enc, dec };
}
const { enc: BYTE_ENC, dec: BYTE_DEC } = byteMaps();
const utf8 = new TextEncoder();
const utf8dec = new TextDecoder('utf-8', { fatal: false });

/** Raw text of a byte-level token string (may be a partial UTF-8 sequence). */
export function bytesOf(tok) { return Uint8Array.from(Array.from(tok), (c) => BYTE_DEC.get(c) ?? 63); }

/**
 * @param {{vocab: string[], merges: (string|string[])[], special?: string[]}} spec
 *   vocab[id] is the byte-level token string; merges in priority order ("a b" or ["a", "b"]).
 */
export function makeBPETokenizer(spec) {
  const vocab = spec.vocab;
  const special = new Set(spec.special || []);
  const index = new Map(vocab.map((t, i) => [t, i]));
  const ranks = new Map(spec.merges.map((m, i) => [Array.isArray(m) ? `${m[0]} ${m[1]}` : m, i]));
  const cache = new Map();

  function bpe(word) {
    const hit = cache.get(word);
    if (hit) return hit;
    let syms = Array.from(word);
    while (syms.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = ranks.get(`${syms[i]} ${syms[i + 1]}`);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = syms[best], b = syms[best + 1], out = [];
      for (let i = 0; i < syms.length;) {
        if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) { out.push(a + b); i += 2; } else { out.push(syms[i]); i += 1; }
      }
      syms = out;
    }
    if (cache.size > 20000) cache.clear();
    cache.set(word, syms);
    return syms;
  }

  const words = (text) => text.match(PAT) || [];
  const toByteChars = (w) => Array.from(utf8.encode(w), (b) => BYTE_ENC[b]).join('');

  function label(id) {
    const t = vocab[id];
    if (t === undefined) return '?';
    if (special.has(t)) return '⟨eot⟩';
    const s = utf8dec.decode(bytesOf(t));
    return s.replace(/^ /, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥') || '␣';
  }

  return {
    kind: 'bpe',
    vocab: vocab.map((_, i) => label(i)),      // display labels, index = id
    size: vocab.length,
    encode(text) {
      const ids = [];
      for (const w of words(text)) for (const p of bpe(toByteChars(w))) {
        const id = index.get(p);
        if (id !== undefined) ids.push(id);
      }
      return { ids, dropped: [] };             // byte-level: every input is representable
    },
    decode(ids) { return utf8dec.decode(Uint8Array.from(ids.flatMap((i) => (special.has(vocab[i]) ? [] : Array.from(bytesOf(vocab[i] ?? ''))))));
    },
    token(id) { return utf8dec.decode(bytesOf(vocab[id] ?? '')); },
    label,
    /** Pre-tokenized words of a text (before merging), with their display form. */
    words(text) { return words(text).map((w) => ({ text: w, label: w.replace(/^ /, '␣') })); },
    /**
     * The real merge sequence for one word: [{pieces: [display...], merge: [a, b] | null, rank}].
     * The first state is the raw bytes; each later state applies the next lowest-rank merge.
     */
    explain(word) {
      const show = (s) => utf8dec.decode(bytesOf(s)).replace(/^ /, '␣') || '·';
      let syms = Array.from(toByteChars(word));
      const states = [{ pieces: syms.map(show), merge: null, rank: null }];
      while (syms.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < syms.length - 1; i++) {
          const r = ranks.get(`${syms[i]} ${syms[i + 1]}`);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = syms[best], b = syms[best + 1], out = [];
        for (let i = 0; i < syms.length;) {
          if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) { out.push(a + b); i += 2; } else { out.push(syms[i]); i += 1; }
        }
        syms = out;
        states.push({ pieces: syms.map(show), merge: [show(a), show(b)], rank: bestRank });
      }
      return states;
    },
    merges: ranks.size,
  };
}
