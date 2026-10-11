// Character-level tokenizer: one token per character of a fixed vocabulary.
// Real LLMs use subword (BPE) vocabularies of 32k-256k entries; the mechanics
// after the id lookup are identical.

export const DEFAULT_VOCAB = " abcdefghijklmnopqrstuvwxyz.,!?'";

export function makeTokenizer(vocab = DEFAULT_VOCAB) {
  const chars = Array.from(vocab);
  const index = new Map(chars.map((c, i) => [c, i]));
  return {
    vocab: chars,
    size: chars.length,
    // Returns {ids, dropped}: characters outside the vocabulary are skipped and reported.
    encode(text) {
      const ids = [], dropped = [];
      for (const ch of Array.from(text.toLowerCase())) {
        const id = index.get(ch);
        if (id === undefined) dropped.push(ch); else ids.push(id);
      }
      return { ids, dropped };
    },
    decode(ids) { return ids.map((i) => chars[i] ?? '?').join(''); },
    token(id) { return chars[id] ?? '?'; },
    // Printable label for a token chip (space shown as a visible glyph).
    label(id) { const c = chars[id]; return c === ' ' ? '␣' : c ?? '?'; },
  };
}
