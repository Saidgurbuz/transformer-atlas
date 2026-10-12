// Tokens for a subword model: the loaded model's own tokenizer (byte-level BPE) on its real ids.
// tokens.js mounts this scene when ctx.state.tokenizer.kind is 'bpe' (the character scene is the other one there).
// Steps: Text, Words (the pre-split), Merges (one word's merge ladder), Ids, Vocabulary. Zoom target: vision.
import { stage2d, sceneHeader, sceneCaption, checkBadge, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, texHTML, reducedMotion } from '../../ui/index.js';

const STEPS = ['Text', 'Words', 'Merges', 'Ids', 'Vocabulary'];
const MERGES = 2;                                       // the index of the Merges step
const EASE = MOTION.ease;
const HX = ZONES.hero.x, HW = 1030;                     // hero column: the field, the words, the ladder, the ids, the vocabulary
const SX = ZONES.side.x, SW = ZONES.side.w;             // side column: this step's maths, a check, a note, the zoom chip
const Y_HERO = 420, Y_TEXT = 196, Y_BODY = 292;         // the field: large in step 0, compact after; the body starts under it
const SIDE_Y = 280;
const BYTE_VALUES = 256;                                // byte-level BPE: one token for each of the 256 byte values
const WORD_CAP = 120, PICK_CAP = 34, USED_CAP = 24;     // drawn words, picker words, used vocabulary entries
const WORDS_H = 470, PICK_H = 108;                      // fitted blocks (design units): the words end above the caption; the picker keeps to three rows
const LADDER_H = 370, LADDER_END = 770;                 // the ladder's height budget (design units), and where it must end (above the caption)
const HEAD_IDS = 8, TAIL_SCAN = 24, TAIL_SHOW = 4;      // vocabulary window: the first entries, the longest of the last merges
const IDS_HEAD = 32, IDS_TAIL = 8;                      // long texts: the first 32 … the last 8 ids (about four rows)
const LADDER_MAX = 8, LADDER_HEAD = 5, LADDER_TAIL = 2; // more than 8 states: the first 5 and the last 2 are drawn

const ZOOM_ICON = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21M10.5 7.5v6M7.5 10.5h6"/></svg>';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
const grouped = (n) => n.toLocaleString('en-US');       // 4096 -> "4,096"
const plural = (n, one, many = `${one}s`) => (n === 1 ? one : many);
const lenOf = (s) => Array.from(s).length;               // characters, not UTF-16 units
const byteLen = (s) => new TextEncoder().encode(s).length;
const isBlank = (s) => !/\S/.test(s);
// Display form of a piece: whitespace controls become visible glyphs (the leading space is already ␣).
const disp = (s) => String(s).replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/\r/g, '␍');
const showWord = (s) => disp(s).replace(/^ /, '␣');
const setAria = (el, hidden) => (hidden ? el.setAttribute('aria-hidden', 'true') : el.removeAttribute('aria-hidden'));
// The largest k in [lo, hi] for which fits(k) holds, by bisection: a block only grows as tiles are added.
function largestFit(lo, hi, fits) {
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (fits(mid)) lo = mid; else hi = mid - 1;
  }
  return lo;
}

useStyle('tokens-bpe', `
.tb-field { position: absolute; left: ${HX}px; width: ${HW}px; box-sizing: border-box; margin: 0; padding: 4px 2px 12px; border: 0; border-bottom: 2px solid var(--line-strong); border-radius: 0; outline: none; background: transparent; color: var(--fg); font: 500 36px/1.2 var(--font-mono); text-overflow: ellipsis; caret-color: var(--accent); cursor: text; transition: top ${MOTION.move}ms ${EASE}, font-size ${MOTION.move}ms ${EASE}, opacity .6s ease, border-color .25s ease; }
.tb-field:focus { border-bottom-color: var(--accent); }
.tb-field::placeholder { color: var(--fg-3); opacity: .8; }
.tb-scene:not([data-step="0"]) .tb-field { opacity: .6; }
.tb-scene .tb-field:hover, .tb-scene .tb-field:focus { opacity: 1; }
.tb-measure { position: absolute; left: 0; top: 0; visibility: hidden; white-space: pre; font: 500 36px/1.2 var(--font-mono); }
.tb-pane { position: absolute; width: ${HW}px; opacity: 0; visibility: hidden; transition: opacity .3s ease, visibility 0s linear .3s; }
.tb-scene[data-step="1"] .tb-pane-words, .tb-scene[data-step="2"] .tb-pane-merges, .tb-scene[data-step="3"] .tb-pane-ids, .tb-scene[data-step="4"] .tb-pane-vocab { opacity: 1; visibility: visible; transition: opacity .5s ease, visibility 0s; }
.tb-more { align-self: center; font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }

/* words: one tile per pre-token; a leading space is part of the tile */
.tb-words { display: flex; flex-wrap: wrap; gap: 10px 8px; align-items: flex-start; }
.tb-word { height: 42px; box-sizing: border-box; display: inline-flex; align-items: center; padding: 0 12px; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--fg); font: 600 19px/1 var(--font-mono); white-space: pre; box-shadow: 0 3px 8px rgba(0, 0, 0, .06); }
.tb-word.ws { color: var(--fg-3); border-style: dashed; }
.tb-scene[data-step="1"] .tb-word { animation: tbIn .5s cubic-bezier(.2, .7, .2, 1) both; animation-delay: var(--d, 0ms); }
@keyframes tbIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* merges: the picker, then the ladder of states, one row per state */
.tb-picker { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.tb-pickbox { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; max-width: 760px; }
.tb-pick { height: 32px; box-sizing: border-box; display: inline-flex; align-items: center; padding: 0 9px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface); color: var(--fg); font: 600 15px/1 var(--font-mono); white-space: pre; cursor: pointer; transition: border-color .2s ease, background-color .2s ease, transform .2s ease; }
.tb-pick:hover { border-color: var(--line-strong); transform: translateY(-1px); }
.tb-pick.ws { color: var(--fg-3); border-style: dashed; }
.tb-pick.sel { border-color: var(--c-embed); background: color-mix(in srgb, var(--c-embed) 18%, var(--surface)); }
.tb-try { width: 200px; height: 32px; box-sizing: border-box; padding: 0 10px; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--fg); font: 600 15px/1 var(--font-mono); outline: none; caret-color: var(--accent); }
.tb-try:focus { border-color: var(--accent); }
.tb-try::placeholder { color: var(--fg-3); font-weight: 500; }
.tb-ladder { margin-top: 22px; }
.tb-row { position: relative; display: flex; align-items: center; gap: 14px; height: var(--rowh, 32px); transition: opacity .35s ease, transform .35s ease; }
.tb-row + .tb-row, .tb-row + .tb-gap, .tb-gap + .tb-row { margin-top: 20px; }
.tb-row.pending, .tb-gap.pending { opacity: 0; transform: translateY(6px); }
.tb-rowlab { flex: 0 0 104px; text-align: right; font: 600 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.tb-tiles { position: relative; display: flex; align-items: center; gap: .2em; }
.tb-piece { height: var(--rowh, 32px); box-sizing: border-box; display: inline-flex; align-items: center; padding: 0 .45em; border-radius: 7px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--fg); font: 600 1em/1 var(--font-mono); white-space: pre; box-shadow: 0 2px 5px rgba(0, 0, 0, .06); transition: background-color .35s ease, border-color .35s ease; }
.tb-piece.merged { background: color-mix(in srgb, var(--c-embed) 16%, var(--surface)); border-color: color-mix(in srgb, var(--c-embed) 55%, var(--line-strong)); }
.tb-piece.src { border-color: var(--accent); background: var(--accent-soft); }
.tb-piece.new { animation: tbPop .55s cubic-bezier(.2, .7, .2, 1); border-color: var(--accent); background: var(--accent-soft); }
@keyframes tbPop { from { transform: scale(.6); opacity: 0; } to { transform: none; opacity: 1; } }
.tb-rung { position: absolute; top: -22px; left: 0; transform: translateX(-50%); white-space: nowrap; font: 600 15px/1 var(--font-mono); color: var(--accent); pointer-events: none; }
.tb-gap { height: 22px; display: flex; align-items: center; justify-content: center; font: 500 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .35s ease, transform .35s ease; }

/* ids: each token as a tile with its label and its vocabulary id under it */
.tb-ids { display: flex; flex-wrap: wrap; gap: 14px 8px; align-items: flex-start; }
.tb-id { display: grid; justify-items: center; gap: 7px; min-width: 60px; box-sizing: border-box; padding: 10px 12px 9px; border-radius: 10px; border: 1px solid var(--c-embed); background: color-mix(in srgb, var(--c-embed) 16%, var(--surface)); box-shadow: 0 3px 8px rgba(0, 0, 0, .08); }
.tb-id b { font: 650 20px/1 var(--font-mono); color: var(--fg); white-space: pre; }
.tb-id small { font: 600 14px/1 var(--font-mono); color: var(--fg-3); }
.tb-id.ell { background: none; border: 0; box-shadow: none; align-content: center; }
.tb-id.ell b { color: var(--fg-3); }
.tb-scene[data-step="3"] .tb-id { animation: tbIn .5s cubic-bezier(.2, .7, .2, 1) both; animation-delay: var(--d, 0ms); }

/* vocabulary: the composition, then a window of entries (lit = used by this text) */
.tb-comp { display: flex; height: 26px; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); background: var(--surface-2); }
.tb-comp i { display: block; height: 100%; transition: flex-grow .6s ${EASE}; }
.tb-comp .m, .tb-sw.m { background: color-mix(in srgb, var(--c-embed) 66%, var(--surface)); }
.tb-comp .b, .tb-sw.b { background: color-mix(in srgb, var(--c-embed) 40%, var(--surface)); }
.tb-comp .s, .tb-sw.s { background: var(--c-head); }
.tb-comp .s { min-width: 4px; }
.tb-legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 10px; font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); }
.tb-sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 7px; vertical-align: -1px; }
.tb-group { display: grid; gap: 9px; margin-top: 18px; }
.tb-group-lab { font: 600 13px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; }
.tb-vtiles { display: flex; flex-wrap: wrap; gap: 8px; }
.tb-vt { display: grid; justify-items: center; align-content: center; min-width: 56px; height: 44px; box-sizing: border-box; padding: 0 9px; border-radius: 9px; border: 1px solid var(--line); background: var(--surface-2); color: var(--fg-3); font: 600 17px/1 var(--font-mono); white-space: pre; transition: background-color .5s ease, border-color .5s ease, color .5s ease; }
.tb-vt small { font: 500 12px/1 var(--font-mono); margin-top: 4px; opacity: .85; }
.tb-vt.lit { background: color-mix(in srgb, var(--c-embed) 24%, var(--surface)); border-color: var(--c-embed); color: var(--fg); }
.tb-count { margin-top: 16px; font: 500 15px/1.4 var(--font-mono); color: var(--fg-3); }

/* the sub line is one line in this scene (the kit's column is narrower than this context line) */
.tb-scene .k-header { width: 1200px; }
.tb-scene .k-scene-sub { max-width: 1200px; }

/* side column */
.tb-side { position: absolute; width: ${SW}px; display: grid; gap: 12px; align-content: start; }
.tb-eq-main { font-size: 22px; line-height: 1.3; color: var(--fg); white-space: nowrap; }
.tb-eq-sub { font-size: 17px; color: var(--fg-2); white-space: nowrap; }
.tb-stats { font: 500 16px/1.5 var(--font-mono); color: var(--fg-2); }
.tb-stats b { color: var(--fg); font-weight: 650; }
.tb-rule { font-size: 17px; line-height: 1.45; color: var(--fg-2); }
.tb-note { font-size: 15px; line-height: 1.45; color: var(--fg-3); min-height: 20px; }
.tb-zoom { justify-self: start; display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 14px 0 12px; border-radius: 999px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--fg-2); font: 600 15px/1 var(--font-mono); white-space: nowrap; cursor: zoom-in; transition: transform .25s ease, box-shadow .25s ease, color .25s ease, border-color .25s ease; }
.tb-zoom:hover, .tb-zoom:focus-visible { color: var(--fg); border-color: var(--fg-3); transform: translateY(-2px); box-shadow: 0 8px 18px rgba(0, 0, 0, .12); outline: none; }
.tb-zoom-icon { display: inline-flex; color: var(--fg-3); }
@media (prefers-reduced-motion: reduce) { .tb-scene * { transition: none !important; animation: none !important; } }
`);

// What the model reads (the loaded tokenizer, the text, the ids), in one place. Word token counts come from
// the tokenizer itself: encode() is per word, so the counts add up to the whole text's tokens.
function model(ctx) {
  const s = ctx.state, tk = s.tokenizer;
  const words = tk.words(s.text);
  const wordTok = words.map((w) => tk.encode(w.text).ids.length);
  return {
    s, tk, text: s.text, ids: s.ids, T: s.ids.length, V: tk.size,
    words, wordTok, wordBytes: words.map((w) => byteLen(w.text)),
    total: wordTok.reduce((a, b) => a + b, 0), block: s.config.block_size,
    comp: composition(tk),
  };
}

// The vocabulary's make-up, from the tokenizer: 1 special entry (⟨eot⟩), the learned merges, and the rest,
// which for byte-level BPE is one entry per byte value. Cached per tokenizer.
let compOf = null, compVal = null;
function composition(tk) {
  if (compOf !== tk) {
    const specials = range(0, tk.size).filter((i) => tk.vocab[i] === '⟨eot⟩');
    compVal = { specials, special: specials.length, merges: tk.merges, bytes: tk.size - tk.merges - specials.length };
    compOf = tk;
  }
  return compVal;
}

// Which word a token index falls in, and the default word for the Merges step: the word of the selected
// token when it has 2+ bytes, else the longest word.
function longestWord(m) {
  return m.wordBytes.reduce((best, b, k) => (b > m.wordBytes[best] ? k : best), 0);
}
function defaultWord(m) {
  if (!m.words.length) return -1;
  const t = m.s.sel.token ?? 0;
  let acc = 0;
  for (let k = 0; k < m.wordTok.length; k++) {
    acc += m.wordTok[k];
    if (t < acc) return m.wordBytes[k] >= 2 ? k : longestWord(m);
  }
  return longestWord(m);
}

// The pieces of one merge: where the pair sits in the row before it, and where the fused piece lands in the row after.
// Same left-to-right scan as the tokenizer (a pair is fused where it occurs, non-overlapping).
function pairSpots(prev, a, b) {
  const srcs = [], outs = [];
  for (let i = 0, o = 0; i < prev.length; o++) {
    if (i + 1 < prev.length && prev[i] === a && prev[i + 1] === b) { srcs.push(i); outs.push(o); i += 2; }
    else i += 1;
  }
  return { srcs, outs };
}

// The merge ladder of one word, from tokenizer.explain(): every state, what is drawn, and each merge's spots.
function ladderOf(tk, text) {
  const states = tk.explain(text);
  const n = states.length - 1;
  const drawn = states.length <= LADDER_MAX ? range(0, states.length) : [...range(0, LADDER_HEAD), ...range(n - LADDER_TAIL + 1, n + 1)];
  const steps = states.map((st, k) => {
    if (k === 0) return null;
    const [a, b] = st.merge;
    return { a, b, rank: st.rank, ...pairSpots(states[k - 1].pieces, a, b) };
  });
  return { states, n, drawn, hidden: states.length - drawn.length, steps, final: states[n].pieces };
}

// The model's own embedding rows, checked against W_E for every id (as in the character scene).
function embedErr(s) {
  const tok = s.trace?.['emb.tok'], W = s.model?.params?.['wte.weight'];
  const T = s.ids.length, d = W?.shape?.[1];
  if (!tok || !W || tok.shape?.[0] !== T || tok.shape?.[1] !== d) return null;
  let worst = 0;
  for (let t = 0; t < T; t++) {
    const at = t * d, row = s.ids[t] * d;
    for (let c = 0; c < d; c++) worst = Math.max(worst, Math.abs(tok.data[at + c] - W.data[row + c]));
  }
  return worst;
}

// Captions: one or two sentences per step, numbers from the live tokenizer and ids.
const CAPS = [
  (m) => (m.total === 0
    ? 'The text is empty, so the model reads one token: ⟨eot⟩, the end-of-text entry (id 0).'
    : 'The model reads integers, not letters. Before the first layer runs, this model’s own tokenizer maps the text to a list of ids.'),
  (m) => (m.words.length === 0
    ? 'The text is empty, so there are no words.'
    : `Before any merging, a fixed pattern splits the text into <b>${m.words.length} ${plural(m.words.length, 'word')}</b>. A space stays with the word after it, shown as ␣.`),
  () => 'Each word starts as its UTF-8 bytes. Repeatedly, the adjacent pair with the lowest rank fuses into one piece; the pieces left at the end are the tokens.',
  (m) => (m.T === 1
    ? 'Each token is replaced by its index in the vocabulary. This <b>1 integer</b> is the model’s real input: the embedding table looks up one row per id.'
    : `Each token is replaced by its index in the vocabulary. These <b>${m.T} integers</b> are the model’s real input: the embedding table looks up one row per id.`),
  (m, used) => `The vocabulary is a fixed table of <b>${grouped(m.V)}</b> entries: ${grouped(m.comp.bytes)} byte tokens, <b>${grouped(m.comp.merges)}</b> learned merges and ${m.comp.special === 1 ? 'one special token' : `${m.comp.special} special tokens`}. This text uses <b>${used}</b> of them.`,
];

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    st.root.classList.add('tb-scene');
    st.root.dataset.step = '0';
    let step = 0, prevStep = 0, dead = false, away = false;
    let pending = false, debounce = null, auto = null, autoTick = false;
    let lastCh = null, lastL = null, lastSame = false;                 // the Merges step's chosen word, its ladder, its check
    let ladderKey = null, ladderMs = 0, timers = [], rows = new Map();
    let pickerKey = null, wordsKey = null, idsKey = null, vocabKey = null;
    let pick = null, typed = null, typeTimer = null;                   // the reader's word: picked from the words, or typed
    let capText = null;

    // ------------------------------------------------------------ shared parts (kit)
    const header = sceneHeader(st, { chips: [], title: 'Tokens', sub: 'A tokenizer turns text into a sequence of integers. This model uses byte-level BPE, so most words are one token.' });
    const cap = sceneCaption(st, { y: ZONES.caption.y, w: 1060 });
    // The kit cross-fades captions. A newer caption cancels a fade still running, so only the latest text is ever written.
    const capSet = (html) => {
      if (html === capText) return;
      capText = html;
      if (reducedMotion()) { cap.el.innerHTML = html; return; }
      cap.el.getAnimations().forEach((a) => a.cancel());
      cap.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140 }).finished.then(() => {
        if (html !== capText) return;
        cap.el.innerHTML = html;
        cap.el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
      }).catch(() => {});
    };

    // ------------------------------------------------------------ the input: an inline field styled as text
    const field = h('input', { class: 'tb-field', type: 'text', spellcheck: 'false', autocomplete: 'off', placeholder: 'type a sentence', 'aria-label': 'Input text', value: ctx.state.text });
    const measure = h('span', { class: 'tb-measure', 'aria-hidden': 'true' });
    st.add(field, { x: HX, y: Y_TEXT });
    st.add(measure, { x: HX, y: 0 });

    // ------------------------------------------------------------ the four bodies (one is shown per step)
    const paneWords = h('div', { class: 'tb-pane tb-pane-words', role: 'group', 'aria-label': 'Words' });
    const wordsBox = h('div', { class: 'tb-words' });
    paneWords.append(wordsBox);

    const paneMerges = h('div', { class: 'tb-pane tb-pane-merges', role: 'group', 'aria-label': 'Merges' });
    const pickBox = h('div', { class: 'tb-pickbox' });
    const tryInput = h('input', { class: 'tb-try', type: 'text', spellcheck: 'false', autocomplete: 'off', placeholder: 'unbelievable', maxlength: '40', 'aria-label': 'Type a word to analyse' });
    const picker = h('div', { class: 'tb-picker' }, h('span', { class: 'k-picker-label' }, 'word'), pickBox, tryInput);
    const ladder = h('div', { class: 'tb-ladder' });
    paneMerges.append(picker, ladder);

    const paneIds = h('div', { class: 'tb-pane tb-pane-ids', role: 'group', 'aria-label': 'Token ids' });
    const idsBox = h('div', { class: 'tb-ids' });
    paneIds.append(idsBox);

    const paneVocab = h('div', { class: 'tb-pane tb-pane-vocab', role: 'group', 'aria-label': 'Vocabulary' });
    const vocabBody = h('div', { class: 'tb-vbody' });
    paneVocab.append(vocabBody);

    st.add(paneWords, { x: HX, y: Y_BODY });
    st.add(paneMerges, { x: HX, y: Y_BODY });
    st.add(paneIds, { x: HX, y: Y_BODY });
    st.add(paneVocab, { x: HX, y: Y_BODY });

    // side column: this step's maths, a check, a note, and the zoom chip (vision) at its foot
    const side = h('div', { class: 'tb-side' });
    st.add(side, { x: SX, y: SIDE_Y });
    const zoom = h('button', { type: 'button', class: 'tb-zoom', dataset: { node: 'vision' }, title: 'Vision tokens: how images become tokens' },
      'Images as tokens', h('span', { class: 'tb-zoom-icon', html: ZOOM_ICON }));

    // ------------------------------------------------------------ text editing (debounced setText, as in the character scene)
    const cancelAuto = () => { clearTimeout(auto); auto = null; };
    const flush = () => {
      clearTimeout(debounce); debounce = null;
      if (!pending || dead) return;
      pending = false;
      ctx.store.setText(field.value, { quiet: true });
    };
    field.addEventListener('input', () => { pending = true; cancelAuto(); clearTimeout(debounce); debounce = setTimeout(flush, 280); });
    field.addEventListener('focus', cancelAuto);
    field.addEventListener('blur', flush);
    field.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === 'Escape') { flush(); field.blur(); } });
    // The walk belongs to the first view only: the first press or key from the reader hands the controls over.
    const interact = () => cancelAuto();
    document.addEventListener('pointerdown', interact, true);
    window.addEventListener('keydown', interact, true);
    el.addEventListener('atlas:interrupt', cancelAuto);
    tryInput.addEventListener('focus', cancelAuto);
    // A typed word is analysed after a short pause; the first word of what is typed is the one analysed.
    tryInput.addEventListener('input', () => {
      clearTimeout(typeTimer);
      typeTimer = setTimeout(() => {
        const found = ctx.state.tokenizer.words(tryInput.value).find((w) => !isBlank(w.text));
        typed = found ? found.text : null;
        paint({ animate: true });
      }, 260);
    });

    // ------------------------------------------------------------ measuring the field (font size at which the text fills the hero)
    function fontsFor(text) {
      measure.textContent = text || ' ';
      const w36 = Math.max(1, measure.getBoundingClientRect().width / st.scale);
      const fit = Math.floor((36 * HW) / w36);
      return { hero: clamp(fit, 18, 56), compact: clamp(fit, 18, 36) };
    }

    // ------------------------------------------------------------ words (step 1)
    // As many tiles as fit above the caption; the rest are counted. The tiles change with the text only; their stagger plays on entry (CSS).
    function paintWords(m) {
      if (wordsKey === m.text) return;
      wordsKey = m.text;
      const n = m.words.length;
      const build = (k) => {
        const tiles = range(0, k).map((i) => { const w = m.words[i]; return h('span', { class: ['tb-word', isBlank(w.text) && 'ws'], style: { '--d': `${Math.min(i * 28, 900)}ms` }, title: `word ${i + 1}` }, disp(w.label)); });
        if (n > k) tiles.push(h('span', { class: 'tb-more' }, `+${n - k} more`));
        wordsBox.replaceChildren(...tiles);
      };
      build(largestFit(0, Math.min(n, WORD_CAP), (k) => { build(k); return wordsBox.offsetHeight <= WORDS_H; }));
    }

    // ------------------------------------------------------------ merges (step 2): the chosen word, its picker and its ladder
    // The word the step analyses: a typed word, else the word picked, else the default word.
    function chosenWord(m) {
      if (typed) return { text: typed, index: -1, typed: true };
      if (pick && m.words[pick.index]?.text === pick.text) return { text: pick.text, index: pick.index, typed: false };
      const k = defaultWord(m);
      return k < 0 ? null : { text: m.words[k].text, index: k, typed: false };
    }

    // The first words of the text, as many as fit in two rows (so the ladder below keeps its room); the analysed word is always offered.
    function renderPicker(m, ch) {
      const n = m.words.length;
      const build = (k) => {
        const idx = range(0, k);
        if (ch && ch.index >= k && ch.index < n) idx.push(ch.index);
        const tiles = idx.map((i) => {
          const w = m.words[i], on = ch?.index === i;
          const b = h('button', { type: 'button', class: ['tb-pick', on && 'sel', isBlank(w.text) && 'ws'], title: `word ${i + 1}`, 'aria-pressed': String(on) }, disp(w.label));
          b.addEventListener('click', () => { pick = { index: i, text: w.text }; typed = null; tryInput.value = ''; paint({ animate: true }); });
          return b;
        });
        if (n > idx.length) tiles.push(h('span', { class: 'tb-more' }, `+${n - idx.length} more`));
        pickBox.replaceChildren(...tiles);
      };
      build(largestFit(0, Math.min(n, PICK_CAP), (k) => { build(k); return pickBox.offsetHeight <= PICK_H; }));
    }

    function clearTimers() {
      timers.forEach(clearTimeout);
      timers = [];
    }
    const after = (ms, fn) => { timers.push(setTimeout(fn, ms)); };

    // Draws the ladder for one word. Animated: row 0 is the bytes; each merge highlights its pair in the row above,
    // then the next row appears with the fused piece; merges that are not drawn pass quickly.
    function renderLadder(m, ch, L, animate) {
      clearTimers();
      rows = new Map();
      ladderMs = 0;
      ladder.replaceChildren();
      if (!L) { ladder.append(h('div', { class: 'tb-gap' }, m.words.length ? 'Click a word above, or type one.' : 'The text has no words: type one below to analyse it.')); return; }
      const bytes = byteLen(ch.text);
      // the rows share the ladder's height; the font is as large as the widest row allows beside its label
      // (pieces take ~0.62em per character, plus padding and gaps), and no larger than the row is tall
      const R = L.drawn.length + (L.hidden > 0 ? 1 : 0) + (L.n === 0 ? 1 : 0);
      const budget = Math.min(LADDER_H, LADDER_END - Y_BODY - ladder.offsetTop);   // less when the word picker wraps to more rows
      const rowH = clamp(Math.floor((budget - 20 * (R - 1)) / R), 26, 56);
      const units = Math.max(...L.drawn.map((k) => L.states[k].pieces.reduce((a, p) => a + lenOf(p) * 0.62 + 0.9 + 0.2, -0.2)));
      const fs = Math.min(Math.floor(((HW - 150) / Math.max(1, units)) * 10) / 10, rowH * 0.5, 28);
      ladder.style.setProperty('--rowh', `${rowH}px`);
      ladder.style.fontSize = `${clamp(fs, 11, 28)}px`;
      let elision = null;
      const els = [];
      L.drawn.forEach((k, i) => {
        if (i === LADDER_HEAD && L.hidden > 0) {
          elision = h('div', { class: ['tb-gap', 'pending'] }, `⋯ ${L.hidden} ${plural(L.hidden, 'merge')} not shown ⋯`);
          els.push(elision);
        }
        const piece = L.states[k].pieces;
        const tiles = h('div', { class: 'tb-tiles' }, piece.map((p) => h('span', { class: ['tb-piece', lenOf(p) > 1 && 'merged'] }, disp(p))));
        const label = k === 0 ? `${bytes} ${plural(bytes, 'byte')}` : k === L.n ? `${piece.length} ${plural(piece.length, 'token')}` : '';
        const row = h('div', { class: ['tb-row', k > 0 && 'pending'] }, h('span', { class: 'tb-rowlab' }, label), tiles);
        if (k > 0) {
          const sp = L.steps[k];
          const rung = h('span', { class: 'tb-rung' }, `merge #${sp.rank} · ${disp(sp.a)} + ${disp(sp.b)}${sp.outs.length > 1 ? ` (×${sp.outs.length})` : ''}`);
          row.append(rung);
          rows.set(k, { row, tiles, rung });
        } else rows.set(k, { row, tiles, rung: null });
        els.push(row);
      });
      if (L.n === 0) els.push(h('div', { class: 'tb-gap' }, 'No adjacent pair is in the merge list, so the bytes stay as they are.'));
      ladder.replaceChildren(...els);
      // each rung sits over the fused piece it made, clamped to the ladder's width
      for (const [k, r] of rows) {
        if (!r.rung) continue;
        const first = r.tiles.children[L.steps[k].outs[0]];
        if (!first) continue;
        const x = r.tiles.offsetLeft + first.offsetLeft + first.offsetWidth / 2, half = r.rung.offsetWidth / 2;
        r.rung.style.left = `${clamp(x, half, r.row.offsetWidth - half)}px`;
      }
      if (!animate || reducedMotion() || L.n === 0) {
        els.forEach((e) => e.classList.remove('pending'));
        return;
      }
      const markSource = (k) => {
        const prev = rows.get(k - 1);
        if (!prev) return;
        for (const i of L.steps[k].srcs) { prev.tiles.children[i]?.classList.add('src'); prev.tiles.children[i + 1]?.classList.add('src'); }
      };
      const reveal = (k) => {
        const r = rows.get(k);
        if (!r) return;
        r.row.classList.remove('pending');
        for (const o of L.steps[k].outs) r.tiles.children[o]?.classList.add('new');
        rows.get(k - 1)?.tiles.querySelectorAll('.src').forEach((el) => el.classList.remove('src'));
      };
      const drawn = new Set(L.drawn);
      const D = clamp(3000 / L.n, 200, 440), DH = clamp(1200 / L.n, 100, 170);
      let t = 350;
      for (let k = 1; k <= L.n; k++) {
        if (drawn.has(k)) {
          if (drawn.has(k - 1)) after(t, () => markSource(k));
          const at = t + D * 0.6;
          after(at, () => reveal(k));
          t += D;
        } else {
          if (k === LADDER_HEAD) {                          // the first merge not drawn: its pair lights up, then the elision appears
            if (drawn.has(k - 1)) after(t, () => markSource(k));
            // the merges in the middle are not drawn: the elision marker appears, and the pair above it lets go
            after(t + DH * 0.6, () => {
              elision?.classList.remove('pending');
              rows.get(k - 1)?.tiles.querySelectorAll('.src').forEach((el) => el.classList.remove('src'));
            });
          }
          t += DH;
        }
      }
      // the fused highlight is for the moment of the merge: once the ladder is drawn, the tiles keep only their tint
      after(t + 200, () => rows.forEach((r) => r.tiles.querySelectorAll('.new').forEach((el) => el.classList.remove('new'))));
      ladderMs = t + 600;
    }

    function paintMerges(m, animate) {
      const ch = chosenWord(m);
      lastCh = ch;
      const pk = `${m.text}\u0001${ch ? ch.index : -2}`;
      if (pk !== pickerKey) { pickerKey = pk; renderPicker(m, ch); }
      const lk = ch ? ch.text : '';
      if (lk !== ladderKey || !lastL) {
        ladderKey = lk;
        lastL = ch ? ladderOf(m.tk, ch.text) : null;
        // the check: the ladder's final pieces are the tokenizer's own pieces for this word
        const want = ch ? m.tk.encode(ch.text).ids.map((id) => m.tk.label(id)) : [];
        const got = lastL ? lastL.final.map(disp) : [];
        lastSame = got.length === want.length && got.every((p, i) => p === want[i]);
        renderLadder(m, ch, lastL, animate);
      }
    }

    // ------------------------------------------------------------ ids (step 3)
    function paintIds(m) {
      const key = m.ids.join(',');
      if (key === idsKey) return;
      idsKey = key;
      const T = m.T, elide = T > IDS_HEAD + IDS_TAIL + 1;   // an elision hides at least two tokens
      const items = elide ? [...range(0, IDS_HEAD), -1, ...range(T - IDS_TAIL, T)] : range(0, T);
      idsBox.replaceChildren(...items.map((t, i) => {
        if (t < 0) return h('div', { class: 'tb-id ell', title: `${T - IDS_HEAD - IDS_TAIL} tokens not shown` }, h('b', null, '⋯'), h('small', null, `${T - IDS_HEAD - IDS_TAIL} more`));
        const id = m.ids[t];
        return h('div', { class: 'tb-id', style: { '--d': `${Math.min(i * 40, 900)}ms` }, title: `token ${t}: id ${id}, row ${id} of W_E` }, h('b', null, disp(m.tk.label(id))), h('small', null, String(id)));
      }));
    }

    // ------------------------------------------------------------ vocabulary (step 4): the make-up, and a window of entries
    function paintVocab(m) {
      const key = `${m.V}\u0001${m.ids.join(',')}`;
      if (key === vocabKey) return;
      vocabKey = key;
      const { tk, V, comp } = m;
      const used = [...new Set(m.ids)], usedSet = new Set(used);
      // the window: the special entry and the first byte tokens (the printable bytes come first), the entries this text uses,
      // and the longest of the last merges (the rarest ones)
      const head = [...comp.specials, ...range(1, Math.min(V, HEAD_IDS + 1)).filter((i) => !comp.specials.includes(i))];
      const shown = new Set(head);
      const usedRest = used.filter((i) => !shown.has(i)).slice(0, USED_CAP);
      usedRest.forEach((i) => shown.add(i));
      const tail = range(Math.max(0, V - TAIL_SCAN), V).filter((i) => !shown.has(i))
        .sort((a, b) => lenOf(tk.vocab[b]) - lenOf(tk.vocab[a]) || b - a).slice(0, TAIL_SHOW).sort((a, b) => a - b);
      tail.forEach((i) => shown.add(i));
      const hidden = V - shown.size;
      const tile = (i) => h('div', { class: ['tb-vt', usedSet.has(i) && 'lit'], title: `id ${i}` }, disp(tk.vocab[i]), h('small', null, String(i)));
      const group = (label, ids) => (ids.length ? h('div', { class: 'tb-group' }, h('div', { class: 'tb-group-lab' }, label), h('div', { class: 'tb-vtiles' }, ids.map(tile))) : null);
      const bar = (cls, n) => h('i', { class: cls, style: { flexGrow: String(n) } });
      vocabBody.replaceChildren(...[
        h('div', { class: 'tb-comp', role: 'img', 'aria-label': `${comp.merges} merges, ${comp.bytes} byte tokens, ${comp.special} special` }, bar('m', comp.merges), bar('b', comp.bytes), bar('s', comp.special)),
        h('div', { class: 'tb-legend' },
          h('span', null, h('i', { class: ['tb-sw', 'm'] }), `${grouped(comp.merges)} learned merges`),
          h('span', null, h('i', { class: ['tb-sw', 'b'] }), `${grouped(comp.bytes)} byte tokens`),
          h('span', null, h('i', { class: ['tb-sw', 's'] }), `${comp.special} special ${plural(comp.special, 'entry', 'entries')}`)),
        group('first entries · the special one, then the bytes', head),
        group(`used by this text · ${used.length} of ${grouped(V)}`, usedRest),
        group('the end of the list · the longest of the last merges', tail),
        h('div', { class: 'tb-count' }, `Showing ${shown.size} of ${grouped(V)} entries; ${grouped(hidden)} are not shown.`),
      ].filter(Boolean));
    }

    // ------------------------------------------------------------ side column: this step's maths, a check, a note, the zoom chip
    function sideFor(m) {
      const [main, sub] = mathFor(m);
      const kids = [h('div', { class: 'k-label', style: { textAlign: 'left' } }, STEPS[step]), h('div', { class: 'tb-eq-main', html: texHTML(main) }), h('div', { class: 'tb-eq-sub', html: texHTML(sub) })];
      if (step === 2 && lastCh && lastL) {
        const n = lastL.n, bytes = byteLen(lastCh.text), tokens = lastL.final.length;
        kids.push(h('div', { class: 'tb-stats' },
          h('div', null, '“', h('b', null, showWord(lastCh.text)), '”'),
          h('div', null, `${bytes} ${plural(bytes, 'byte')} → ${tokens} ${plural(tokens, 'token')} · ${n} ${plural(n, 'merge')}`)));
        kids.push(h('span', { class: ['k-check', lastSame ? 'ok' : 'bad'] }, lastSame ? '✓ same pieces as the tokenizer' : '✗ differs from the tokenizer'));
        kids.push(h('div', { class: 'tb-rule' }, 'Rule: repeatedly merge the adjacent pair with the lowest rank, until no adjacent pair is in the merge list.'));
      }
      if (step === 3 && m.T) {
        const err = embedErr(ctx.state);
        if (err != null) kids.push(checkBadge(err, 0, { label: 'every id reads its row of W_E' }));
      }
      if (step === 4) {
        const ok = m.comp.bytes === BYTE_VALUES;
        kids.push(h('span', { class: ['k-check', ok ? 'ok' : 'bad'] }, ok ? `✓ ${BYTE_VALUES} byte tokens, one per byte value` : `✗ ${m.comp.bytes} byte tokens, not ${BYTE_VALUES}`));
      }
      kids.push(h('div', { class: 'tb-note' }, notesFor(m)), zoom);
      return kids;
    }

    function mathFor(m) {
      switch (step) {
        case 0: return ['\\mathrm{tok}\\colon\\mathit{text}\\mapsto\\mathbb{N}^{T}', `T=${m.T}`];
        case 1: return ['\\mathrm{words}(\\mathit{text})=(w_1,\\dots,w_N)', `N=${m.words.length}`];
        case 2: return ['\\mathrm{merge}(a,b)=ab', '(a,b)=\\arg\\min\\,\\mathrm{rank}(a,b)'];
        case 3: return ['\\mathit{idx}_t=\\mathrm{id}(p_t)', `\\mathit{idx}\\in\\mathbb{N}^{${m.T}}`];
        default: return [`\\mathrm{id}\\colon\\mathcal{V}\\to\\{0,\\dots,${m.V - 1}\\}`, `${m.V}=${m.comp.special}+${m.comp.bytes}+${m.comp.merges}`];
      }
    }

    // The side note: the edit hint on every step, what the text did to the model's input, then this step's point.
    // Short: one point per step, plus what the text itself did (empty, cut to the context window).
    function notesFor(m) {
      const bits = [];
      if (step === 0) {
        bits.push('Click the text to edit it. Each change reruns the model.');
        if (/[A-Z]/.test(m.text)) bits.push('Upper case is kept: the tokenizer does not lower-case the text.');
      }
      if (!m.total && step === 0) bits.push('Empty text: the model reads one token, ⟨eot⟩ (id 0).');
      if (m.total > m.block && (step === 0 || step === 3)) bits.push(`Cut to the ${m.block}-token context window: the first ${m.block} of ${grouped(m.total)} tokens are read.`);
      bits.push([null, 'Punctuation is split off as a word of its own.',
        'Click a word, or type one. Lower rank means learned earlier, so more frequent.',
        'The number under each piece is its index in the vocabulary.',
        'Every byte has its own token, so any text can be encoded and nothing is skipped.'][step]);
      return bits.filter(Boolean).join(' ');
    }

    // The parts that are off screen at this step are also out of the accessibility tree.
    function syncAria() {
      setAria(paneWords, step !== 1);
      setAria(paneMerges, step !== MERGES);
      setAria(paneIds, step !== 3);
      setAria(paneVocab, step !== 4);
    }

    // ------------------------------------------------------------ paint
    function paint({ animate = true } = {}) {
      if (dead) return;
      const s = ctx.state;
      if (!pending && document.activeElement !== field && field.value !== s.text) field.value = s.text;
      const m = model(ctx);
      const f = fontsFor(field.value);
      field.style.top = `${step === 0 ? Y_HERO : Y_TEXT}px`;
      field.style.fontSize = `${step === 0 ? f.hero : f.compact}px`;
      header.setChips([[`V = ${grouped(m.V)}`, 'embed'], [`T = ${m.T}`, 'resid']]);
      if (step === 1) paintWords(m);
      if (step === MERGES) paintMerges(m, animate);
      if (step === 3) paintIds(m);
      if (step === 4) paintVocab(m);
      side.replaceChildren(...sideFor(m).filter(Boolean));
      capSet(CAPS[step](m, new Set(m.ids).size));
      syncAria();
    }

    function setStep(i) {
      if (dead) return;
      if (!autoTick) cancelAuto();          // a step the reader picked (a dot, an arrow key) ends the walk
      prevStep = step;
      step = clamp(i, 0, STEPS.length - 1);
      if (prevStep === MERGES && step !== MERGES) clearTimers();
      if (step === MERGES) { pickerKey = null; ladderKey = null; lastL = null; }   // entering Merges draws the ladder again, with its motion
      st.root.dataset.step = String(step);
      paint({ animate: true });
      if (!reducedMotion()) side.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 480, easing: 'ease-out' });
    }

    // ------------------------------------------------------------ start
    paint({ animate: false });
    ctx.setSteps(STEPS, setStep);
    appear([header.el, field, side], { stagger: MOTION.stagger });
    document.fonts?.ready.then(() => { if (!dead && !away) { pickerKey = null; wordsKey = null; paint({ animate: false }); } });   // re-measure once the mono face is loaded: the fitted blocks depend on it
    // explain itself once: one step every MOTION.step, the Merges step for as long as its ladder takes
    const dwell = () => (step === MERGES ? ladderMs + 500 : MOTION.step);
    const walk = () => {
      if (step >= STEPS.length - 1) return;
      autoTick = true;
      try { ctx.setStep(step + 1); } finally { autoTick = false; }
      auto = setTimeout(walk, dwell());
    };
    auto = setTimeout(walk, 900);
    const off = ctx.on('trace', () => { if (!away) paint({ animate: false }); });
    const offSel = ctx.on('select', () => { if (!away && step === MERGES && !pick && !typed) paint({ animate: true }); });

    return {
      destroy() {
        off();                              // first: the flush below must not repaint this scene
        offSel();
        flush();
        dead = true;
        cancelAuto();
        clearTimers();
        clearTimeout(typeTimer);
        document.removeEventListener('pointerdown', interact, true);
        window.removeEventListener('keydown', interact, true);
        el.removeEventListener('atlas:interrupt', cancelAuto);   // the layer outlives a swap: leave no listener behind
        st.destroy();
      },
      // a ladder paused mid-motion would keep its unrevealed rows hidden: it is drawn again, whole, on resume
      pause() { away = true; flush(); cancelAuto(); clearTimers(); ladderKey = null; },
      resume() { away = false; paint({ animate: false }); },   // re-measure: the field was hidden while a pending edit flushed
    };
  },
};
