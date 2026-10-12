// Tokens: how text becomes integers, on the real tokenizer and the real ids (ctx.state.ids).
// The input text is split into character tiles; each tile flips to show its vocabulary id; the ids
// drop into the integer row the model reads; the 32-entry vocabulary shows which entries this text
// uses. A toy BPE, labelled as an illustration, shows subword pieces. Zoom target: vision.
import { stage2d, sceneHeader, sceneCaption, checkBadge, card, token, chip, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, texHTML, reducedMotion, slider } from '../../ui/index.js';

const STEPS = ['Text', 'Characters', 'Ids', 'Vocabulary', 'Subwords'];
const EASE = MOTION.ease;
const HX = ZONES.hero.x, HW = 1030;                  // hero column: the text, the tiles, the vocabulary
const SX = ZONES.side.x, SW = ZONES.side.w;          // side column: this step's maths, a check, a note, the zoom card
const Y_HERO = 420, Y_TEXT = 196;                    // the input: large in the middle (step 0), then compact
const LAB_Y = 304;                                   // row labels sit 26 units above their row
const Y_TOP = 330, Y_IDS = 500, IDS_LAB_Y = 472;     // characters, then the integer row they flip into
const Y_VOCAB = 660, VOCAB_LAB_Y = 636;              // the 32-entry vocabulary, under the integer row
const SIDE_Y = 330, CARD_Y = 630;
const TILE_H = 50, ROW_GAP = 8, GAP = 6;             // tile height, row gap, column gap
const PER_ROW = 32, SHOW = 64;                       // at most 32 tiles a row; longer runs elide the middle
const VOCAB_COLS = 16, MAX_MERGES = 12;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const show = (t) => String(t).replace(/ /g, '␣');   // a space as a visible glyph
const strong = (x) => h('b', null, String(x));
const grouped = (n) => n.toLocaleString('en-US');    // 1050 -> "1,050"
const plural = (n, one) => (n === 1 ? one : `${one}s`);
const usedIds = (ids) => [...new Set(ids)];          // first-use order (a Set keeps insertion order)

useStyle('tokens', `
.tk-field { position: absolute; left: ${HX}px; width: ${HW}px; box-sizing: border-box; margin: 0; padding: 4px 2px 12px; border: 0; border-bottom: 2px solid var(--line-strong); border-radius: 0; outline: none; background: transparent; color: var(--fg); font: 500 36px/1.2 var(--font-mono); caret-color: var(--accent); cursor: text; transition: top ${MOTION.move}ms ${EASE}, font-size ${MOTION.move}ms ${EASE}, opacity .6s ease, border-color .25s ease; }
.tk-field:focus { border-bottom-color: var(--accent); }
.tk-field::placeholder { color: var(--fg-3); opacity: .8; }
.tk-scene:not([data-step="0"]) .tk-field { opacity: .4; }
.tk-scene .tk-field:hover, .tk-scene .tk-field:focus { opacity: 1; }
.tk-measure { position: absolute; left: 0; top: 0; visibility: hidden; white-space: pre; font: 500 36px/1.2 var(--font-mono); }
.tk-board { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.tk-tile { position: absolute; transition: left ${MOTION.move}ms ${EASE}, top ${MOTION.move}ms ${EASE}, opacity .5s ease, transform ${MOTION.move}ms ${EASE}; }
.tk-tile.off { pointer-events: none; }
.tk-flip { position: absolute; inset: 0; transform-style: preserve-3d; transition: transform .6s ${EASE}; }
.tk-face { position: absolute; inset: 0; box-sizing: border-box; border-radius: 11px; backface-visibility: hidden; -webkit-backface-visibility: hidden; }
.tk-front { width: auto; height: auto; min-width: 0; padding: 0 2px; font-size: 22px; }
.tk-front small { font-size: 12px; margin-top: 3px; }
.tk-back { display: grid; place-items: center; align-content: center; gap: 3px; transform: rotateY(180deg); background: color-mix(in srgb, var(--c-embed) 22%, var(--surface)); border: 1px solid var(--c-embed); box-shadow: 0 4px 10px rgba(0, 0, 0, .12); font: 650 18px/1 var(--font-mono); color: var(--fg); }
.tk-back small { font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.tk-ell { display: grid; place-items: center; background: none; border: 0; box-shadow: none; font: 600 22px/1 var(--font-mono); color: var(--fg-3); }
.tk-lab { position: absolute; font: 600 15px/1 var(--font-mono); color: var(--fg-3); letter-spacing: .02em; white-space: nowrap; opacity: 0; transition: opacity .5s ease; }
.tk-scene[data-step="1"] .tk-lab-chars, .tk-scene[data-step="2"] .tk-lab-ids, .tk-scene[data-step="3"] .tk-lab-ids, .tk-scene[data-step="3"] .tk-lab-vocab, .tk-scene[data-step="4"] .tk-lab-bpe { opacity: 1; }
.tk-vocab { position: absolute; display: grid; grid-template-columns: repeat(${VOCAB_COLS}, 58px); gap: 8px 6px; }
.tk-vc { display: grid; justify-items: center; align-content: center; height: 46px; box-sizing: border-box; border-radius: 9px; border: 1px solid var(--line); background: var(--surface-2); color: var(--fg-3); font: 600 20px/1 var(--font-mono); opacity: 0; transition: opacity .5s ease, background-color .5s ease, border-color .5s ease, color .5s ease, box-shadow .5s ease; }
.tk-vc small { font: 500 12px/1 var(--font-mono); margin-top: 4px; opacity: .85; }
.tk-vc.lit { background: color-mix(in srgb, var(--c-embed) 24%, var(--surface)); border-color: var(--c-embed); color: var(--fg); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c-embed) 18%, transparent); }
.tk-scene[data-step="3"] .tk-vc { opacity: .42; }
.tk-scene[data-step="3"] .tk-vc.lit { opacity: 1; }
.tk-side { position: absolute; width: ${SW}px; display: grid; gap: 12px; align-content: start; }
.tk-step { font: 600 13px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; }
.tk-eq-main { font-size: 22px; line-height: 1.3; color: var(--fg); white-space: nowrap; }
.tk-eq-sub { font-size: 17px; color: var(--fg-2); white-space: nowrap; }
.tk-note { font-size: 15px; line-height: 1.45; color: var(--fg-3); min-height: 20px; }
.tk-bpe { position: absolute; width: ${HW}px; display: grid; gap: 16px; opacity: 0; pointer-events: none; transition: opacity .6s ease; }
.tk-scene[data-step="4"] .tk-bpe { opacity: 1; pointer-events: auto; }
.tk-pieces { display: flex; flex-wrap: wrap; gap: 8px; }
.tk-piece { padding: 9px 12px 8px; border-radius: 10px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--fg); font: 600 18px/1 var(--font-mono); white-space: pre; box-shadow: 0 3px 8px rgba(0, 0, 0, .08); transition: background-color .4s ease, border-color .4s ease; }
.tk-piece.merged { background: var(--accent-soft); border-color: var(--accent); }
.tk-piece.new { animation: tkPop .6s cubic-bezier(.2, .7, .2, 1); }
@keyframes tkPop { from { transform: scale(.7); opacity: 0; } to { transform: none; opacity: 1; } }
.tk-stats { font: 500 16px/1.5 var(--font-mono); color: var(--fg-2); }
.tk-stats b { color: var(--fg); font-weight: 650; }
.tk-ctl { width: 460px; }
.tk-last { font-size: 17px; line-height: 1.45; color: var(--fg-2); min-height: 26px; }
.tk-last b { color: var(--fg); }
@media (prefers-reduced-motion: reduce) { .tk-scene * { transition: none !important; animation: none !important; } }
`);

// Greedy toy BPE with GPT-style pre-tokenisation: a space opens a new word and stays on its front
// (" the"), so no merge crosses a word edge. Each step merges the most frequent adjacent pair inside
// the words. Ties go to the pair whose first occurrence is leftmost (Map keeps insertion order).
// Stops when no pair occurs twice. Each step returns the flat piece list it leaves behind.
function toyBpe(symbols, maxMerges) {
  let words = [], cur = [];
  for (const sym of symbols) {
    if (sym === ' ' && cur.length) { words.push(cur); cur = []; }
    cur.push(sym);
  }
  if (cur.length) words.push(cur);
  const steps = [];
  for (let k = 0; k < maxMerges; k++) {
    const counts = new Map();
    for (const w of words) {
      for (let i = 0; i + 1 < w.length; i++) {
        const key = `${w[i]}\u0001${w[i + 1]}`;
        const e = counts.get(key);
        if (e) e.n += 1; else counts.set(key, { a: w[i], b: w[i + 1], n: 1 });
      }
    }
    let best = null;
    for (const e of counts.values()) if (!best || e.n > best.n) best = e;
    if (!best || best.n < 2) break;
    const merged = best.a + best.b;
    words = words.map((w) => {
      const next = [];
      for (let i = 0; i < w.length;) {
        if (i + 1 < w.length && w[i] === best.a && w[i + 1] === best.b) { next.push(merged); i += 2; }
        else { next.push(w[i]); i += 1; }
      }
      return next;
    });
    steps.push({ a: best.a, b: best.b, n: best.n, merged, seq: words.flat() });
  }
  return steps;
}

// Step 1 states both counts: the characters typed, and the tokens the model reads (the context window may cut them).
function charsCaption(m, c) {
  if (!c.chars) return 'The text is empty, so the model reads one placeholder token.';
  if (!c.enc) return `None of the <b>${grouped(c.chars)}</b> ${plural(c.chars, 'character')} is in the vocabulary, so the model reads one placeholder token.`;
  const lead = 'Each character in the vocabulary becomes one token.';
  const tokens = `<b>${m.T}</b> ${plural(m.T, 'token')}`;
  if (c.cut) return `${lead} This text has <b>${grouped(c.chars)}</b> characters; the model reads the first ${tokens}.`;
  if (c.enc < c.chars) return `${lead} This text has <b>${grouped(c.chars)}</b> characters, <b>${c.enc}</b> of them in the vocabulary, so the model reads ${tokens}.`;
  return `${lead} This text has <b>${grouped(c.chars)}</b> ${plural(c.chars, 'character')}, so the model reads ${tokens}.`;
}

const CAPS = [
  () => 'A model reads integers, not letters. Before the first layer runs, a tokenizer maps this text to a list of ids.',
  (m, c) => charsCaption(m, c),
  (m) => (m.T === 1
    ? 'Each token is replaced by its index in the vocabulary. This <b>1 integer</b> is the model’s real input: the embedding table looks up one row per id.'
    : `Each token is replaced by its index in the vocabulary. These <b>${m.T} integers</b> are the model’s real input: the embedding table looks up one row per id.`),
  (m, c, n) => `The vocabulary is a fixed table of ${m.V} entries, the model’s whole alphabet. Lit entries are the ones this text uses: <b>${n} of ${m.V}</b>.`,
  () => 'Illustration, not this model: merging the most frequent neighbouring pair builds subword pieces. Real LLMs use vocabularies of about 32k to 256k entries, so many common words are one token.',
];

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    st.root.classList.add('tk-scene');
    st.root.dataset.step = '0';
    let step = 0, prevStep = 0, lastM = null, dead = false, away = false;
    let pending = false, debounce = null, auto = null, autoTick = false;
    let tiles = [], tileKey = '';
    let vocabKey = '', cells = [];
    let bpeK = 4, bpeKey = '', bpe = { syms: [], steps: [] }, shown = new Set();

    // ------------------------------------------------------------ shared parts (kit)
    const header = sceneHeader(st, { chips: [], title: 'Tokens', sub: 'A tokenizer turns text into a sequence of integers; in this model, one token is one character.' });
    const cap = sceneCaption(st, { y: ZONES.caption.y, w: 1060 });

    // ------------------------------------------------------------ the input: an inline field styled as text
    const field = h('input', { class: 'tk-field', type: 'text', spellcheck: 'false', autocomplete: 'off', placeholder: 'type a sentence', 'aria-label': 'Input text', value: ctx.state.text });
    const measure = h('span', { class: 'tk-measure', 'aria-hidden': 'true' });
    st.add(field, { x: HX, y: Y_TEXT });
    st.add(measure, { x: HX, y: 0 });

    const board = h('div', { class: 'tk-board' });
    st.add(board, { x: 0, y: 0 });

    const labChars = h('div', { class: 'tk-lab tk-lab-chars' });
    const labIds = h('div', { class: 'tk-lab tk-lab-ids' }, 'idx · the model’s input ids');
    const labVocab = h('div', { class: 'tk-lab tk-lab-vocab' });
    const labBpe = h('div', { class: 'tk-lab tk-lab-bpe' }, 'subword pieces · illustration');
    st.add(labChars, { x: HX, y: LAB_Y });
    st.add(labIds, { x: HX, y: IDS_LAB_Y });
    st.add(labVocab, { x: HX, y: VOCAB_LAB_Y });
    st.add(labBpe, { x: HX, y: LAB_Y });

    const vocabWrap = h('div', { class: 'tk-vocab', role: 'group', 'aria-label': 'Vocabulary' });
    st.add(vocabWrap, { x: HX, y: Y_VOCAB });

    // subword pieces (illustration) with a small slider for the number of merges
    const pieces = h('div', { class: 'tk-pieces' });
    const stats = h('div', { class: 'tk-stats' });
    const last = h('div', { class: 'tk-last' });
    const merges = slider({
      id: 'tokens-bpe-merges', label: 'Merges applied', min: 0, max: 0, step: 1, value: 0,
      format: (v) => `${v} ${v === 1 ? 'merge' : 'merges'}`,
      onInput: (v) => { bpeK = v; renderBpe(); },
    });
    const bpeBox = h('div', { class: 'tk-bpe' }, pieces, stats, h('div', { class: 'tk-ctl' }, merges.el), last);
    st.add(bpeBox, { x: HX, y: Y_TOP });

    // side column: this step's maths, a check, a note; then the zoom card
    const side = h('div', { class: 'tk-side' });
    st.add(side, { x: SX, y: SIDE_Y });
    const zoom = card({ title: 'Images as tokens', sub: 'A ViT cuts an image into patches, and each patch becomes a token.', kind: 'embed', node: 'vision', w: SW });
    st.add(zoom, { x: SX, y: CARD_Y });

    // ------------------------------------------------------------ text editing (debounced setText)
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
    // The walk belongs to the first view only: the first press or key from the reader hands the
    // controls over. The story bar sits outside this element, so the listeners go on the document.
    const interact = () => cancelAuto();
    document.addEventListener('pointerdown', interact, true);
    window.addEventListener('keydown', interact, true);

    // ------------------------------------------------------------ model -> geometry
    function model() {
      const s = ctx.state, tk = s.tokenizer, ids = s.ids, T = ids.length;
      const slots = T <= SHOW
        ? ids.map((id, t) => ({ t, id }))
        : [...ids.slice(0, SHOW - 4).map((id, t) => ({ t, id })), { t: -1, id: -1 },
          ...ids.slice(T - 3).map((id, k) => ({ t: T - 3 + k, id }))];
      return { ids, T, slots, V: tk.size, vocab: tk.vocab, tk };
    }

    function geometry(n) {
      const rows = Math.ceil(n / PER_ROW), per = Math.ceil(n / rows);
      const w = clamp(Math.floor((HW - (per - 1) * GAP) / per), 24, 64);
      return {
        w,
        x: (k) => HX + (k % per) * (w + GAP),
        y: (k, base) => base + Math.floor(k / per) * (TILE_H + ROW_GAP),
      };
    }

    // width of the text at 36 px, in design units (the field's own font, measured)
    function measureAt36(text) {
      measure.textContent = text || ' ';
      return Math.max(1, measure.getBoundingClientRect().width / st.scale);
    }

    // where tile k sits in each step, whether it shows its id, and how visible it is
    function target(stp, k, slot, g, adv) {
      if (stp === 0) return { x: Math.min(HX + k * adv, HX + HW - g.w), y: Y_HERO + 6, o: 0, s: 0.5, flip: false };
      const x = g.x(k);
      if (stp === 1) return { x, y: g.y(k, Y_TOP), o: 1, s: 1, flip: false };
      if (stp === 2 || stp === 3) return { x, y: g.y(k, Y_IDS), o: 1, s: 1, flip: slot.t >= 0 };
      return { x, y: g.y(k, Y_IDS), o: 0, s: 0.6, flip: slot.t >= 0 };      // subwords: the illustration takes the stage
    }

    // staggered timing: characters split out one by one; each flips to its id, then drops into the row
    function delaysFor(stp, k, from) {
      if (stp === 0) return { pos: k * 12, op: 0, flip: 0 };
      if (stp === 1) return { pos: k * 24, op: k * 24, flip: 0 };
      if (stp === 2) return from < 2 ? { pos: 260 + k * 30, op: 0, flip: k * 30 } : { pos: 0, op: k * 12, flip: 0 };
      if (stp === 3) return { pos: 0, op: k * 12, flip: 0 };
      return { pos: 0, op: 0, flip: 0 };
    }

    // ------------------------------------------------------------ tiles
    function buildTile(slot, m) {
      const ell = slot.t < 0;
      let front;
      if (ell) front = h('div', { class: 'tk-face tk-front tk-ell' }, '⋯');
      else { front = token(show(m.tk.token(slot.id)), slot.t, { kind: 'embed' }); front.classList.add('tk-face', 'tk-front'); }
      // the back shows the id and the character it stands for, so the mapping reads before the vocabulary step
      const back = ell ? null : h('div', { class: 'tk-face tk-back' }, strong(slot.id), h('small', null, show(m.tk.token(slot.id))));
      const flip = h('div', { class: 'tk-flip' }, front, back);
      const outer = h('div', { class: 'tk-tile' }, flip);
      board.append(outer);
      return { outer, flip, slot };
    }

    function setTile(tile, tg, w, d, animate) {
      const { outer, flip, slot } = tile;
      outer.style.width = `${w}px`;
      outer.style.height = `${TILE_H}px`;
      outer.style.left = `${tg.x}px`;
      outer.style.top = `${tg.y}px`;
      outer.style.opacity = String(tg.o);
      outer.style.transform = `scale(${tg.s})`;
      outer.classList.toggle('off', tg.o === 0);
      flip.style.transform = tg.flip ? 'rotateY(180deg)' : 'none';
      // a flipped tile is an integer, not an embedding: it names its row of W_E, and carries no trace key
      if (tg.flip) outer.title = `id ${slot.id}: row ${slot.id} of W_E`; else outer.removeAttribute('title');
      const p = animate ? d.pos : 0, o = animate ? d.op : 0, f = animate ? d.flip : 0;
      outer.style.transitionDelay = `${p}ms, ${p}ms, ${o}ms, ${p}ms`;
      flip.style.transitionDelay = `${f}ms`;
    }

    function syncTiles(m, g, adv, animate) {
      const key = m.slots.map((sl) => `${sl.t}:${sl.id}`).join(',');
      if (key !== tileKey) {
        tileKey = key;
        board.replaceChildren();
        tiles = m.slots.map((slot) => buildTile(slot, m));
        animate = false;                  // new tiles appear where they belong
      }
      tiles.forEach((tile, k) => setTile(tile, target(step, k, tile.slot, g, adv), g.w, delaysFor(step, k, prevStep), animate));
    }

    // ------------------------------------------------------------ vocabulary (32 entries, used ones lit)
    function paintVocab(m, usedList) {
      const key = m.vocab.join('\u0001');
      if (key !== vocabKey) {
        vocabKey = key;
        vocabWrap.replaceChildren(...m.vocab.map((c, i) => h('div', { class: 'tk-vc' }, strong(show(c)), h('small', null, String(i)))));
        cells = Array.from(vocabWrap.children);
      }
      const order = new Map(usedList.map((id, n) => [id, n]));
      cells.forEach((cell, i) => {
        const n = order.get(i), lit = n !== undefined;
        cell.classList.toggle('lit', lit);
        const fade = step === 3 ? i * 14 : 0, light = lit ? 240 + n * 70 : 0;
        cell.style.transitionDelay = `${fade}ms, ${light}ms, ${light}ms, ${light}ms, ${light}ms`;
      });
    }

    // ------------------------------------------------------------ subwords (illustration on this text)
    function computeBpe(m) {
      const key = m.ids.join(',');
      if (key !== bpeKey) {
        bpeKey = key;
        const syms = m.ids.map((id) => m.tk.token(id));
        bpe = { syms, steps: toyBpe(syms, MAX_MERGES) };
      }
      const max = bpe.steps.length;
      bpeK = clamp(bpeK, 0, max);
      merges.input.max = String(max);
      merges.input.disabled = max === 0;
      merges.set(bpeK);
    }

    function renderBpe() {
      if (!lastM) return;
      const { syms, steps } = bpe, k = bpeK, V = lastM.V;
      const seq = k ? steps[k - 1].seq : syms;
      pieces.replaceChildren(...seq.map((p) => h('span', { class: ['tk-piece', p.length > 1 && 'merged', k && !shown.has(p) && 'new'] }, show(p))));
      shown = new Set(seq);
      const n = syms.length, np = seq.length;
      stats.replaceChildren(strong(n), ` ${plural(n, 'character')} → `, strong(np),
        ` ${plural(np, 'token')} (${(n / Math.max(1, np)).toFixed(2)} characters per token) · illustrative vocabulary ${V} + `, strong(k), ' = ', strong(V + k), ' entries');
      if (!steps.length) last.textContent = 'No pair occurs twice in this text, so there is nothing to merge.';
      else if (!k) last.textContent = 'No merges applied: every character is its own token.';
      else {
        const r = steps[k - 1];
        last.replaceChildren(`Merge ${k}: `, strong(`“${show(r.a)}” + “${show(r.b)}” → “${show(r.merged)}”`),
          `, the most frequent adjacent pair, seen ${r.n} times.`);
      }
    }

    // ------------------------------------------------------------ text for the step (truth on screen)
    // The side note: the edit hint on every step, then whatever the text did to the model's input.
    function noteFor(s, m, c) {
      const bits = ['Click the text to edit it. Each change reruns the model.'];
      if (!c.chars) bits.push(`Empty input: the model gets one token, “${show(m.vocab[0])}” (id 0).`);
      else if (!c.enc) bits.push(`None of these characters is in the vocabulary: the model gets one token, “${show(m.vocab[0])}” (id 0).`);
      if (/[A-Z]/.test(s.text)) bits.push('Upper case is lowered first.');
      const drop = [...new Set(s.dropped)];
      if (drop.length) bits.push(`Skipped, not in the vocabulary: ${drop.slice(0, 12).map((ch) => `“${show(ch)}”`).join(' ')}${drop.length > 12 ? ' …' : ''}.`);
      if (c.cut) bits.push(`Cut to the ${s.config.block_size}-token context window.`);
      if (m.T > SHOW) bits.push(`${m.T} tokens: the middle is elided here.`);
      if (s.config.vocab_size !== m.V) bits.push(`The model has ${s.config.vocab_size} rows; the tokenizer has ${m.V} entries.`);
      return bits.join(' ');
    }

    // this step's maths: a main line and a short second line, numbers from the live state
    function sideMaths(m, n) {
      const T = m.T, V = m.V, last = V - 1;
      switch (step) {
        case 0: return ['\\mathit{idx}\\in\\mathbb{N}^{T}', `T=${T}`];
        case 1: return ['c_t\\in\\mathcal{V}', `|\\mathcal{V}|=${V}`];
        case 2: return ['\\mathit{idx}_t=\\mathrm{id}(c_t)', `\\mathit{idx}\\in\\mathbb{N}^{${T}}`];
        case 3: return [`\\mathrm{id}\\colon\\mathcal{V}\\to\\{0,\\dots,${last}\\}`, `\\text{one-to-one, ${n} of ${V} used}`];
        default: return ['\\mathrm{merge}(a,b)=ab', '(a,b)=\\arg\\max\\#(a,b)'];
      }
    }

    // ------------------------------------------------------------ paint
    function paint({ animate = true } = {}) {
      if (dead) return;
      const s = ctx.state;
      if (!pending && document.activeElement !== field && field.value !== s.text) field.value = s.text;
      const m = model();
      lastM = m;
      const enc = s.tokenizer.encode(s.text).ids;          // every in-vocabulary character, before the context cut
      const c = { chars: Array.from(s.text).length, enc: enc.length, cut: enc.length > m.T };
      const n = usedIds(m.ids).length;
      const g = geometry(m.slots.length);
      const w36 = measureAt36(field.value);
      const fit = Math.floor((36 * HW) / w36);                // the size at which the text fills the column
      const fsHero = clamp(fit, 18, 56), fsCompact = clamp(fit, 18, 36);
      const adv = (w36 / Math.max(1, field.value.length)) * (fsHero / 36);
      field.style.top = `${step === 0 ? Y_HERO : Y_TEXT}px`;
      field.style.fontSize = `${step === 0 ? fsHero : fsCompact}px`;
      syncTiles(m, g, adv, animate);

      header.el.querySelector('.k-eyebrow').replaceChildren(
        chip(`V = ${m.V}`, 'embed'), chip(`T = ${m.T}`, 'resid'), ...(step === 4 ? [chip('illustration')] : []));
      labChars.textContent = c.cut ? `characters · first ${m.T} of ${grouped(c.chars)}` : `characters · ${m.T} ${plural(m.T, 'token')}`;
      labVocab.textContent = `vocabulary · ${m.V} entries · ${n} used`;
      paintVocab(m, usedIds(m.ids));
      if (step === 4) { computeBpe(m); renderBpe(); }

      const [main, sub] = sideMaths(m, n);
      const recon = (enc.length ? enc : [0]).slice(0, s.config.block_size).join(',');   // the ids the model should read, recomputed
      side.replaceChildren(...[                              // replaceChildren would print a null child as text
        h('div', { class: 'tk-step' }, STEPS[step]),
        h('div', { class: 'tk-eq-main', html: texHTML(main) }),
        h('div', { class: 'tk-eq-sub', html: texHTML(sub) }),
        step === 2 ? checkBadge(recon, s.ids.join(','), { label: 'matches the model’s input ids' }) : null,
        h('div', { class: 'tk-note' }, noteFor(s, m, c)),
      ].filter(Boolean));
      cap.set(CAPS[step](m, c, n));
    }

    function setStep(i) {
      if (dead) return;
      if (!autoTick) cancelAuto();          // a step the reader picked (a dot, an arrow key) ends the walk
      prevStep = step;
      step = clamp(i, 0, STEPS.length - 1);
      st.root.dataset.step = String(step);
      paint({ animate: true });
      if (!reducedMotion()) side.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 480, easing: 'ease-out' });
    }

    // ------------------------------------------------------------ start
    paint({ animate: false });
    ctx.setSteps(STEPS, setStep);
    appear([header.el, field, side, zoom], { stagger: MOTION.stagger });
    document.fonts?.ready.then(() => { if (!dead && !away) paint({ animate: false }); });   // re-measure once the mono face is loaded
    // explain itself once: one step every 1.7 s until the reader takes over
    const walk = () => {
      if (step >= STEPS.length - 1) return;
      autoTick = true;
      try { ctx.setStep(step + 1); } finally { autoTick = false; }
      auto = setTimeout(walk, MOTION.step);
    };
    auto = setTimeout(walk, 900);
    const off = ctx.on('trace', () => { if (!away) paint({ animate: false }); });

    return {
      destroy() {
        off();                              // first: the flush below must not repaint this scene
        flush();
        dead = true;
        cancelAuto();
        document.removeEventListener('pointerdown', interact, true);
        window.removeEventListener('keydown', interact, true);
        st.destroy();
      },
      pause() { away = true; flush(); cancelAuto(); },
      resume() { away = false; paint({ animate: false }); },   // re-measure: the field was hidden while a pending edit flushed
    };
  },
};
