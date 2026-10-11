// Root scene: the whole model as a clean 3D object you can orbit.
// Tokens at the bottom, the embedding plate, one plate per block (attention
// module with each head's real attention pattern on top, MLP module with its
// neuron activity), the residual stream rising behind them, and the next-token
// prediction above the LM head. Hover a part to see what it is; click to zoom in.
import { SceneEngine, THREE, label, themeColor, ease } from '../../scene/engine3d.js';
import { h, reducedMotion } from '../../ui/index.js';

const WIDTH = 15;          // plate width (x)
const DEPTH = 5.4;         // plate depth (z)
const GAP = 2.7;           // vertical distance between blocks

export default {
  mount(el, ctx) {
    const host = h('div', { class: 'atlas-3d' });
    el.append(host);
    const nodeOf = new Map();       // mesh -> {node, trace, label}
    const hot = { mesh: null };
    const engine = new SceneEngine(host, {
      onHover: (info) => setHover(info),
      onHoverMove: () => {},
      onPick: () => {},
      onTheme: () => build(),
    });
    engine.controls.minDistance = 8;
    engine.controls.maxDistance = 90;
    const root = new THREE.Group();
    engine.scene.add(root);
    let parts = [];                 // {mesh, node, kind, baseEmissive}
    let labels = [];
    let blockLabels = [];           // per block: module labels shown while that block is hovered
    let tubeMat = null, headTex = [];
    let interacted = false;
    engine.controls.addEventListener('start', () => { interacted = true; });

    function mat(kind, { opacity = 0.92, emissive = 0.18 } = {}) {
      const c = themeColor(`--c-${kind}`);
      return new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: emissive, roughness: 0.45, metalness: 0.05, transparent: opacity < 1, opacity });
    }
    function edges(mesh, kind, opacity = 0.8) {
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color: themeColor(`--c-${kind}`), transparent: true, opacity }));
      mesh.add(e);
      return e;
    }
    function addPart(mesh, { node, kind, trace = null, title }) {
      mesh.userData = { node, kind, trace, title };
      nodeOf.set(mesh, mesh.userData);
      engine.addPickable(mesh);
      parts.push({ mesh, node, kind, base: mesh.material.emissiveIntensity ?? 0 });
      return mesh;
    }
    function addLabel(html, pos, { kind = 'resid', side = 'left', cls = 'm3-label' } = {}) {
      const l = label(`<span class="${cls}" style="--kind: var(--c-${kind})">${html}</span>`, {});
      l.position.copy(pos);
      l.center.set(side === 'left' ? 1 : side === 'right' ? 0 : 0.5, 0.5);
      root.add(l);
      labels.push(l);
      return l;
    }

    // ---------------------------------------------------------------- build
    function build() {
      // clear
      for (const p of parts) engine.removePickable(p.mesh);
      parts = []; nodeOf.clear();
      for (const l of labels) l.userData.el.remove();
      labels = [];
      blockLabels = [];
      headTex.forEach((t) => t.dispose()); headTex = [];
      root.traverse((o) => { o.geometry?.dispose?.(); if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.dispose(); }); });
      root.clear();

      const st = ctx.state, cfg = st.config, T = st.ids.length, L = cfg.n_layer;
      const tokSpace = Math.min(1.0, (WIDTH - 1) / Math.max(1, T));
      const xTok = (t) => (t - (T - 1) / 2) * tokSpace;
      let y = 0;

      // tokens
      const tokGroup = new THREE.Group();
      const tokGeo = new THREE.BoxGeometry(tokSpace * 0.78, 0.16, 0.9);
      for (let t = 0; t < T; t++) {
        const m = new THREE.Mesh(tokGeo, mat('embed', { opacity: 1, emissive: 0.25 }));
        m.position.set(xTok(t), y, 1.2);
        tokGroup.add(m);
        if (T <= 40 || t % Math.ceil(T / 40) === 0) {
          const l = label(`<span class="m3-token">${escape(ctx.label(t))}</span>`);
          l.position.set(xTok(t), y + 0.05, 2.0);
          root.add(l); labels.push(l);
        }
      }
      root.add(tokGroup);
      const tokHit = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.4, 1.4), new THREE.MeshBasicMaterial({ visible: false }));
      tokHit.position.set(0, y, 1.2);
      root.add(tokHit);
      addPart(tokHit, { node: 'tokens', kind: 'embed', title: 'Tokens' });
      addLabel(`<i></i>Tokens<small>T=${T}</small>`, new THREE.Vector3(-WIDTH / 2 - 0.6, y, 1.2), { kind: 'embed' });

      // embedding plate
      y += 1.5;
      const emb = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.32, DEPTH), mat('embed', { opacity: 0.88 }));
      emb.position.set(0, y, 0);
      edges(emb, 'embed');
      root.add(emb);
      addPart(emb, { node: 'embedding', kind: 'embed', trace: 'emb.out', title: 'Embedding' });
      addLabel(`<i></i>Embedding<small>V=${cfg.vocab_size} → d=${cfg.d_model}</small>`, new THREE.Vector3(-WIDTH / 2 - 0.6, y, 0), { kind: 'embed' });
      const yEmb = y;

      // blocks
      const blockY = [];
      for (let i = 0; i < L; i++) {
        y += GAP;
        blockY.push(y);
        const plate = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.22, DEPTH), mat('resid', { opacity: 0.35, emissive: 0.08 }));
        plate.position.set(0, y, 0);
        edges(plate, 'resid', 0.55);
        root.add(plate);
        addPart(plate, { node: `block.${i}`, kind: 'resid', title: `Block ${i}` });
        addLabel(`<i></i>Block ${i}`, new THREE.Vector3(-WIDTH / 2 - 0.6, y, 0), { kind: 'resid' });

        // attention module with real head patterns on top
        const aw = WIDTH / 2 - 0.9, ad = DEPTH - 1.2;
        const attn = new THREE.Mesh(new THREE.BoxGeometry(aw, 0.9, ad), mat('attn', { opacity: 0.9, emissive: 0.16 }));
        attn.position.set(-WIDTH / 4, y + 0.56, 0);
        edges(attn, 'attn', 0.9);
        root.add(attn);
        addPart(attn, { node: `attention.${i}`, kind: 'attn', trace: `L${i}.attn.probs`, title: `Attention · block ${i}` });
        const probs = st.trace[`L${i}.attn.probs`];
        if (probs) {
          const H = probs.shape[0], tile = Math.min((aw - 0.6) / H - 0.12, ad - 0.6);
          for (let hh = 0; hh < H; hh++) {
            const tex = patternTexture(probs, hh);
            headTex.push(tex);
            const pl = new THREE.Mesh(new THREE.PlaneGeometry(tile, tile), new THREE.MeshBasicMaterial({ map: tex, transparent: true }));
            pl.rotation.x = -Math.PI / 2;
            pl.position.set(-WIDTH / 4 - (aw - 0.6) / 2 + (hh + 0.5) * ((aw - 0.6) / H), y + 1.02, 0);
            root.add(pl);
            addPart(pl, { node: `head.${i}.${hh}`, kind: 'attn', trace: `L${i}.attn.probs`, title: `Head ${hh} · block ${i}` });
          }
        }
        const la = addLabel(`<i></i>Attention<small>${cfg.n_head} heads${cfg.n_kv_head < cfg.n_head ? ` · ${cfg.n_kv_head} KV` : ''}</small>`, new THREE.Vector3(-WIDTH / 4, y + 1.35, -ad / 2), { kind: 'attn', side: 'center' });

        // MLP module with neuron activity bars
        const mlp = new THREE.Mesh(new THREE.BoxGeometry(aw, 0.9, ad), mat('mlp', { opacity: 0.9, emissive: 0.16 }));
        mlp.position.set(WIDTH / 4, y + 0.56, 0);
        edges(mlp, 'mlp', 0.9);
        root.add(mlp);
        addPart(mlp, { node: `mlp.${i}`, kind: 'mlp', trace: `L${i}.mlp.act`, title: `MLP · block ${i}` });
        const act = st.trace[`L${i}.mlp.act`];
        if (act) {
          const F = act.shape[1], B = 24, per = Math.ceil(F / B), mean = new Array(B).fill(0);
          for (let t = 0; t < act.shape[0]; t++) for (let f = 0; f < F; f++) mean[Math.floor(f / per)] += Math.abs(act.data[t * F + f]);
          const mx = Math.max(...mean) || 1;
          const barGeo = new THREE.BoxGeometry((aw - 0.8) / B * 0.7, 1, 0.5);
          mean.forEach((v, b) => {
            const hgt = 0.08 + 0.75 * (v / mx);
            const bar = new THREE.Mesh(barGeo, mat('mlp', { opacity: 1, emissive: 0.35 + 0.5 * v / mx }));
            bar.scale.y = hgt;
            bar.position.set(WIDTH / 4 - (aw - 0.8) / 2 + (b + 0.5) * ((aw - 0.8) / B), y + 1.01 + hgt / 2, 0);
            root.add(bar);
          });
        }
        const lm = addLabel(`<i></i>MLP<small>${cfg.d_model}→${cfg.d_ff}→${cfg.d_model}</small>`, new THREE.Vector3(WIDTH / 4, y + 1.35, -ad / 2), { kind: 'mlp', side: 'center' });
        la.visible = lm.visible = false;
        blockLabels.push([la, lm]);
      }

      // final norm, LM head
      if (cfg.norm_placement === 'pre') {
        y += GAP * 0.8;
        const ln = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.14, DEPTH), mat('norm', { opacity: 0.7 }));
        ln.position.set(0, y, 0);
        edges(ln, 'norm', 0.6);
        root.add(ln);
        addPart(ln, { node: 'final-norm', kind: 'norm', trace: 'lnf', title: 'Final norm' });
        addLabel('<i></i>Final norm', new THREE.Vector3(-WIDTH / 2 - 0.6, y, 0), { kind: 'norm' });
      }
      y += GAP * 0.8;
      const head = new THREE.Mesh(new THREE.BoxGeometry(WIDTH, 0.3, DEPTH), mat('head', { opacity: 0.9 }));
      head.position.set(0, y, 0);
      edges(head, 'head');
      root.add(head);
      addPart(head, { node: 'lm-head', kind: 'head', trace: 'logits', title: 'LM head' });
      addLabel(`<i></i>LM head<small>d=${cfg.d_model} → V=${cfg.vocab_size}</small>`, new THREE.Vector3(-WIDTH / 2 - 0.6, y, 0), { kind: 'head' });
      const yHead = y;

      // next-token prediction for the last position
      const probs = st.trace.probs, V = cfg.vocab_size, last = T - 1;
      const row = Array.from(probs.data.subarray(last * V, (last + 1) * V));
      const top = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 5);
      const predGroup = new THREE.Group();
      top.forEach((v, rnk) => {
        const p = row[v], hgt = 0.2 + 3.2 * p;
        const bar = new THREE.Mesh(new THREE.BoxGeometry(1.1, hgt, 1.1), mat('head', { opacity: 1, emissive: 0.25 + 0.6 * p }));
        const x = (rnk - 2) * 2.0;
        bar.position.set(x, yHead + 0.25 + hgt / 2, 0);
        predGroup.add(bar);
        const l = label(`<span class="m3-pred">“${escape(st.tokenizer.label(v))}” ${(p * 100).toFixed(1)}%</span>`);
        l.position.set(x, yHead + 0.45 + hgt, 0);
        root.add(l); labels.push(l);
      });
      root.add(predGroup);
      const predHit = new THREE.Mesh(new THREE.BoxGeometry(10.5, 3.8, 1.6), new THREE.MeshBasicMaterial({ visible: false }));
      predHit.position.set(0, yHead + 2.1, 0);
      root.add(predHit);
      addPart(predHit, { node: 'sampling', kind: 'head', trace: 'probs', title: `Next token after “${ctx.label(last)}”` });
      addLabel('<i></i>Next token', new THREE.Vector3(-WIDTH / 2 - 0.6, yHead + 2.2, 0), { kind: 'head' });

      // residual stream: a glowing tube behind the plates with read/write branches
      const tubeX = 0, tubeZ = -DEPTH / 2 - 0.9;
      const curve = new THREE.LineCurve3(new THREE.Vector3(tubeX, yEmb, tubeZ), new THREE.Vector3(tubeX, yHead, tubeZ));
      tubeMat = new THREE.ShaderMaterial({
        transparent: true,
        uniforms: { time: { value: 0 }, color: { value: themeColor('--c-resid') }, glow: { value: themeColor('--accent') } },
        vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
        fragmentShader: `uniform float time; uniform vec3 color; uniform vec3 glow; varying vec2 vUv;
          void main(){ float p = fract(vUv.x * 3.0 - time * 0.35); float pulse = smoothstep(0.0, 0.08, p) * (1.0 - smoothstep(0.08, 0.3, p));
            gl_FragColor = vec4(mix(color, glow, pulse * 0.8), 0.55 + 0.45 * pulse);
            #include <colorspace_fragment>
          }`,
      });
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 64, 0.22, 20, false), tubeMat);
      root.add(tube);
      addLabel('<i></i>Residual stream', new THREE.Vector3(tubeX + 0.5, (yEmb + yHead) / 2, tubeZ), { kind: 'resid', side: 'right' });
      const branchMat = new THREE.MeshBasicMaterial({ color: themeColor('--c-resid'), transparent: true, opacity: 0.45 });
      blockY.forEach((by) => {
        for (const x of [-WIDTH / 4, WIDTH / 4]) {
          const c = new THREE.QuadraticBezierCurve3(new THREE.Vector3(tubeX, by - 0.6, tubeZ), new THREE.Vector3(x * 0.6, by - 0.2, tubeZ + 0.4), new THREE.Vector3(x, by + 0.5, -DEPTH / 2 + 0.6));
          root.add(new THREE.Mesh(new THREE.TubeGeometry(c, 24, 0.05, 8, false), branchMat));
          const c2 = new THREE.QuadraticBezierCurve3(new THREE.Vector3(x, by + 1.0, -DEPTH / 2 + 0.6), new THREE.Vector3(x * 0.6, by + 1.4, tubeZ + 0.4), new THREE.Vector3(tubeX, by + 1.6, tubeZ));
          root.add(new THREE.Mesh(new THREE.TubeGeometry(c2, 24, 0.05, 8, false), branchMat));
        }
      });

      bounds = new THREE.Box3().setFromObject(root);
      engine.invalidate();
    }

    function patternTexture(probs, hh) {
      const T = probs.shape[1], S = Math.max(T, 2), px = new Uint8Array(S * S * 4);
      const lo = themeColor('--heat-lo'), hi = themeColor('--c-attn');
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const v = Math.pow(probs.data[(hh * T + i) * T + j], 0.6);
        const c = lo.clone().lerp(hi, v).convertLinearToSRGB();
        const o = ((T - 1 - i) * S + j) * 4;
        px[o] = c.r * 255; px[o + 1] = c.g * 255; px[o + 2] = c.b * 255; px[o + 3] = 255;
      }
      const tex = new THREE.DataTexture(px, S, S, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      return tex;
    }

    // ---------------------------------------------------------------- hover & click
    function setHover(info) {
      const d = info ? nodeOf.get(info.object) : null;
      if (hot.mesh && hot.mesh !== info?.object) {
        const p = parts.find((x) => x.mesh === hot.mesh);
        if (p?.mesh.material.emissiveIntensity !== undefined) p.mesh.material.emissiveIntensity = p.base;
      }
      hot.mesh = d ? info.object : null;
      const bi = d ? /^(?:block|attention|mlp|head)\.(\d+)/.exec(d.node)?.[1] : null;
      blockLabels.forEach((pair, i) => pair.forEach((l) => { l.visible = String(i) === bi; }));
      const canvas = engine.renderer.domElement;
      if (d) {
        if (info.object.material.emissiveIntensity !== undefined) info.object.material.emissiveIntensity = 0.55;
        canvas.dataset.node = d.node;
        if (d.trace) canvas.dataset.trace = d.trace; else delete canvas.dataset.trace;
      } else {
        delete canvas.dataset.node;
        delete canvas.dataset.trace;
      }
      engine.invalidate();
    }

    engine.renderer.domElement.addEventListener('click', (e) => {
      const info = engine.pick(e);
      const d = info && nodeOf.get(info.object);
      e.preventDefault();          // the atlas does not handle this click; we zoom ourselves
      if (!d) return;
      const target = info.object.getWorldPosition(new THREE.Vector3());
      const dir = engine.camera.position.clone().sub(engine.controls.target).normalize();
      const dist = engine.camera.position.distanceTo(engine.controls.target);
      engine.flyTo({ target, position: target.clone().add(dir.multiplyScalar(dist * 0.45)), duration: reducedMotion() ? 0 : 520 })
        .then(() => ctx.go(d.node, engine.screenRect(info.object)));
    });
    // right-click: the atlas menu reads data-trace / data-node from the canvas (set on hover)

    // ---------------------------------------------------------------- idle motion
    let bounds = null, t0 = performance.now();
    const offFrame = engine.onFrame((t) => {
      if (tubeMat) tubeMat.uniforms.time.value = t;
      if (!interacted && bounds && !reducedMotion()) {
        const c = bounds.getCenter(new THREE.Vector3());
        const a = Math.sin((performance.now() - t0) / 5200) * 0.22;
        const r = home.radius;
        engine.camera.position.set(c.x + Math.sin(home.az + a) * r * home.cosEl, c.y + home.h, c.z + Math.cos(home.az + a) * r * home.cosEl);
        engine.controls.target.copy(c);
      }
      return true;
    });

    const home = { az: 0.55, radius: 30, cosEl: 0.93, h: 6 };
    function frameHome(duration = 0) {
      const f = engine.framing(bounds, new THREE.Vector3(Math.sin(home.az), 0.34, Math.cos(home.az)), 0.86);
      const off = f.position.clone().sub(f.target);
      home.radius = Math.hypot(off.x, off.z) / home.cosEl;
      home.h = off.y;
      return engine.flyTo({ ...f, duration });
    }

    build();
    engine.setInsets({ top: 60, bottom: 30, left: 0, right: 0 });
    frameHome(0);
    const offTrace = ctx.on('trace', () => { build(); });

    return {
      pause: () => engine.pause(),
      resume: () => { engine.resume(); interacted = true; frameHome(700).then(() => { interacted = false; t0 = performance.now(); }); },
      destroy() { offFrame(); offTrace(); labels.forEach((l) => l.userData.el.remove()); headTex.forEach((t) => t.dispose()); engine.dispose(); host.remove(); },
    };
  },
};

function escape(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
