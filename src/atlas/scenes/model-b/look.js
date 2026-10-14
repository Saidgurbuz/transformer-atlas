// The look of the root view (variant b): one studio, two moods.
// Light: saturated jewel-tone bodies under a soft overhead softbox, white "screens" for data,
// a pale radial backdrop and a blurred contact shadow. Dark: the same bodies, deepened, with the
// data (attention patterns, neuron tops, the stream, the prediction) emitting light.
// Both themes render directly. Colours come from the app's CSS tokens and follow theme changes.
import { THREE, themeColor } from '../../../scene/engine3d.js';

const METAL = 0.55;      // anodized: coloured metal bodies
const KINDS = ['embed', 'pos', 'attn', 'q', 'k', 'v', 'o', 'mlp', 'norm', 'resid', 'head', 'mask'];

export function createLook(engine) {
  const R = engine.renderer, scene = engine.scene;
  const look = { dark: false, c: {}, env: null };

  // ------------------------------------------------------------ backdrop (screen space, still)
  const tri = new THREE.BufferGeometry();
  tri.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const bgMat = new THREE.ShaderMaterial({
    uniforms: { cIn: { value: new THREE.Color() }, cOut: { value: new THREE.Color() }, aspect: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 1.0, 1.0); }',
    fragmentShader: `uniform vec3 cIn; uniform vec3 cOut; uniform float aspect; varying vec2 vUv;
      void main() {
        vec2 p = (vUv - vec2(0.5, 0.52)) * vec2(aspect, 1.0);
        float t = smoothstep(0.0, 1.0, clamp(length(p) / 1.05, 0.0, 1.0));
        gl_FragColor = vec4(mix(cIn, cOut, t), 1.0);
        #include <colorspace_fragment>
      }`,
    depthTest: false, depthWrite: false, toneMapped: false,
  });
  const backdrop = new THREE.Mesh(tri, bgMat);
  backdrop.frustumCulled = false;
  backdrop.renderOrder = -1000;
  scene.add(backdrop);

  // ------------------------------------------------------------ lights (no shadow maps)
  const key = new THREE.DirectionalLight(0xffffff, 1);
  key.position.set(-4, 18, 5);
  const rim = new THREE.DirectionalLight(0xffffff, 0.5);
  rim.position.set(7, 6, -12);
  scene.add(key, rim);
  look.key = key;

  // ------------------------------------------------------------ studio environment (reflections)
  const pmrem = new THREE.PMREMGenerator(R);
  let envTarget = null;
  function studio(dark) {
    const s = new THREE.Scene();
    const dome = new THREE.SphereGeometry(40, 48, 24);
    const cols = [], pos = dome.attributes.position;
    const top = new THREE.Color(dark ? 0x1a2030 : 0xf4f6fa).multiplyScalar(dark ? 0.6 : 0.75);
    const mid = new THREE.Color(dark ? 0x121722 : 0xd5dbe4).multiplyScalar(dark ? 0.45 : 0.42);
    const low = new THREE.Color(dark ? 0x07090d : 0x7d8696).multiplyScalar(dark ? 0.3 : 0.22);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i) / 40;
      if (y > 0) c.copy(mid).lerp(top, Math.pow(y, 0.7)); else c.copy(mid).lerp(low, Math.pow(-y, 0.6));
      cols.push(c.r, c.g, c.b);
    }
    dome.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    s.add(new THREE.Mesh(dome, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
    const panel = (w, h, k, p) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(1, 1, 1).multiplyScalar(k), side: THREE.DoubleSide }));
      m.position.set(...p);
      m.lookAt(0, 0, 0);
      s.add(m);
    };
    panel(22, 22, dark ? 3.2 : 3.4, [0, 24, 2]);      // overhead softbox
    panel(14, 9, dark ? 3.6 : 4.0, [-17, 9, 15]);     // key, front left
    panel(9, 9, dark ? 1.2 : 1.6, [18, 4, 13]);       // fill, front right
    panel(26, 2.6, dark ? 4.5 : 4.0, [2, 7, -20]);    // rim strip, behind
    const target = pmrem.fromScene(s, 0.035);
    s.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    return target;
  }

  const offResize = engine.onResize((w, h) => {
    bgMat.uniforms.aspect.value = w / Math.max(1, h);
  });
  bgMat.uniforms.aspect.value = engine.container.clientWidth / Math.max(1, engine.container.clientHeight);

  // ------------------------------------------------------------ palette
  look.refresh = () => {
    const dark = look.dark = engine.isDark;
    for (const k of KINDS) look.c[k] = themeColor(`--c-${k}`);
    for (const k of ['bg', 'fg', 'fg-2', 'fg-3', 'accent', 'surface', 'surface-2', 'surface-3', 'line', 'heat-lo', 'heat-hi', 'heat-neg', 'heat-pos', 'heat-mid']) look.c[k] = themeColor(`--${k}`);
    // backdrop: a breath of light behind the model, fading to the page colour at the edges
    const bg = look.c.bg;
    bgMat.uniforms.cOut.value.copy(bg);
    bgMat.uniforms.cIn.value.copy(dark ? bg.clone().lerp(new THREE.Color('#1c2230'), 0.55) : bg.clone().lerp(new THREE.Color(1, 1, 1), 0.8));
    scene.background = null;
    scene.fog = null;
    R.setClearColor(bg, 1);
    R.toneMapping = THREE.NoToneMapping;
    R.toneMappingExposure = 1;
    engine.hemi.color.set(0xffffff);
    engine.hemi.groundColor.copy(dark ? new THREE.Color(0x0b0d12) : new THREE.Color(0xc9ced8));
    engine.hemi.intensity = dark ? 0.2 : 0.25;
    engine.sun.intensity = 0;
    key.intensity = dark ? 1.0 : 1.55;
    rim.intensity = dark ? 1.1 : 0.3;
    rim.color.copy(dark ? look.c.accent.clone().lerp(new THREE.Color(1, 1, 1), 0.5) : new THREE.Color(1, 1, 1));
    envTarget?.dispose();
    envTarget = studio(dark);
    look.env = envTarget.texture;
    scene.environment = look.env;
    engine.setRenderFn(null);
    engine.invalidate();
  };

  /** The body colour of a component: the token itself in light; deeper, richer in dark. */
  look.body = (kind) => {
    const c = look.c[kind].clone();
    const hsl = {};
    c.getHSL(hsl);
    return look.dark ? c.setHSL(hsl.h, Math.min(1, hsl.s * 0.82), hsl.l * 0.46) : c.setHSL(hsl.h, Math.min(1, hsl.s * 1.0), hsl.l * 0.86);
  };
  /** The colour data glows with (dark) or is inked in (light). */
  look.ink = (kind) => {
    const c = look.c[kind].clone();
    if (look.dark) return c;
    const hsl = {};
    c.getHSL(hsl);
    return c.setHSL(hsl.h, hsl.s, hsl.l * 0.86);
  };
  /** A pale (light) or deep (dark) tint of a component's colour: the tray its data stands on. */
  look.tray = (kind) => (look.dark ? new THREE.Color(0x06080c).lerp(look.c[kind], 0.07) : new THREE.Color(1, 1, 1).lerp(look.c[kind], 0.1));
  /** Neutral "screen" surface that carries data. */
  look.screen = () => (look.dark ? new THREE.Color(0x07090e) : new THREE.Color(0xffffff));

  look.bodyMat = (kind, opts = {}) => new THREE.MeshPhysicalMaterial({
    color: opts.color ?? look.body(kind),
    roughness: opts.roughness ?? (look.dark ? 0.34 : 0.36),
    metalness: opts.metalness ?? METAL,
    clearcoat: opts.clearcoat ?? 0.35,
    clearcoatRoughness: 0.12,
    envMapIntensity: opts.env ?? (look.dark ? 0.9 : 1.0),
    emissive: (opts.glow ?? look.c[kind]).clone(),
    emissiveIntensity: opts.emissive ?? 0,
    sheen: look.dark ? 0.25 : 0,
    sheenRoughness: 0.45,
    sheenColor: look.c[kind].clone().multiplyScalar(0.6),
    transparent: (opts.opacity ?? 1) < 1,
    opacity: opts.opacity ?? 1,
    depthWrite: (opts.opacity ?? 1) >= 1,
  });
  look.screenMat = (opts = {}) => new THREE.MeshPhysicalMaterial({
    color: opts.color ?? look.screen(),
    roughness: look.dark ? 0.62 : 0.55,
    metalness: 0,
    clearcoat: look.dark ? 0 : 0.2,
    clearcoatRoughness: 0.3,
    envMapIntensity: look.dark ? 0.25 : 0.5,
    emissive: new THREE.Color(0, 0, 0),
    emissiveIntensity: 0,
  });

  look.dispose = () => {
    offResize();
    engine.setRenderFn(null);
    envTarget?.dispose();
    pmrem.dispose();
    scene.remove(backdrop, key, rim);
    tri.dispose();
    bgMat.dispose();
  };
  return look;
}
