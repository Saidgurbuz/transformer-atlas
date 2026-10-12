// Geometry for the root view (variant b): soft "tablets", rounded rectangles in plan with a fully
// rounded edge profile and smooth normals, so every part reads as one designed object family.
import { THREE } from '../../../scene/engine3d.js';

const cache = new Map();

/**
 * A tablet centred on the origin: w (x) × h (y) × d (z), plan corner radius rp, edge radius bev.
 * Geometries are cached by size and shared; callers must not dispose them (see disposeTablets).
 */
export function tablet(w, h, d, rp = 0.3, bev = 0.06) {
  bev = Math.min(bev, h / 2 - 1e-3, w / 4, d / 4);
  const key = [w, h, d, rp, bev].map((v) => v.toFixed(4)).join(',');
  if (cache.has(key)) return cache.get(key);
  const iw = w - 2 * bev, id = d - 2 * bev, r = Math.max(0.001, Math.min(rp - bev, iw / 2 - 1e-3, id / 2 - 1e-3));
  const s = new THREE.Shape();
  const x0 = -iw / 2, x1 = iw / 2, y0 = -id / 2, y1 = id / 2;
  s.moveTo(x0 + r, y0);
  s.lineTo(x1 - r, y0);
  s.absarc(x1 - r, y0 + r, r, -Math.PI / 2, 0, false);
  s.lineTo(x1, y1 - r);
  s.absarc(x1 - r, y1 - r, r, 0, Math.PI / 2, false);
  s.lineTo(x0 + r, y1);
  s.absarc(x0 + r, y1 - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(x0, y0 + r);
  s.absarc(x0 + r, y0 + r, r, Math.PI, Math.PI * 1.5, false);
  let g = new THREE.ExtrudeGeometry(s, { depth: Math.max(1e-3, h - 2 * bev), bevelEnabled: true, bevelThickness: bev, bevelSize: bev, bevelSegments: 6, curveSegments: 10 });
  g.rotateX(-Math.PI / 2);                  // plan in x/z, thickness along +y
  g.translate(0, -(h / 2 - bev), 0);
  g = smooth(g);
  g.userData.shared = true;
  cache.set(key, g);
  return g;
}

/** Weld coincident vertices and recompute normals: rounded edges shade as one smooth surface. */
function smooth(g) {
  const pos = g.attributes.position, n = pos.count, map = new Map(), index = [], verts = [];
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const k = `${Math.round(x * 1e4)},${Math.round(y * 1e4)},${Math.round(z * 1e4)}`;
    let j = map.get(k);
    if (j === undefined) { j = verts.length / 3; map.set(k, j); verts.push(x, y, z); }
    index.push(j);
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  out.setIndex(index);
  out.computeVertexNormals();
  out.computeBoundingBox();
  out.computeBoundingSphere();
  g.dispose();
  return out;
}

export function disposeTablets() {
  for (const g of cache.values()) g.dispose();
  cache.clear();
}
