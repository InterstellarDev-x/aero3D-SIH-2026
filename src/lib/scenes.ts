// Procedural "satellite" scenes: perfectly registered RGB texture + heightfield pairs.
// These stand in for real sensor captures in the prototype.

import { mulberry32, fbm, ridged, gauss } from './rng';

export type LandscapeKind = 'urban' | 'hilly' | 'forest' | 'olympic';

// Land-cover classes (also used by the simulated classifier)
export const LCLASS = { GROUND: 0, BUILDING: 1, ROAD: 2, VEGETATION: 3, WATER: 4 } as const;
export const LCLASS_NAMES = ['Bare ground', 'Building', 'Road', 'Vegetation', 'Water'];
export const LCLASS_COLORS: [number, number, number][] = [
  [168, 142, 100], // ground — tan
  [231, 76, 60],   // building — red
  [241, 196, 15],  // road — yellow
  [46, 204, 113],  // vegetation — green
  [52, 152, 219],  // water — blue
];

export interface SceneData {
  id: string;
  name: string;
  kind: LandscapeKind;
  size: number;            // grid resolution (N x N)
  worldSize: number;       // meters across
  heights: Float32Array;   // meters above local datum, N*N
  labels: Uint8Array;      // land-cover class per pixel, N*N (ground truth)
  rgb: HTMLCanvasElement;  // top-down "optical" texture
  baseElevation: number;   // fake MSL base, meters
  relief: number;          // fake max relief, meters
  blurb: string;
}

const N = 256;

function makeCanvas(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; img: ImageData } {
  const canvas = document.createElement('canvas');
  canvas.width = N; canvas.height = N;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(N, N);
  return { canvas, ctx, img };
}

function setPx(img: ImageData, x: number, y: number, r: number, g: number, b: number) {
  const i = (y * N + x) * 4;
  img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
}

function shade(base: [number, number, number], f: number): [number, number, number] {
  return [base[0] * f, base[1] * f, base[2] * f].map(v => Math.max(0, Math.min(255, v))) as [number, number, number];
}

// ---------------------------------------------------------------- urban ---
function genUrban(seed: number): SceneData {
  const rand = mulberry32(seed);
  const heights = new Float32Array(N * N);
  const ground = new Float32Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    ground[y * N + x] = 1.5 + fbm(x / 90, y / 90, seed) * 3;
  }
  heights.set(ground);

  const isRoad = new Uint8Array(N * N);
  const isBldg = new Uint8Array(N * N);
  const roofTint = new Float32Array(N * N);

  const cell = 46, road = 7;
  // pick one park block
  const parkBX = 1 + Math.floor(rand() * 3), parkBY = 1 + Math.floor(rand() * 3);

  for (let by = 0; by * cell < N; by++) for (let bx = 0; bx * cell < N; bx++) {
    const x0 = bx * cell, y0 = by * cell;
    for (let y = y0; y < Math.min(y0 + cell, N); y++) for (let x = x0; x < Math.min(x0 + cell, N); x++) {
      const lx = x - x0, ly = y - y0;
      if (lx < road || ly < road) isRoad[y * N + x] = 1;
    }
    const ix0 = x0 + road, iy0 = y0 + road, ix1 = Math.min(x0 + cell, N), iy1 = Math.min(y0 + cell, N);
    if (ix1 - ix0 < 12 || iy1 - iy0 < 12) continue;
    const isPark = bx === parkBX && by === parkBY;
    if (isPark) continue; // stays ground -> painted green later
    // 2x2 lots
    const mx = (ix0 + ix1) / 2, my = (iy0 + iy1) / 2;
    const lots = [[ix0, iy0, mx, my], [mx, iy0, ix1, my], [ix0, my, mx, iy1], [mx, my, ix1, iy1]];
    for (const [ax, ay, bx2, by2] of lots) {
      if (bx2 - ax < 8 || by2 - ay < 8) continue;
      if (rand() < 0.12) continue; // empty lot
      const pad = 2 + rand() * 2;
      const rx0 = Math.floor(ax + pad), ry0 = Math.floor(ay + pad);
      const rx1 = Math.floor(bx2 - pad), ry1 = Math.floor(by2 - pad);
      // downtown factor: taller near center
      const dc = Math.hypot((ax + bx2) / 2 - N / 2, (ay + by2) / 2 - N / 2) / (N / 2);
      const hMax = 14 + (1 - dc) * 55 + rand() * 12;
      const h = 9 + Math.pow(rand(), 1.6) * hMax;
      const tint = rand();
      for (let y = ry0; y < ry1; y++) for (let x = rx0; x < rx1; x++) {
        heights[y * N + x] = h;
        isBldg[y * N + x] = 1;
        roofTint[y * N + x] = tint;
      }
      // rooftop clutter box
      if (rand() < 0.6) {
        const cw = 2 + Math.floor(rand() * 3);
        const cx = rx0 + 2 + Math.floor(rand() * Math.max(1, rx1 - rx0 - cw - 4));
        const cy = ry0 + 2 + Math.floor(rand() * Math.max(1, ry1 - ry0 - cw - 4));
        for (let y = cy; y < Math.min(cy + cw, ry1); y++) for (let x = cx; x < Math.min(cx + cw, rx1); x++) {
          heights[y * N + x] = h + 2.2;
          roofTint[y * N + x] = tint;
        }
      }
    }
  }

  const { canvas, ctx, img } = makeCanvas();
  const roofPalettes: [number, number, number][] = [
    [186, 178, 162], [168, 164, 154], [198, 190, 176], [150, 158, 168], [176, 166, 150],
  ];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const n = fbm(x / 24, y / 24, seed + 7);
    let c: [number, number, number];
    if (isRoad[i]) {
      c = shade([52, 55, 62], 0.85 + n * 0.3);
      // lane dashes
      if (((x + y) % 24) < 8 && !isRoad[Math.min(N * N - 1, i + N)]) c = shade([120, 118, 110], 0.9);
    } else if (isBldg[i]) {
      const p = roofPalettes[Math.floor(roofTint[i] * roofPalettes.length) % roofPalettes.length];
      c = shade(p, 0.86 + n * 0.28);
    } else {
      // ground: concrete / park
      const bx = Math.floor(x / cell), by = Math.floor(y / cell);
      const park = bx === parkBX && by === parkBY;
      if (park) c = shade([74, 124, 62], 0.8 + n * 0.4);
      else c = shade([148, 146, 138], 0.82 + n * 0.32);
    }
    setPx(img, x, y, c[0], c[1], c[2]);
  }
  // parapet edges: darken building borders
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const i = y * N + x;
    if (isBldg[i] && (!isBldg[i - 1] || !isBldg[i + 1] || !isBldg[i - N] || !isBldg[i + N])) {
      const o = i * 4;
      img.data[o] *= 0.62; img.data[o + 1] *= 0.62; img.data[o + 2] *= 0.62;
    }
  }
  ctx.putImageData(img, 0, 0);

  // land-cover labels: road=2, building=1, park=3, else ground=0
  const labels = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const bx = Math.floor(x / cell), by = Math.floor(y / cell);
    if (isRoad[i]) labels[i] = LCLASS.ROAD;
    else if (isBldg[i]) labels[i] = LCLASS.BUILDING;
    else if (bx === parkBX && by === parkBY) labels[i] = LCLASS.VEGETATION;
    else labels[i] = LCLASS.GROUND;
  }

  return {
    id: 'urban', name: 'Metro Core — Urban', kind: 'urban', size: N, worldSize: 1200,
    heights, labels, rgb: canvas, baseElevation: 542, relief: 95,
    blurb: 'Dense downtown blocks, 9–80 m structures, road grid and a central park.',
  };
}

// ----------------------------------------------------------------- hilly ---
function genHilly(seed: number): SceneData {
  const rand = mulberry32(seed);
  const heights = new Float32Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const r = ridged(x / 46, y / 46, seed, 5);
    const d = fbm(x / 130, y / 130, seed + 21);
    heights[y * N + x] = r * 150 + d * 22;
  }
  // a winding river valley carved along a curve
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const riverX = N * 0.5 + Math.sin(y / N * Math.PI * 3.2) * N * 0.16;
    const dd = Math.abs(x - riverX);
    if (dd < 7) {
      const k = Math.max(0, 1 - dd / 7);
      heights[y * N + x] = heights[y * N + x] * (1 - k * 0.55);
    }
  }

  const { canvas, ctx, img } = makeCanvas();
  const maxH = 172;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const h = heights[i];
    const t = Math.min(1, h / maxH);
    const gx = (heights[i + 1 < N * N ? i + 1 : i] - heights[i - 1 >= 0 ? i - 1 : i]);
    const gy = (heights[Math.min(N * N - 1, i + N)] - heights[Math.max(0, i - N)]);
    const slope = Math.min(1, Math.hypot(gx, gy) / 14);
    const n = fbm(x / 30, y / 30, seed + 5);
    let c: [number, number, number];
    if (t < 0.32) c = [46 + n * 30, 96 + n * 40, 44 + n * 22];          // valley green
    else if (t < 0.58) c = [104 + n * 34, 118 + n * 30, 62 + n * 20];   // scrub
    else if (t < 0.8) c = [138 + n * 26, 112 + n * 24, 82 + n * 18];    // brown
    else c = [150 + n * 30, 148 + n * 30, 146 + n * 30];                // rock
    // river
    const riverX = N * 0.5 + Math.sin(y / N * Math.PI * 3.2) * N * 0.16;
    if (Math.abs(x - riverX) < 2.5) c = [74, 128, 158];
    const light = (1 - slope * 0.55) * (0.82 + n * 0.3);
    c = shade(c, light);
    setPx(img, x, y, c[0], c[1], c[2]);
  }
  ctx.putImageData(img, 0, 0);

  // labels: river=4, vegetated lowlands=3, rock/high=0
  const labels = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const riverX = N * 0.5 + Math.sin(y / N * Math.PI * 3.2) * N * 0.16;
    if (Math.abs(x - riverX) < 2.5) labels[i] = LCLASS.WATER;
    else labels[i] = heights[i] / maxH < 0.55 ? LCLASS.VEGETATION : LCLASS.GROUND;
  }

  void rand;
  return {
    id: 'hilly', name: 'Ridgeline — Hilly', kind: 'hilly', size: N, worldSize: 2400,
    heights, labels, rgb: canvas, baseElevation: 1180, relief: 175,
    blurb: 'Ridged terrain to ~170 m relief with a carved river valley.',
  };
}

// ---------------------------------------------------------------- forest ---
function genForest(seed: number): SceneData {
  const rand = mulberry32(seed);
  const heights = new Float32Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    heights[y * N + x] = 10 + fbm(x / 70, y / 70, seed) * 16;
  }
  const treeTint = new Float32Array(N * N).fill(-1);
  const trees: { x: number; y: number; r: number; h: number; t: number }[] = [];
  for (let k = 0; k < 1100; k++) {
    const x = rand() * N, y = rand() * N;
    // keep a couple of clearings
    if (Math.hypot(x - N * 0.3, y - N * 0.68) < 26) continue;
    if (Math.hypot(x - N * 0.74, y - N * 0.3) < 20) continue;
    trees.push({ x, y, r: 2 + rand() * 2.6, h: 6 + rand() * 8, t: rand() });
  }
  for (const tr of trees) {
    const r0 = Math.ceil(tr.r * 2);
    for (let dy = -r0; dy <= r0; dy++) for (let dx = -r0; dx <= r0; dx++) {
      const x = Math.round(tr.x + dx), y = Math.round(tr.y + dy);
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      const d = Math.hypot(dx, dy) / tr.r;
      if (d > 2) continue;
      const bump = tr.h * Math.exp(-d * d * 1.4);
      const i = y * N + x;
      if (heights[i] < 10 + bump + 6) { heights[i] = 10 + bump + 6; treeTint[i] = tr.t; }
    }
  }

  const { canvas, ctx, img } = makeCanvas();
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const n = fbm(x / 26, y / 26, seed + 11);
    let c: [number, number, number];
    if (treeTint[i] >= 0) {
      const t = treeTint[i];
      c = [34 + t * 26 + n * 22, 82 + t * 30 + n * 30, 30 + t * 16 + n * 16];
    } else {
      const clearing = Math.hypot(x - N * 0.3, y - N * 0.68) < 26 || Math.hypot(x - N * 0.74, y - N * 0.3) < 20;
      c = clearing ? shade([150, 128, 88], 0.85 + n * 0.3) : shade([88, 128, 62], 0.8 + n * 0.4);
    }
    setPx(img, x, y, c[0], c[1], c[2]);
  }
  ctx.putImageData(img, 0, 0);

  // labels: canopy/grass=3, clearings=0
  const labels = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const clearing = Math.hypot(x - N * 0.3, y - N * 0.68) < 26 || Math.hypot(x - N * 0.74, y - N * 0.3) < 20;
    labels[i] = clearing ? LCLASS.GROUND : LCLASS.VEGETATION;
  }

  return {
    id: 'forest', name: 'Canopy — Forested', kind: 'forest', size: N, worldSize: 1600,
    heights, labels, rgb: canvas, baseElevation: 315, relief: 42,
    blurb: 'Dense canopy, 6–14 m crowns over gentle undulation, two clearings.',
  };
}

/** Decode a base64 string into a Uint8Array. */
function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// --- Olympic Park: real satellite photo + image-derived heightfield --------
import {
  OLYMPIC_TEX_SIZE,
  OLYMPIC_TEX_RGBA_B64,
  OLYMPIC_HGT_B64,
  OLYMPIC_LBL_B64,
} from './olympicData';

/**
 * Build the Olympic Park scene from the embedded photo data module.
 * The 512x512 RGBA texture is the real satellite photo; the 256x256
 * heightfield and land-cover labels were derived from the photo's
 * colors and shapes (stadium bowl, velodrome dome, pools, rooftops).
 */
function genOlympicPark(): SceneData {
  const texBytes = b64ToBytes(OLYMPIC_TEX_RGBA_B64);
  const hgtBytes = b64ToBytes(OLYMPIC_HGT_B64);
  const lblBytes = b64ToBytes(OLYMPIC_LBL_B64);
  const TS = OLYMPIC_TEX_SIZE;
  const N = 256;
  if (texBytes.length !== TS * TS * 4) {
    throw new Error(`olympicData: texture length ${texBytes.length}, expected ${TS * TS * 4}`);
  }
  if (hgtBytes.length !== N * N * 4) {
    throw new Error(`olympicData: heights length ${hgtBytes.length}, expected ${N * N * 4}`);
  }
  if (lblBytes.length !== N * N) {
    throw new Error(`olympicData: labels length ${lblBytes.length}, expected ${N * N}`);
  }
  const heights = new Float32Array(hgtBytes.buffer, hgtBytes.byteOffset, (N * N));
  const labels = new Uint8Array(lblBytes.buffer, lblBytes.byteOffset, N * N);
  const canvas = document.createElement('canvas');
  canvas.width = TS; canvas.height = TS;
  const ctx = canvas.getContext('2d')!;
  const img = new ImageData(new Uint8ClampedArray(texBytes), TS, TS);
  ctx.putImageData(img, 0, 0);

  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < heights.length; i++) {
    const v = heights[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return {
    id: 'olympic', name: 'Olympic Park — Real Photo', kind: 'olympic', size: N, worldSize: 1100,
    heights, labels, rgb: canvas, baseElevation: 120, relief: mx - mn,
    blurb: 'Real satellite photo: Olympic sports complex — stadium, velodrome, pools, courts and halls with image-derived heights.',
  };
}

export function generateScene(kind: LandscapeKind): SceneData {
  const seeds = { urban: 1207, hilly: 4242, forest: 777 };
  if (kind === 'urban') return genUrban(seeds.urban);
  if (kind === 'hilly') return genHilly(seeds.hilly);
  if (kind === 'olympic') return genOlympicPark();
  return genForest(seeds.forest);
}

/** Slope magnitude (rise/run) per pixel, for overlays and readouts */
export function slopeGrid(heights: Float32Array, n: number, cell: number): Float32Array {
  const s = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    const dx = heights[y * n + Math.min(n - 1, x + 1)] - heights[y * n + Math.max(0, x - 1)];
    const dy = heights[Math.min(n - 1, y + 1) * n + x] - heights[Math.max(0, y - 1) * n + x];
    s[i] = Math.hypot(dx, dy) / (2 * cell);
  }
  return s;
}

export function gridStats(h: Float32Array): { min: number; max: number; mean: number } {
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < h.length; i++) { const v = h[i]; if (v < min) min = v; if (v > max) max = v; sum += v; }
  return { min, max, mean: sum / h.length };
}

// --- DSM world-space accessors (unexaggerated) --------------------------------
// Survey tools (spot elevations, smart grids) sample the TRUE DSM — the
// unexaggerated Float32Array in meters — using the same world convention as
// the render mesh (mesh.ts sampleHeightWorld): world x,z in meters centered
// on the scene, mapped linearly to grid coords 0..N-1. mesh.ts is off-limits
// to geo specialists, so this module carries the mirror implementation.

/** Grid/world conversion constants for a scene's DSM. */
export interface SceneGeo {
  size: number;      // grid resolution N (N×N heights)
  worldSize: number; // meters across
  gsd: number;       // meters per grid cell = worldSize / (N - 1)
  half: number;      // world half-extent in meters = worldSize / 2
}

/** Geometry constants for a scene's DSM grid (accepts SceneData or any {size, worldSize}). */
export function sceneGeo(s: Pick<SceneData, 'size' | 'worldSize'>): SceneGeo {
  return {
    size: s.size, worldSize: s.worldSize,
    gsd: s.worldSize / (s.size - 1), half: s.worldSize / 2,
  };
}

/** World (x,z in meters, centered on the scene) -> grid coords (fx,fz in [0, size-1]). */
export function worldToGrid(g: SceneGeo, x: number, z: number): { fx: number; fz: number } {
  return {
    fx: ((x + g.half) / g.worldSize) * (g.size - 1),
    fz: ((z + g.half) / g.worldSize) * (g.size - 1),
  };
}

/** Grid coords (fx,fz in [0, size-1]) -> world (x,z in meters, centered on the scene). */
export function gridToWorld(g: SceneGeo, fx: number, fz: number): { x: number; z: number } {
  return {
    x: (fx / (g.size - 1)) * g.worldSize - g.half,
    z: (fz / (g.size - 1)) * g.worldSize - g.half,
  };
}

/** Clamp a world x/z into the scene's valid extent (with a small inset in cells). */
export function clampWorld(g: SceneGeo, x: number, z: number, insetCells = 1): { x: number; z: number } {
  const lim = Math.max(0, g.half - insetCells * g.gsd);
  return {
    x: Math.max(-lim, Math.min(lim, x)),
    z: Math.max(-lim, Math.min(lim, z)),
  };
}

/**
 * Bilinear sample of an unexaggerated DSM at world coords (meters, centered).
 * This is the "truth" value survey tools report — NOT the exaggerated render
 * height (see viewer.heightAt, which multiplies by the vertical exaggeration).
 */
export function sampleDSM(heights: Float32Array, g: SceneGeo, x: number, z: number): number {
  const n = g.size;
  if (heights.length !== n * n || n < 2) return 0;
  const { fx, fz } = worldToGrid(g, x, z);
  const x0 = Math.max(0, Math.min(n - 2, Math.floor(fx)));
  const z0 = Math.max(0, Math.min(n - 2, Math.floor(fz)));
  const tx = Math.max(0, Math.min(1, fx - x0));
  const tz = Math.max(0, Math.min(1, fz - z0));
  const a = heights[z0 * n + x0], b = heights[z0 * n + x0 + 1];
  const c = heights[(z0 + 1) * n + x0], d = heights[(z0 + 1) * n + x0 + 1];
  return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
}
