// SIMULATED elevation pipeline. No real model runs here — this is the demo stand-in
// for the monocular depth backbone + calibration stage of the SIH problem.

import { mulberry32, blur, gauss, ridged } from './rng';
import { gridStats, sceneGeo, sampleDSM, clampWorld, type SceneData, type SceneGeo } from './scenes';

export interface DepthResult {
  relative: Float32Array; // 0..1 "predicted" relative depth
  size: number;
  model: string;
  inferenceMs: number;
}

/**
 * Fake monocular depth prediction.
 * - For built-in scenes we know the true heights: prediction = truth + sensor noise,
 *   which is exactly what a good model output looks like next to its reference.
 * - For user uploads there is no truth: derive a plausible relief map from
 *   image luminance (bright rooftops read taller), blurred + noise.
 */
export function estimateDepth(scene: SceneData | null, img: ImageData, size: number): DepthResult {
  const t0 = performance.now();
  const n = size;
  const rel = new Float32Array(n * n);
  const rand = mulberry32(9001);

  if (scene) {
    const { min, max } = gridStats(scene.heights);
    const span = Math.max(1e-6, max - min);
    for (let i = 0; i < rel.length; i++) {
      rel[i] = (scene.heights[i] - min) / span + gauss(rand) * 0.016;
    }
  } else {
    // luminance heuristic: brighter -> higher (rooftops), then blur
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const o = (y * n + x) * 4;
      const lum = (img.data[o] * 0.299 + img.data[o + 1] * 0.587 + img.data[o + 2] * 0.114) / 255;
      rel[y * n + x] = lum;
    }
    blur(rel, n); blur(rel, n);
    const { min, max } = gridStats(rel);
    const span = Math.max(1e-6, max - min);
    for (let i = 0; i < rel.length; i++) rel[i] = (rel[i] - min) / span + gauss(rand) * 0.02;
  }
  // clamp + smooth
  for (let i = 0; i < rel.length; i++) rel[i] = Math.min(1, Math.max(0, rel[i]));
  blur(rel, n);

  return {
    relative: rel, size: n,
    model: 'mono-depth v2.1 (simulated)',
    inferenceMs: Math.round(performance.now() - t0 + 1800 + rand() * 900),
  };
}

export interface Calibration {
  absolute: Float32Array; // meters above MSL (fake)
  size: number;
  georeferenced: boolean;
  method: string;
  baseElevation: number;
  relief: number;
  minH: number; maxH: number;
}

/**
 * Fake scale calibration: relative depth -> metric elevation.
 * Georeferenced path pretends to anchor on SRTM-30m stats; the
 * non-georeferenced path keeps a relative datum (rDSM).
 */
export function calibrate(rel: Float32Array, size: number, opts: {
  georeferenced: boolean; baseElevation: number; relief: number;
}): Calibration {
  const abs = new Float32Array(rel.length);
  const { georeferenced, baseElevation, relief } = opts;
  for (let i = 0; i < rel.length; i++) abs[i] = baseElevation + rel[i] * relief;
  const { min, max } = gridStats(abs);
  return {
    absolute: abs, size,
    georeferenced,
    method: georeferenced
      ? 'SRTM-30m anchored affine fit (simulated)'
      : 'rDSM — relative datum, no absolute anchor',
    baseElevation, relief, minH: min, maxH: max,
  };
}

export interface Metrics { rmse: number; mae: number; corr: number; bias: number }

/** Real error stats of prediction vs. known truth (built-in scenes only). */
export function computeMetrics(pred: Float32Array, truth: Float32Array): Metrics {
  const n = pred.length;
  let se = 0, ae = 0, sp = 0, st = 0, spt = 0, spt2 = 0, stt2 = 0;
  for (let i = 0; i < n; i++) {
    const e = pred[i] - truth[i];
    se += e * e; ae += Math.abs(e);
    sp += pred[i]; st += truth[i];
  }
  const mp = sp / n, mt = st / n;
  for (let i = 0; i < n; i++) {
    const dp = pred[i] - mp, dt = truth[i] - mt;
    spt += dp * dt; spt2 += dp * dp; stt2 += dt * dt;
  }
  return {
    rmse: Math.sqrt(se / n),
    mae: ae / n,
    corr: spt / Math.sqrt(Math.max(1e-9, spt2 * stt2)),
    bias: mp - mt,
  };
}

// --- turbo-ish colormap for depth/error visualization -----------------------
const TURBO: [number, number, number][] = [
  [48, 18, 59], [70, 30, 120], [60, 80, 180], [40, 140, 200], [30, 180, 170],
  [60, 200, 120], [140, 210, 70], [220, 220, 60], [250, 180, 40], [240, 110, 30], [210, 60, 40],
];

export function turbo(t: number): [number, number, number] {
  const x = Math.min(0.9999, Math.max(0, t)) * (TURBO.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = TURBO[i], b = TURBO[Math.min(TURBO.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Render a float grid to a canvas with the turbo colormap. */
export function renderColormap(grid: Float32Array, size: number, min: number, max: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const span = Math.max(1e-6, max - min);
  for (let i = 0; i < grid.length; i++) {
    const [r, g, b] = turbo((grid[i] - min) / span);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// --- cut/fill volume measurement (Pix4D-style survey metric) -----------------

export interface WorldXZ { x: number; z: number }

export interface VolumeResult {
  cutM3: number;   // m³ — volume above the base plane (material to remove)
  fillM3: number;  // m³ — volume below the base plane (material to add)
  netM3: number;   // m³ — cut - fill (positive = net removal)
  baseM: number;   // m — base plane height (mean sampled along the boundary)
  areaM2: number;  // m² — rasterized polygon area
  cells: number;   // number of rasterized grid cells
}

/**
 * Cut/fill volume inside a user-drawn polygon.
 *
 * The polygon (world coords, meters, centered on the scene) is rasterized over
 * the height grid. The base plane is the mean height sampled along the polygon
 * boundary; every interior cell is then integrated: cells above the base plane
 * count toward cut, cells below toward fill.
 */
export function calcVolume(
  polygonPts: WorldXZ[],
  heights: Float32Array,
  gsdMeters: number,
): VolumeResult {
  const empty: VolumeResult = { cutM3: 0, fillM3: 0, netM3: 0, baseM: 0, areaM2: 0, cells: 0 };
  if (!polygonPts || polygonPts.length < 3 || !heights || heights.length === 0) return empty;
  const size = Math.round(Math.sqrt(heights.length));
  if (size * size !== heights.length || size < 2 || !(gsdMeters > 0)) return empty;
  if (!polygonPts.every(p => p && Number.isFinite(p.x) && Number.isFinite(p.z))) return empty;

  // grid geometry: n points span -half..+half in world meters
  const half = (size - 1) * gsdMeters / 2;
  const toGrid = (x: number) => (x + half) / gsdMeters; // world -> grid fx
  const { sample } = makeHeightSampler(heights);

  // base plane = mean height sampled densely along the polygon boundary
  let baseSum = 0, baseN = 0;
  for (let i = 0; i < polygonPts.length; i++) {
    const a = polygonPts[i], b = polygonPts[(i + 1) % polygonPts.length];
    const ax = toGrid(a.x), az = toGrid(a.z), bx = toGrid(b.x), bz = toGrid(b.z);
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az) * 2));
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      baseSum += sample(ax + (bx - ax) * t, az + (bz - az) * t);
      baseN++;
    }
  }
  if (baseN === 0) return empty;
  const base = baseSum / baseN;

  // rasterize: cell centers inside the polygon (grid coords)
  const gx = polygonPts.map(p => toGrid(p.x));
  const gz = polygonPts.map(p => toGrid(p.z));
  let minX = size, maxX = 0, minZ = size, maxZ = 0;
  for (let i = 0; i < gx.length; i++) {
    if (gx[i] < minX) minX = gx[i]; if (gx[i] > maxX) maxX = gx[i];
    if (gz[i] < minZ) minZ = gz[i]; if (gz[i] > maxZ) maxZ = gz[i];
  }
  const cx0 = Math.max(0, Math.floor(minX)), cx1 = Math.min(size - 1, Math.ceil(maxX));
  const cz0 = Math.max(0, Math.floor(minZ)), cz1 = Math.min(size - 1, Math.ceil(maxZ));
  const inside = (px: number, pz: number): boolean => {
    let hit = false;
    for (let i = 0, j = gx.length - 1; i < gx.length; j = i++) {
      const xi = gx[i], zi = gz[i], xj = gx[j], zj = gz[j];
      if ((zi > pz) !== (zj > pz) && px < (xj - xi) * (pz - zi) / (zj - zi) + xi) hit = !hit;
    }
    return hit;
  };

  const cellArea = gsdMeters * gsdMeters;
  let cut = 0, fill = 0, cells = 0;
  for (let cy = cz0; cy <= cz1; cy++) {
    for (let cx = cx0; cx <= cx1; cx++) {
      if (!inside(cx + 0.5, cy + 0.5)) continue;
      const d = sample(cx + 0.5, cy + 0.5) - base;
      if (d >= 0) cut += d * cellArea; else fill += -d * cellArea;
      cells++;
    }
  }
  return {
    cutM3: cut, fillM3: fill, netM3: cut - fill,
    baseM: base, areaM2: cells * cellArea, cells,
  };
}

/** Compact display string for the measurement panel. */
export function fmtVolume(r: VolumeResult): string {
  const f = (v: number) => {
    const a = Math.abs(v);
    const s = a >= 10000 ? (a / 1000).toFixed(1) + 'k' : Math.round(a).toLocaleString('en-US');
    return s;
  };
  const sign = r.netM3 < 0 ? '−' : '+';
  return `Cut ${f(r.cutM3)} m³ · Fill ${f(r.fillM3)} m³ · Net ${sign}${f(r.netM3)} m³`;
}

// --- shared bilinear DSM sampler -------------------------------------------

export interface HeightSampler {
  size: number;
  /** Bilinear height sample at grid coords (fx, fz in [0, size-1]), clamped. */
  sample: (fx: number, fz: number) => number;
}

/** Make a clamped bilinear sampler over a square N*N height grid. */
export function makeHeightSampler(heights: Float32Array): HeightSampler {
  const size = Math.round(Math.sqrt(heights.length));
  const sample = (fx: number, fz: number): number => {
    if (size < 2) return heights[0] ?? 0;
    const x0 = Math.max(0, Math.min(size - 2, Math.floor(fx)));
    const z0 = Math.max(0, Math.min(size - 2, Math.floor(fz)));
    const tx = Math.max(0, Math.min(1, fx - x0));
    const tz = Math.max(0, Math.min(1, fz - z0));
    const a = heights[z0 * size + x0], b = heights[z0 * size + x0 + 1];
    const c = heights[(z0 + 1) * size + x0], d = heights[(z0 + 1) * size + x0 + 1];
    return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
  };
  return { size, sample };
}

// --- elevation profile traverse (Virtual Surveyor 'Profile View' pattern) ---

export interface ProfileSample {
  dist: number; // cumulative horizontal distance along the traverse, meters
  h: number;    // unexaggerated elevation, meters
  x: number; z: number; // world coords of the sample (for 3D cursor sync)
}

export interface ProfileResult {
  samples: ProfileSample[];
  lengthM: number;      // total horizontal length, meters
  minH: number; maxH: number; // elevation range, meters
  gainM: number;        // total ascent (sum of positive deltas), meters
  lossM: number;        // total descent (sum of |negative deltas|), meters
  meanSlopePct: number; // mean |grade| between samples, percent
  maxSlopePct: number;  // max |grade| between samples, percent
  segBreaks: number[];  // sample indices where each segment starts
}

/**
 * Sample the (unexaggerated) DSM along a polyline traverse.
 *
 * Points are world coords in meters centered on the scene, same convention as
 * calcVolume; the returned heights come from the bilinear grid sampler, so
 * values reflect the true DSM — not the exaggerated render mesh.
 */
export function computeProfile(
  linePts: WorldXZ[],
  heights: Float32Array,
  gsdMeters: number,
  opts?: { stepM?: number },
): ProfileResult {
  const empty: ProfileResult = {
    samples: [], lengthM: 0, minH: 0, maxH: 0, gainM: 0, lossM: 0,
    meanSlopePct: 0, maxSlopePct: 0, segBreaks: [],
  };
  if (!linePts || linePts.length < 2 || !heights || heights.length === 0) return empty;
  const { size, sample } = makeHeightSampler(heights);
  if (size * size !== heights.length || size < 2 || !(gsdMeters > 0)) return empty;
  if (!linePts.every(p => p && Number.isFinite(p.x) && Number.isFinite(p.z))) return empty;

  const half = (size - 1) * gsdMeters / 2;
  const toGrid = (x: number) => (x + half) / gsdMeters; // world -> grid fx
  const step = Math.max(0.25 * gsdMeters, opts?.stepM ?? gsdMeters);

  const samples: ProfileSample[] = [];
  const segBreaks: number[] = [];
  let dist = 0, gain = 0, loss = 0, slopeSum = 0, slopeN = 0, maxSlope = 0;
  let minH = Infinity, maxH = -Infinity;
  let prevH: number | null = null;

  const push = (x: number, z: number) => {
    const h = sample(toGrid(x), toGrid(z));
    const prev = samples[samples.length - 1];
    const stepDist = prev ? Math.hypot(x - prev.x, z - prev.z) : 0;
    if (prevH !== null) {
      const d = h - prevH;
      if (d > 0) gain += d; else loss -= d;
    }
    dist += stepDist;
    if (prev && stepDist > 1e-9) {
      const s = Math.abs(h - prev.h) / stepDist * 100;
      slopeSum += s; slopeN++;
      if (s > maxSlope) maxSlope = s;
    }
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
    samples.push({ dist, h, x, z });
    prevH = h;
  };

  for (let i = 0; i < linePts.length - 1; i++) {
    const a = linePts[i], b = linePts[i + 1];
    segBreaks.push(samples.length);
    const segLen = Math.hypot(b.x - a.x, b.z - a.z);
    const steps = Math.max(1, Math.ceil(segLen / step));
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      push(a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t);
    }
  }
  const last = linePts[linePts.length - 1];
  push(last.x, last.z);
  if (samples.length === 0) return empty;

  return {
    samples, lengthM: dist, minH, maxH,
    gainM: gain, lossM: loss,
    meanSlopePct: slopeN ? slopeSum / slopeN : 0,
    maxSlopePct: maxSlope,
    segBreaks,
  };
}

/** Compact display string for the profile panel header. */
export function fmtProfile(r: ProfileResult): string {
  if (r.samples.length === 0) return 'No profile';
  const L = r.lengthM >= 1000 ? (r.lengthM / 1000).toFixed(2) + ' km' : Math.round(r.lengthM) + ' m';
  return `${L} · ${r.minH.toFixed(1)}–${r.maxH.toFixed(1)} m · ↑${Math.round(r.gainM)} m ↓${Math.round(r.lossM)} m · avg ${r.meanSlopePct.toFixed(1)}%`;
}

/**
 * Render a lightweight SVG elevation-profile chart for ProfileResult.
 *
 * Includes a cursor group with id "profile-cursor" (a vertical line at x=0 +
 * a dot at cx=0, initially hidden) that the UI scrubs with profileCursorXY:
 *   const { x, y } = profileCursorXY(r, sampleIndex, chartW, chartH);
 *   cursor.setAttribute('transform', `translate(${x},0)`);
 *   dot.setAttribute('cy', y); cursor.removeAttribute('visibility');
 * The layout constants are exported so the UI can position the cursor and
 * sync it with a 3D marker without re-implementing the math.
 */
export const PROFILE_CHART = { padL: 46, padR: 12, padT: 10, padB: 24 } as const;

/** Given a chart width and a sample, return the cursor-line x in SVG units. */
export function profileCursorX(r: ProfileResult, sampleIndex: number, width: number): number {
  return profileCursorXY(r, sampleIndex, width, 200).x;
}

/**
 * Cursor position in SVG units for a sample: x = distance axis, y = elevation
 * axis on the profile line. The UI scrubs the cursor by setting
 *   cursor.setAttribute('transform', `translate(${x},0)`)
 *   dot.setAttribute('cy', y)
 * on the #profile-cursor group (line at x=0, dot at cx=0).
 */
export function profileCursorXY(
  r: ProfileResult, sampleIndex: number, width: number, height = 200,
): { x: number; y: number } {
  const { padL, padR, padT, padB } = PROFILE_CHART;
  const iw = Math.max(1, width - padL - padR), ih = Math.max(1, height - padT - padB);
  const i = Math.max(0, Math.min(r.samples.length - 1, sampleIndex));
  const s = r.samples[i];
  if (!s) return { x: padL, y: padT + ih };
  const pad = Math.max(0.5, (r.maxH - r.minH) * 0.08);
  const lo = r.minH - pad, span = Math.max(1e-6, (r.maxH + pad) - lo);
  const t = r.lengthM > 1e-9 ? s.dist / r.lengthM : 0;
  return {
    x: padL + t * iw,
    y: padT + ih - ((s.h - lo) / span) * ih,
  };
}

export function profileChartSVG(r: ProfileResult, opts?: { width?: number; height?: number }): string {
  const W = opts?.width ?? 520, H = opts?.height ?? 200;
  const { padL, padR, padT, padB } = PROFILE_CHART;
  const iw = Math.max(1, W - padL - padR), ih = Math.max(1, H - padT - padB);
  const f = (v: number) => +v.toFixed(2);
  let body = '';

  if (r.samples.length >= 2) {
    const pad = Math.max(0.5, (r.maxH - r.minH) * 0.08);
    const lo = r.minH - pad, hi = r.maxH + pad, span = Math.max(1e-6, hi - lo);
    const X = (d: number) => padL + (r.lengthM > 1e-9 ? d / r.lengthM : 0) * iw;
    const Y = (h: number) => padT + ih - ((h - lo) / span) * ih;

    // y gridlines + elevation labels
    for (let g = 0; g <= 4; g++) {
      const hv = lo + (span * g) / 4, y = Y(hv);
      body += `<line x1="${f(padL)}" y1="${f(y)}" x2="${f(padL + iw)}" y2="${f(y)}" stroke="#334155" stroke-width="1" opacity="${g === 0 || g === 4 ? 0.7 : 0.35}"/>` +
        `<text x="${f(padL - 6)}" y="${f(y + 3.5)}" text-anchor="end" font-size="9" fill="#94a3b8">${hv.toFixed(1)} m</text>`;
    }
    // x ticks
    for (const t of [0, 0.5, 1]) {
      const d = r.lengthM * t, x = X(d);
      const lbl = d >= 1000 ? (d / 1000).toFixed(1) + ' km' : Math.round(d) + ' m';
      body += `<line x1="${f(x)}" y1="${f(padT + ih)}" x2="${f(x)}" y2="${f(padT + ih + 5)}" stroke="#64748b" stroke-width="1"/>` +
        `<text x="${f(x)}" y="${f(padT + ih + 17)}" text-anchor="middle" font-size="9" fill="#94a3b8">${lbl}</text>`;
    }

    // filled area + profile line
    const pts = r.samples.map(s => `${f(X(s.dist))},${f(Y(s.h))}`).join(' ');
    const dPath = 'M' + r.samples.map(s => `${f(X(s.dist))} ${f(Y(s.h))}`).join(' L');
    body += `<path d="${dPath} L${f(padL + iw)} ${f(padT + ih)} L${f(padL)} ${f(padT + ih)} Z" fill="#22d3ee" opacity="0.22"/>` +
      `<path d="${dPath}" fill="none" stroke="#22d3ee" stroke-width="1.8"/>` +
      `<polygon points="${pts}" fill="none" stroke="none"/>`;
    // segment vertices
    for (const i of r.segBreaks) {
      const s = r.samples[i];
      if (s) body += `<circle cx="${f(X(s.dist))}" cy="${f(Y(s.h))}" r="3" fill="#f472b6" stroke="#0f172a" stroke-width="1"/>`;
    }
    // cursor (UI positions via transform translate; sample linked in data attrs)
    body += `<g id="profile-cursor" visibility="hidden">` +
      `<line x1="0" y1="${f(padT)}" x2="0" y2="${f(padT + ih)}" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="4 3"/>` +
      `<circle cx="0" cy="0" r="4" fill="#fbbf24" stroke="#0f172a" stroke-width="1.5"/></g>`;
  } else {
    body = `<text x="${f(W / 2)}" y="${f(H / 2)}" text-anchor="middle" font-size="12" fill="#94a3b8">Draw a line to see its elevation profile</text>`;
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" ` +
    `style="display:block;max-width:100%" data-profile-len="${f(r.lengthM)}">${body}</svg>`;
}

// --- change detection: pre/post monsoon DSM (simulated) ---------------------

export interface PostEventResult {
  post: Float32Array;    // post-event DSM, meters — same grid, units, datum as input
  size: number;
  meanLoweringM: number; // mean (baseline − post) over affected cells, meters
  maxErosionM: number;   // max (baseline − post), meters
  affectedPct: number;   // % of cells whose lowering exceeds the noise threshold
  erodedCells: number;
  /** Δ-volume cut/fill between baseline and post-event DSMs (VolumeResult semantics). */
  changeVolume: VolumeResult;
}

/**
 * Synthesize a post-event (post-monsoon) variant of the unexaggerated baseline DSM.
 *
 * Two simulated processes, both relief-scaled so behavior stays sane across scenes:
 *  1. Downslope sediment transport — several passes where each interior cell sheds
 *     a slope-scaled fraction of its drop to the lowest 3×3 neighbor, smoothing
 *     ridges and building small deposition lobes in lows.
 *  2. Seeded gully carving — ridged fBm channels (rng.ridged) cut deeper on steep
 *     cells, followed by one blur pass to soften the sediment field.
 * Deterministic (fixed seed 6141), so the demo is reproducible.
 *
 * @param gsdMeters ground sample distance, meters per cell = worldSize/(size−1)
 *   (the same value the measure tool passes to calcVolume). Defaults to 1, i.e.
 *   per-cell areas of 1 m² — pass the scene's real GSD for true m³ volumes.
 *   The result's changeVolume is computed with this GSD; read it via
 *   getChangeVolume() / fmtChangeDetection().
 */
export function synthesizePostEventDSM(heights: Float32Array, size: number, gsdMeters = 1): PostEventResult {
  const empty: PostEventResult = {
    post: new Float32Array(0), size,
    meanLoweringM: 0, maxErosionM: 0, affectedPct: 0, erodedCells: 0,
    changeVolume: { cutM3: 0, fillM3: 0, netM3: 0, baseM: 0, areaM2: 0, cells: 0 },
  };
  const n = size;
  if (!heights || heights.length !== n * n || n < 3) return empty;

  const rand = mulberry32(6141);
  const h = Float32Array.from(heights); // work on a copy — input stays the baseline
  const idx = (x: number, y: number) => y * n + x;

  const relief = Math.max(1e-6, gridStats(h).max - gridStats(h).min);

  // 1) downslope transport along the DSM gradient
  const passes = 6;
  const delta = new Float32Array(n * n);
  for (let p = 0; p < passes; p++) {
    delta.fill(0);
    for (let y = 1; y < n - 1; y++) {
      for (let x = 1; x < n - 1; x++) {
        const i = idx(x, y);
        // lowest of the 8 neighbors
        let low = i, lowV = h[i];
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const j = idx(x + dx, y + dy);
            if (h[j] < lowV) { lowV = h[j]; low = j; }
          }
        }
        const drop = h[i] - lowV;
        if (drop <= 1e-9) continue;
        const slopeT = Math.min(1, drop / (relief * 0.05));
        const move = drop * (0.04 + 0.10 * slopeT);
        delta[i] -= move;
        delta[low] += move;
      }
    }
    for (let i = 0; i < h.length; i++) h[i] += delta[i];
  }

  // 2) seeded gully carving + light smoothing
  for (let y = 1; y < n - 1; y++) {
    for (let x = 1; x < n - 1; x++) {
      const i = idx(x, y);
      const gx = h[idx(x + 1, y)] - h[idx(x - 1, y)];
      const gy = h[idx(x, y + 1)] - h[idx(x, y - 1)];
      const slopeT = Math.min(1, Math.hypot(gx, gy) / (relief * 0.03));
      const ch = ridged(x / 42, y / 42, 90210, 4); // 0..1 channel mask
      h[i] -= ch * ch * slopeT * relief * 0.09 * (0.6 + 0.8 * rand());
    }
  }
  blur(h, n);

  // 3) change stats vs the baseline
  const thresh = Math.max(0.02, relief * 0.0005);
  let sum = 0, max = 0, cnt = 0;
  for (let i = 0; i < h.length; i++) {
    const d = heights[i] - h[i];
    if (d > thresh) { sum += d; cnt++; if (d > max) max = d; }
  }
  return {
    post: h, size: n,
    meanLoweringM: cnt > 0 ? sum / cnt : 0,
    maxErosionM: max,
    affectedPct: (cnt / h.length) * 100,
    erodedCells: cnt,
    changeVolume: computeChangeVolume(heights, h, n, gsdMeters, { minDeltaM: thresh }),
  };
}

/**
 * Δ-volume over time: cut/fill volume between a baseline DSM and its
 * post-event variant, in m³.
 *
 * Reuses the VolumeResult semantics of calcVolume() (Pix4D-style cut/fill) so
 * the UI can share fmtVolume(): here the reference surface is the baseline DSM
 * itself, cell by cell — no planar base fit (baseM is 0 for that reason).
 * Cells where the post-event surface sits BELOW the baseline count toward cut
 * (material removed: erosion), cells where it sits ABOVE count toward fill
 * (material added: deposition). netM3 = cut − fill, positive = net removal —
 * same sign convention as calcVolume().
 *
 * @param baseline pre-event DSM, meters, size×size
 * @param post     post-event DSM, meters — same grid, units, datum as baseline
 * @param gsdMeters ground sample distance, meters per cell (worldSize/(size−1))
 * @param opts.minDeltaM per-cell |Δ| below this is treated as noise and ignored;
 *   pass the same detection threshold as the erosion stats for a consistent
 *   "detectable change" volume. Default 0 (raw integration).
 */
export function computeChangeVolume(
  baseline: Float32Array,
  post: Float32Array,
  size: number,
  gsdMeters: number,
  opts?: { minDeltaM?: number },
): VolumeResult {
  const empty: VolumeResult = { cutM3: 0, fillM3: 0, netM3: 0, baseM: 0, areaM2: 0, cells: 0 };
  if (!baseline || !post || baseline.length !== size * size || post.length !== size * size) return empty;
  if (size < 2 || !(gsdMeters > 0)) return empty;
  const minDelta = Math.max(0, opts?.minDeltaM ?? 0);
  const cellArea = gsdMeters * gsdMeters;
  let cut = 0, fill = 0, cells = 0;
  for (let i = 0; i < baseline.length; i++) {
    const d = post[i] - baseline[i]; // + = material added (fill), − = material removed (cut)
    if (Math.abs(d) <= minDelta) continue;
    if (d >= 0) fill += d * cellArea; else cut += -d * cellArea;
    cells++;
  }
  return {
    cutM3: cut, fillM3: fill, netM3: cut - fill,
    baseM: 0, // no planar base plane: the baseline DSM is the per-cell reference
    areaM2: cells * cellArea, cells,
  };
}

/** Zero VolumeResult for the no-change / no-result case. */
function zeroVolume(): VolumeResult {
  return { cutM3: 0, fillM3: 0, netM3: 0, baseM: 0, areaM2: 0, cells: 0 };
}

/**
 * Getter for the Δ-volume cut/fill computed alongside the Δ-erosion stat at
 * change-detection time. Returns the VolumeResult stored on the
 * PostEventResult (same semantics as calcVolume(); display with fmtVolume()).
 * Safe on null/undefined — yields a zero volume.
 */
export function getChangeVolume(r: PostEventResult | null | undefined): VolumeResult {
  if (!r || !r.changeVolume) return zeroVolume();
  return r.changeVolume;
}

/**
 * Pack the post-event grid into texture-upload order.
 * Mesh uv.y = 1 − row/(n−1) while a THREE.DataTexture's row 0 sits at v=0,
 * so rows are flipped for sampling via vMapUv — same convention as the
 * viewshed mask in viewer.ts.
 */
export function packChangeTexture(r: PostEventResult): Float32Array {
  const n = r.size;
  const out = new Float32Array(n * n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      out[(n - 1 - row) * n + col] = r.post[row * n + col];
    }
  }
  return out;
}

/**
 * Compact display string for the Δ-erosion stat in the metrics panel.
 *
 * Whenever compare (change-detection) mode is active this also carries the
 * Δ-volume cut/fill readout, so the metrics panel refresh path
 * (updateMetrics → refreshMetrics) shows it with no extra wiring.
 */
export function fmtChangeDetection(r: PostEventResult): string {
  if (r.erodedCells === 0) return 'No detectable change';
  const vol = getChangeVolume(r);
  const volLine = vol.cells > 0 ? ` · Δ volume cut/fill: ${fmtVolume(vol)}` : '';
  return `−${r.meanLoweringM.toFixed(2)} m mean · −${r.maxErosionM.toFixed(1)} m max · ${r.affectedPct.toFixed(1)}% of area${volLine}`;
}

// --- spot elevation tool with smart grid (Pix4D-style field verification) ---

/**
 * Spot elevations: click the terrain to drop labeled spot-height markers plus
 * an auto "smart grid" of evenly spaced spot heights. Each marker's h is the
 * UNEXAGGERATED DSM elevation in meters (sampled via scenes.sampleDSM), so the
 * numbers are survey-truth values, not the exaggerated render height.
 *
 * Cross-area wiring (handled outside this module):
 * - The three.js viewer must expose projectToScreen(x, y, z) -> {x, y} | null
 *   and feed it to SpotElevationTool as the projector.
 * - app.ts owns the label-layer container, the marker list panel, fly-to, and
 *   wiring terrain clicks into tool.clickWorld(x, z) and keydown into
 *   tool.handleKey(e). The tool fires onChange whenever the marker set changes.
 */

export type SpotSource = 'manual' | 'grid';

/** One labeled spot-height marker. h is the unexaggerated DSM elevation, meters. */
export interface SpotMarker {
  id: string;          // stable id, e.g. "sp-3"
  label: string;       // display label, e.g. "SP-03"
  x: number; z: number; // world coords in meters (centered on the scene)
  h: number;           // unexaggerated DSM elevation, meters above datum
  source: SpotSource;  // 'manual' = clicked by the user, 'grid' = smart-grid generated
  cls?: number;        // optional land-cover class id at the point (see classify.classAt)
}

export interface SmartGridOptions {
  spacingM?: number;    // meters between grid points; default worldSize / 8
  marginM?: number;     // keep points this far from the scene edge; default 5% of worldSize
  labelPrefix?: string; // default 'G'
  idPrefix?: string;    // default 'sg'
}

/**
 * Build an evenly spaced smart grid of spot-height markers over the scene.
 * Pure function: ids are `${idPrefix}-${k}` and labels `${labelPrefix}-01..`
 * numbering in row-major order (north rows first). Manual markers are NOT
 * touched — the tool's generateSmartGrid() replaces existing 'grid' markers.
 */
export function buildSmartGrid(
  heights: Float32Array, size: number, worldSize: number,
  opts: SmartGridOptions = {},
): SpotMarker[] {
  if (!heights || heights.length !== size * size || size < 2 || !(worldSize > 0)) return [];
  const g: SceneGeo = sceneGeo({ size, worldSize });
  const spacing = Math.max(g.gsd * 4, opts.spacingM ?? worldSize / 8);
  const margin = Math.max(g.gsd, opts.marginM ?? worldSize * 0.05);
  const labelPrefix = opts.labelPrefix ?? 'G';
  const idPrefix = opts.idPrefix ?? 'sg';

  const xs: number[] = [];
  for (let x = -g.half + margin; x <= g.half - margin + 1e-6; x += spacing) xs.push(x);
  if (xs.length === 0) xs.push(0); // tiny scene fallback: one center point

  const markers: SpotMarker[] = [];
  let k = 0;
  for (const z of xs) for (const x of xs) {
    k++;
    markers.push({
      id: `${idPrefix}-${k}`,
      label: `${labelPrefix}-${String(k).padStart(2, '0')}`,
      x, z,
      h: sampleDSM(heights, g, x, z),
      source: 'grid',
    });
  }
  return markers;
}

/** Compact display string for a marker: "SP-03 · 542.6 m". */
export function fmtSpot(m: SpotMarker): string {
  return `${m.label} · ${m.h.toFixed(1)} m`;
}

export interface SpotStats {
  count: number;
  minH: number; maxH: number; meanH: number;
  rangeM: number;
}

/** Summary stats over the current markers (null when there are none). */
export function spotStats(markers: SpotMarker[]): SpotStats | null {
  if (markers.length === 0) return null;
  let min = Infinity, max = -Infinity, sum = 0;
  for (const m of markers) {
    if (m.h < min) min = m.h;
    if (m.h > max) max = m.h;
    sum += m.h;
  }
  return { count: markers.length, minH: min, maxH: max, meanH: sum / markers.length, rangeM: max - min };
}

/** World->screen projector provided by the viewer (null = behind the camera). */
export type SpotProjector = (x: number, y: number, z: number) => { x: number; y: number } | null;

export interface SpotToolHooks {
  /** Label-layer container div (positioned over the canvas, pointer-events none). */
  layer: HTMLElement;
  /** Screen projection from the viewer. */
  projectToScreen: SpotProjector;
  /** Called with a fresh marker array after every mutation (panel list refresh). */
  onChange?: (markers: SpotMarker[]) => void;
}

/** Vertical exaggeration + DSM handed to the tool per scene. */
export interface SpotTerrain {
  heights: Float32Array; // UNEXAGGERATED DSM, meters
  size: number;          // N (N×N grid)
  worldSize: number;     // meters across
  exaggeration?: number; // vertical exaggeration of the render mesh (labels ride on it)
}

/**
 * SpotElevationTool: the geo-area bulk of the spot-elevation feature.
 *
 * The tool owns the marker set, DSM sampling (always unexaggerated), smart
 * grid generation, and the HTML label divs projected every frame via the
 * viewer's projectToScreen. app.ts owns toolbar/panel wiring:
 *   const tool = new SpotElevationTool({ layer, projectToScreen, onChange });
 *   tool.setTerrain({ heights: cal.absolute, size, worldSize, exaggeration });
 *   // click wiring:  tool.setActive(true); ... tool.clickWorld(hit.x, hit.z)
 *   // key wiring:   tool.handleKey(e)   (Esc exits placing mode + clears)
 *   // frame loop:   tool.update()
 * Label anchors sit on the rendered (exaggerated) surface: y = h * exaggeration.
 */
export class SpotElevationTool {
  private layer: HTMLElement;
  private projector: SpotProjector;
  private onChange?: (markers: SpotMarker[]) => void;

  private terrain: (SpotTerrain & { g: SceneGeo }) | null = null;
  private heightsRef: Float32Array | null = null;
  private markers: SpotMarker[] = [];
  private divs = new Map<string, HTMLDivElement>();
  private active = false; // placing mode: terrain clicks drop markers
  private seqManual = 0;
  private seqGrid = 0;
  private layerW = 0;
  private layerH = 0;

  constructor(hooks: SpotToolHooks) {
    this.layer = hooks.layer;
    this.projector = hooks.projectToScreen;
    this.onChange = hooks.onChange;
    this.refreshLayerSize();
    window.addEventListener('resize', this.refreshLayerSize);
  }

  private refreshLayerSize = (): void => {
    this.layerW = this.layer.clientWidth || 800;
    this.layerH = this.layer.clientHeight || 600;
  };

  /** Load (or reload) the DSM for the current scene. Changing the heights array clears markers. */
  setTerrain(t: SpotTerrain): void {
    if (!t.heights || t.heights.length !== t.size * t.size || t.size < 2 || !(t.worldSize > 0)) {
      this.terrain = null;
      return;
    }
    if (this.heightsRef !== t.heights) this.clear(); // new scene -> drop stale markers
    this.heightsRef = t.heights;
    this.terrain = { ...t, g: sceneGeo({ size: t.size, worldSize: t.worldSize }) };
    this.refreshLayerSize();
  }

  /** Update only the vertical exaggeration (render anchor for labels), keeping markers. */
  setExaggeration(ex: number): void {
    if (this.terrain && ex > 0) this.terrain.exaggeration = ex;
  }

  isActive(): boolean { return this.active; }

  /** Enter/exit placing mode (terrain clicks drop markers). */
  setActive(on: boolean): void {
    this.active = on;
  }

  /**
   * Drop a manual spot marker at world coords (x, z in meters, centered).
   * Only works while placing mode is on (setActive(true)). Coordinates are
   * clamped into the scene extent. Returns the new marker or null.
   */
  clickWorld(x: number, z: number): SpotMarker | null {
    if (!this.active || !this.terrain || !Number.isFinite(x) || !Number.isFinite(z)) return null;
    const { g } = this.terrain;
    const c = clampWorld(g, x, z);
    this.seqManual++;
    const m: SpotMarker = {
      id: `sp-${this.seqManual}`,
      label: `SP-${String(this.seqManual).padStart(2, '0')}`,
      x: c.x, z: c.z,
      h: sampleDSM(this.terrain.heights, g, c.x, c.z),
      source: 'manual',
    };
    this.addMarker(m);
    return m;
  }

  /**
   * Generate the smart grid of evenly spaced spot heights. Replaces any
   * existing 'grid' markers (manual markers are kept). Returns the new grid markers.
   */
  generateSmartGrid(opts: SmartGridOptions = {}): SpotMarker[] {
    if (!this.terrain) return [];
    // drop old grid markers first
    for (const m of this.markers) {
      if (m.source === 'grid') {
        this.divs.get(m.id)?.remove();
        this.divs.delete(m.id);
      }
    }
    this.markers = this.markers.filter(m => m.source !== 'grid');
    this.seqGrid++;
    const { heights, size, worldSize } = this.terrain;
    const grid = buildSmartGrid(heights, size, worldSize, {
      ...opts,
      idPrefix: `${opts.idPrefix ?? 'sg'}-r${this.seqGrid}`,
    });
    for (const m of grid) this.addMarker(m, true);
    this.emit();
    return grid;
  }

  /** Remove one marker by id. Returns true if it existed. */
  removeMarker(id: string): boolean {
    const i = this.markers.findIndex(m => m.id === id);
    if (i < 0) return false;
    this.markers.splice(i, 1);
    this.divs.get(id)?.remove();
    this.divs.delete(id);
    this.emit();
    return true;
  }

  /** Remove all markers and their label divs. */
  clear(): void {
    if (this.markers.length === 0 && this.divs.size === 0) { this.emit(); return; }
    for (const el of this.divs.values()) el.remove();
    this.divs.clear();
    this.markers = [];
    this.seqManual = 0;
    this.seqGrid = 0;
    this.emit();
  }

  /** Shallow copy of the current markers (for the panel list). */
  getMarkers(): SpotMarker[] { return this.markers.slice(); }

  /** Keyboard handling for the tool. Esc: exit placing mode and clear markers (consumed). */
  handleKey(e: KeyboardEvent): boolean {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return false;
    if (e.key !== 'Escape') return false;
    this.setActive(false);
    this.clear();
    return true;
  }

  /**
   * Per-frame label projection. Call from the render loop: each marker's div
   * is repositioned via the viewer's projectToScreen at its rendered surface
   * point (y = h * exaggeration). Offscreen/behind-camera markers hide.
   */
  update(): void {
    if (!this.terrain || this.markers.length === 0) return;
    const ex = this.terrain.exaggeration ?? 1;
    for (const m of this.markers) {
      const el = this.divs.get(m.id);
      if (!el) continue;
      let p: { x: number; y: number } | null = null;
      try { p = this.projector(m.x, m.h * ex, m.z); } catch { p = null; }
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
          p.x < -120 || p.y < -60 || p.x > this.layerW + 120 || p.y > this.layerH + 60) {
        el.style.display = 'none';
        continue;
      }
      el.style.display = 'flex';
      el.style.transform = `translate(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px) translate(-50%, -130%)`;
    }
  }

  /** Detach: remove labels, listeners, and state. */
  dispose(): void {
    window.removeEventListener('resize', this.refreshLayerSize);
    this.clear();
  }

  // ------------------------------------------------------------- internal ---

  private addMarker(m: SpotMarker, silent = false): void {
    this.markers.push(m);
    this.divs.set(m.id, this.makeLabelDiv(m));
    if (!silent) this.emit();
  }

  private makeLabelDiv(m: SpotMarker): HTMLDivElement {
    const el = document.createElement('div');
    el.className = m.source === 'grid' ? 'spot-label spot-label-grid' : 'spot-label';
    el.dataset.spotId = m.id;
    // Inline baseline styling so the tool works with no stylesheet changes;
    // the UI specialist can restyle via .spot-label in styles.css.
    el.style.cssText =
      'position:absolute;left:0;top:0;pointer-events:none;display:none;' +
      'align-items:center;gap:5px;' +
      'background:rgba(2,6,23,0.85);border:1px solid rgba(45,212,191,0.55);color:#e2e8f0;' +
      'font:600 11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;' +
      'padding:1px 7px;border-radius:8px;white-space:nowrap;' +
      'box-shadow:0 2px 8px rgba(0,0,0,0.45);';
    const dot = document.createElement('span');
    dot.style.cssText =
      'width:7px;height:7px;border-radius:50%;display:inline-block;flex:none;' +
      (m.source === 'grid'
        ? 'background:#38bdf8;box-shadow:0 0 4px #38bdf8;'
        : 'background:#2dd4bf;box-shadow:0 0 4px #2dd4bf;');
    const txt = document.createElement('span');
    txt.textContent = fmtSpot(m);
    el.append(dot, txt);
    this.layer.appendChild(el);
    return el;
  }

  private emit(): void {
    this.onChange?.(this.getMarkers());
  }
}
