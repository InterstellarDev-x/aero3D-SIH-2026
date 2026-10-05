import * as THREE from 'three';

/** Build (or rebuild) a heightfield terrain mesh. Y-up, centered at origin. */
export function buildTerrainGeometry(
  heights: Float32Array, n: number, worldSize: number, exaggeration: number,
): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);
  const half = worldSize / 2;
  const cell = worldSize / (n - 1);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    verts[i * 3] = -half + x * cell;
    verts[i * 3 + 1] = heights[i] * exaggeration;
    verts[i * 3 + 2] = -half + y * cell;
    uvs[i * 2] = x / (n - 1);
    uvs[i * 2 + 1] = 1 - y / (n - 1);
  }
  const idx: number[] = [];
  for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
    const a = y * n + x, b = a + 1, c = a + n, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  geo.userData.baseHeights = heights;
  geo.userData.n = n;
  return geo;
}

/** Update Y positions in place when vertical exaggeration changes. */
export function applyExaggeration(geo: THREE.BufferGeometry, exaggeration: number): void {
  const heights: Float32Array = geo.userData.baseHeights;
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < heights.length; i++) pos.setY(i, heights[i] * exaggeration);
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

/** Bilinear height sample in world coords (x,z in meters, centered). */
export function sampleHeightWorld(
  heights: Float32Array, n: number, worldSize: number, x: number, z: number,
): number {
  const half = worldSize / 2;
  const fx = ((x + half) / worldSize) * (n - 1);
  const fz = ((z + half) / worldSize) * (n - 1);
  const x0 = Math.max(0, Math.min(n - 2, Math.floor(fx)));
  const z0 = Math.max(0, Math.min(n - 2, Math.floor(fz)));
  const tx = Math.max(0, Math.min(1, fx - x0));
  const tz = Math.max(0, Math.min(1, fz - z0));
  const a = heights[z0 * n + x0], b = heights[z0 * n + x0 + 1];
  const c = heights[(z0 + 1) * n + x0], d = heights[(z0 + 1) * n + x0 + 1];
  return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
}

/**
 * Excess Green (ExG) vegetation index band coverage, computed on the CPU once
 * per scene from the optical RGB canvas. ExG = 2G - R - B (0..1 channels).
 * Bands mirror the terrain shader's uOverlay==4 tint thresholds:
 *   healthy  exg >= 0.50 (green), moderate exg >= 0.10 (yellow),
 *   stressed otherwise (red).
 * Sampling is strided (≤256² samples) so even 2k textures stay cheap.
 */
export interface ExGBands { healthy: number; moderate: number; stressed: number; }
export const EXG_T_HEALTHY = 0.5;
export const EXG_T_MODERATE = 0.1;
export const VEG_BAND_NAMES = ['Healthy', 'Moderate', 'Stressed'];
export const VEG_BAND_COLORS = ['#2eae4d', '#f2d12e', '#e0291f']; // match the shader tint ramp

/**
 * Ironbow (FLIR thermal) color ramp, mirroring the terrain shader's
 * ironbow(): black -> purple -> red -> orange -> pale yellow-white.
 * THERMAL_BAND_STOPS are the legend/elevation stops; bands are centered
 * on the inner stops with half-width 0.125.
 */
export const THERMAL_BAND_STOPS = [0, 0.25, 0.5, 0.75, 1];
export const THERMAL_BAND_NAMES = ['Cold', 'Cool', 'Warm', 'Hot', 'Peak'];

export function ironbowColor(t: number, out: THREE.Color): THREE.Color {
  const c0 = [0.00, 0.00, 0.00], c1 = [0.45, 0.05, 0.50], c2 = [0.95, 0.10, 0.05],
        c3 = [1.00, 0.65, 0.05], c4 = [1.00, 1.00, 0.90];
  const mix3 = (a: number[], b: number[], k: number) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
  const c = t < 0.25 ? mix3(c0, c1, t / 0.25)
    : t < 0.50 ? mix3(c1, c2, (t - 0.25) / 0.25)
    : t < 0.75 ? mix3(c2, c3, (t - 0.50) / 0.25)
    : mix3(c3, c4, (t - 0.75) / 0.25);
  return out.setRGB(c[0], c[1], c[2]);
}

/**
 * Elevation slice / clip-plane inspection uniforms (Cesium 'slicing' analysis
 * pattern): a draggable vertical clipping plane that discards terrain on one
 * side so the mesh renders split at the plane, with a glowing edge at the cut
 * so it reads as a cross-section wall. Wired into the terrain material by the
 * viewer's onBeforeCompile overlay injection (same pattern as the viewshed
 * DataTexture overlay).
 */
export interface ClipPlaneUniforms {
  uClipOn: { value: number };   // 1 = slice active
  uClipAxis: { value: number }; // 0 = plane at world x, 1 = plane at world z
  uClipFrac: { value: number }; // 0..1 position across [uClipMin, uClipMax]
  uClipMin: { value: number };  // world extent lower bound (meters)
  uClipMax: { value: number };  // world extent upper bound (meters)
}

export function createClipPlaneUniforms(): ClipPlaneUniforms {
  return {
    uClipOn: { value: 0 },
    uClipAxis: { value: 0 },
    uClipFrac: { value: 0.5 },
    uClipMin: { value: -1 },
    uClipMax: { value: 1 },
  };
}

/** GLSL declarations — appended to the terrain fragment <common> injection. */
export const CLIP_PLANE_GLSL_COMMON = `
uniform float uClipOn; uniform float uClipAxis; uniform float uClipFrac;
uniform float uClipMin; uniform float uClipMax;
`;

/**
 * GLSL — first statement of the terrain map_fragment injection: discard the
 * far side of the plane and tint the cut edge. diffuseColor is already the
 * sampled optical texel at this point (the original #include <map_fragment>
 * runs first).
 */
export const CLIP_PLANE_GLSL_DISCARD = `
if (uClipOn > 0.5) {
  float clipAt = mix(uClipMin, uClipMax, clamp(uClipFrac, 0.0, 1.0));
  float clipD = uClipAxis < 0.5 ? vWPos.x : vWPos.z;
  if (clipD > clipAt) discard; // keep the near side, cut the far side
  // glowing slice-face edge so the cut reads as a cross-section wall
  float edgeW = (uClipMax - uClipMin) * 0.012;
  float edge = 1.0 - smoothstep(0.0, edgeW, clipAt - clipD);
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.13, 0.84, 1.0), edge * 0.85);
}
`;

/**
 * Excess Green (ExG) vegetation index band coverage, computed on the CPU once
 * per scene from the optical RGB canvas. ExG = 2G - R - B (0..1 channels).
 * Bands mirror the terrain shader's uOverlay==4 tint thresholds:
 *   healthy  exg >= 0.50 (green), moderate exg >= 0.10 (yellow),
 *   stressed otherwise (red).
 * Sampling is strided (≤256² samples) so even 2k textures stay cheap.
 */
export function computeExGBands(canvas: HTMLCanvasElement): ExGBands {
  const w = canvas.width, h = canvas.height;
  if (w <= 0 || h <= 0) return { healthy: 0, moderate: 0, stressed: 0 };
  const ctx = canvas.getContext('2d');
  if (!ctx) return { healthy: 0, moderate: 0, stressed: 0 };
  const stride = Math.max(1, Math.floor(Math.min(w, h) / 256));
  const img = ctx.getImageData(0, 0, w, h).data;
  let healthy = 0, moderate = 0, stressed = 0, n = 0;
  for (let y = 0; y < h; y += stride) {
    const row = y * w;
    for (let x = 0; x < w; x += stride) {
      const k = (row + x) * 4;
      const r = img[k] / 255, g = img[k + 1] / 255, b = img[k + 2] / 255;
      const exg = 2 * g - r - b;
      if (exg >= EXG_T_HEALTHY) healthy++;
      else if (exg >= EXG_T_MODERATE) moderate++;
      else stressed++;
      n++;
    }
  }
  return { healthy: healthy / n, moderate: moderate / n, stressed: stressed / n };
}
