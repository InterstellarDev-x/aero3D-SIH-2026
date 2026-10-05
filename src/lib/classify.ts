// SIMULATED land-cover classification head. Built-in scenes carry per-pixel
// ground-truth labels (we generated them), so the "model" returns truth +
// a little label noise — exactly what a strong segmentation output looks
// like. Uploads fall back to color heuristics.

import { mulberry32 } from './rng';
import { LCLASS, LCLASS_NAMES, LCLASS_COLORS, type SceneData } from './scenes';

export { LCLASS, LCLASS_NAMES, LCLASS_COLORS };

export interface ClassificationResult {
  labels: Uint8Array;      // N*N, class ids 0..4
  size: number;
  model: string;
  coverage: number[];      // fraction of pixels per class
  perClassIoU: number[] | null; // vs truth (built-in scenes only)
  meanIoU: number | null;
  heuristic: boolean;      // true for uploads (no learned model)
}

function coverageOf(labels: Uint8Array, nClasses: number): number[] {
  const c = new Array(nClasses).fill(0);
  for (let i = 0; i < labels.length; i++) c[labels[i]]++;
  return c.map(v => v / labels.length);
}

/** Built-in scene: truth labels + 2.5% flip noise, plus fake IoU metrics. */
export function classifyScene(scene: SceneData): ClassificationResult {
  const rand = mulberry32(4242);
  const labels = new Uint8Array(scene.labels);
  const n = labels.length;
  const flips = Math.floor(n * 0.025);
  for (let k = 0; k < flips; k++) {
    const i = Math.floor(rand() * n);
    labels[i] = Math.floor(rand() * LCLASS_NAMES.length);
  }
  // plausible per-class IoU (deterministic-ish, high 0.8s-0.9s)
  const perClassIoU = LCLASS_NAMES.map((_, c) => {
    const base = [0.83, 0.91, 0.88, 0.93, 0.86][c] ?? 0.85;
    return Math.min(0.99, base + (rand() - 0.5) * 0.04);
  });
  const meanIoU = perClassIoU.reduce((a, b) => a + b, 0) / perClassIoU.length;
  return {
    labels, size: scene.size,
    model: 'seg-landcover v1.3 (simulated)',
    coverage: coverageOf(labels, LCLASS_NAMES.length),
    perClassIoU, meanIoU, heuristic: false,
  };
}

/** Uploads: color-heuristic segmentation (no truth available). */
export function classifyUpload(img: ImageData, size: number): ClassificationResult {
  const n = size * size;
  const labels = new Uint8Array(n);
  const rand = mulberry32(777);
  for (let y = 0; y < size; y++) for (let x = 0; x < n / size; x++) {
    const o = (y * size + x) * 4;
    const r = img.data[o] / 255, g = img.data[o + 1] / 255, b = img.data[o + 2] / 255;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    let c: number = LCLASS.GROUND;
    if (b > r + 0.08 && b > g + 0.04) c = LCLASS.WATER;
    else if (g > r + 0.06 && g > b + 0.02) c = LCLASS.VEGETATION;
    else if (lum < 0.25) c = LCLASS.ROAD;
    else if (lum > 0.62) c = LCLASS.BUILDING;
    if (rand() < 0.04) c = Math.floor(rand() * LCLASS_NAMES.length); // speckle
    labels[y * size + x] = c;
  }
  return {
    labels, size,
    model: 'color-heuristic fallback (simulated)',
    coverage: coverageOf(labels, LCLASS_NAMES.length),
    perClassIoU: null, meanIoU: null, heuristic: true,
  };
}

/** Render labels to a color-map canvas for previews. */
export function renderLabels(labels: Uint8Array, size: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < labels.length; i++) {
    const [r, g, b] = LCLASS_COLORS[labels[i]] ?? [128, 128, 128];
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export function cssColor(c: [number, number, number]): string {
  return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
}

/**
 * Nearest land-cover class id at grid coords (fx,fz in [0,size-1]) — clamped.
 * The label grids share the DSM's N×N geometry, so spot-elevation markers can
 * report the surface class at their point for field verification.
 */
export function classAt(labels: Uint8Array, size: number, fx: number, fz: number): number {
  if (size < 1 || labels.length === 0) return 0;
  const x = Math.max(0, Math.min(size - 1, Math.round(fx)));
  const z = Math.max(0, Math.min(size - 1, Math.round(fz)));
  return labels[z * size + x] ?? 0;
}
