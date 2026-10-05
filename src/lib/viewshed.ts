// Viewshed / line-of-sight analysis over the unexaggerated DSM.
// Classic radial sweep: for each azimuth, walk outward from the observer
// tracking the maximum elevation angle; cells whose angle meets or exceeds
// it are intervisible (Esri/Supergeo viewshed convention).

import { sampleHeightWorld } from './mesh';

export interface ViewshedResult {
  vis: Uint8Array; // n*n, 1 = visible from the observer, 0 = occluded
  n: number;
  pct: number;     // fraction of the grid visible (0..1)
}

/**
 * Compute a 360° visibility grid over the DSM.
 * heights: unexaggerated heights in meters, n x n grid, worldSize meters wide.
 * (ox, oz): observer ground position in world coords.
 * eyeH: observer eye height in meters above datum (ground + tower/mast height).
 */
export function computeViewshed(
  heights: Float32Array, n: number, worldSize: number,
  ox: number, oz: number, eyeH: number,
): ViewshedResult {
  const vis = new Uint8Array(n * n);
  const half = worldSize / 2;
  const cell = worldSize / (n - 1);
  const maxD = worldSize * 1.5; // reach the far corner from anywhere on the grid
  const step = cell * 0.75;
  const nRays = Math.max(720, Math.min(7200, Math.ceil((2 * Math.PI * maxD) / step)));
  let count = 0;
  for (let r = 0; r < nRays; r++) {
    const a = (r / nRays) * Math.PI * 2;
    const dx = Math.cos(a), dz = Math.sin(a);
    let maxAng = -Infinity;
    for (let d = 0; d <= maxD; d += step) {
      const x = ox + dx * d, z = oz + dz * d;
      if (x < -half || x > half || z < -half || z > half) break;
      const h = sampleHeightWorld(heights, n, worldSize, x, z);
      const ang = Math.atan2(h - eyeH, d);
      if (ang >= maxAng) {
        maxAng = ang;
        const col = Math.max(0, Math.min(n - 1, Math.round(((x + half) / worldSize) * (n - 1))));
        const row = Math.max(0, Math.min(n - 1, Math.round(((z + half) / worldSize) * (n - 1))));
        const i = row * n + col;
        if (!vis[i]) { vis[i] = 1; count++; }
      }
    }
  }
  return { vis, n, pct: count / (n * n) };
}
