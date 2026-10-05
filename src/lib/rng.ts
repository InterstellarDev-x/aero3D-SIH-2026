// Seeded RNG + value-noise fBm. Deterministic so demo scenes are reproducible.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash2(x: number, y: number, seed: number): number {
  let h = seed + x * 374761393 + y * 668265263;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

function smooth(t: number): number { return t * t * (3 - 2 * t); }

/** Value noise in [0,1] */
export function valueNoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const a = hash2(xi, yi, seed), b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
  const u = smooth(xf), v = smooth(yf);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/** Fractal Brownian motion in [0,1] */
export function fbm(x: number, y: number, seed: number, octaves = 4): number {
  let sum = 0, amp = 0.5, freq = 1, norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise(x * freq, y * freq, seed + o * 101);
    norm += amp; amp *= 0.5; freq *= 2.03;
  }
  return sum / norm;
}

/** Ridged fBm in [0,1] — sharp crests, good for hills */
export function ridged(x: number, y: number, seed: number, octaves = 4): number {
  let sum = 0, amp = 0.55, freq = 1, norm = 0;
  for (let o = 0; o < octaves; o++) {
    const n = valueNoise(x * freq, y * freq, seed + o * 131);
    sum += amp * (1 - Math.abs(2 * n - 1));
    norm += amp; amp *= 0.5; freq *= 2.11;
  }
  return sum / norm;
}

/** Box blur on a Float32Array grid (in place, 1 pass) */
export function blur(grid: Float32Array, n: number): void {
  const tmp = new Float32Array(grid);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < n && yy < n) { s += tmp[yy * n + xx]; c++; }
      }
      grid[y * n + x] = s / c;
    }
  }
}

/** Gaussian-ish random via CLT */
export function gauss(rand: () => number): number {
  return (rand() + rand() + rand() + rand() - 2) * 1.2;
}
