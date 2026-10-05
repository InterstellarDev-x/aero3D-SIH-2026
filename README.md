# Aero3D — Single-View Elevation → 3D Flythrough (Prototype)

SIH 2026 · ISRO problem-statement prototype: turn a single optical RGB image into an
elevation map and a navigable 3D terrain — **fully simulated pipeline, no real model runs**.

## What it does

1. **Input** — pick a built-in survey scene (Urban / Hilly / Forested) or upload your own
   PNG / JPG / TIFF. `.tif` inputs take the *georeferenced* (absolute DSM) path, everything
   else takes the *non-georeferenced* (relative rDSM) path.
2. **Simulated pipeline** — animated stages: ingest → monocular depth inference (fake) →
   SRTM-anchored scale calibration (fake) → mesh generation → texture projection.
3. **3D flythrough** — Three.js terrain with the optical image draped over it:
   - 🛰 **Orbit** — rotate / zoom
   - 🚁 **Drone** — WASD + drag-look, `R`/`F` altitude, wheel = speed, `Shift` boost, 🎬 cinematic auto-fly
   - 🧍 **Walk** — first-person, click for pointer-lock mouse look
   - 📍 **Probe** — click terrain to read elevation + slope
   - Layers: optical / slope analysis / hypsometric tint, contour lines, vertical exaggeration
4. **Exports** — DSM as PNG (grayscale heightmap), survey report as `.txt`.

## Run it

```bash
cd aero3d
npm install
npm run dev      # dev server
# or
npm run build && npm run preview   # production build
```

Then open the printed URL (needs http — ES modules don't run from `file://`).

## Project layout

```
src/
  main.ts            entry
  styles.css         dark UI theme
  lib/
    rng.ts           seeded RNG, value-noise fBm
    scenes.ts        procedural survey scenes (RGB + registered heightfield)
    pipeline.ts      simulated depth estimation, calibration, metrics, colormaps
    mesh.ts          heightfield → BufferGeometry
  viewer/
    viewer.ts        Three.js scene, controls, overlays, probe
  ui/
    app.ts           screens + wiring + exports
```

## Honesty notes (for the demo)

- The "depth model" is simulated: built-in scenes reuse their known heightfield + noise
  (so the RMSE/MAE/correlation numbers are real stats of prediction-vs-truth);
  uploads derive a plausible relief from image luminance.
- Calibration, SRTM anchoring, LiDAR reference and GeoTIFF metadata are all simulated.
- The report export says this explicitly. Don't present numbers as survey-grade.
