// Aero3D app orchestration: landing -> simulated pipeline -> 3D viewer.
import { Vector3 } from 'three';
import { generateScene, gridStats, type SceneData, type LandscapeKind } from '../lib/scenes';
import { estimateDepth, calibrate, computeMetrics, renderColormap, type Calibration, type DepthResult } from '../lib/pipeline';
import {
  computeProfile, fmtProfile, profileChartSVG, profileCursorXY, PROFILE_CHART,
  synthesizePostEventDSM, packChangeTexture, fmtChangeDetection,
  SpotElevationTool, fmtSpot, spotStats,
  type ProfileResult, type PostEventResult, type SpotMarker,
} from '../lib/pipeline';
import { classifyScene, classifyUpload, renderLabels, classAt, LCLASS_NAMES, LCLASS_COLORS, cssColor, type ClassificationResult } from '../lib/classify';
import { TerrainViewer, type Bookmark, type ViewMode, type Overlay } from '../viewer/viewer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
/**
 * Shared SVG icon: every UI icon resolves through the single inline sprite in
 * index.html (Lucide-style, 24x24 viewBox, ~1.5px stroke, currentColor), so
 * stroke weight and color stay consistent across all chrome.
 */
const uiIcon = (name: string): string =>
  `<svg class="svg-ic" aria-hidden="true" focusable="false"><use href="#i-${name}"/></svg>`;
const SIZE = 256;

interface Job {
  scene: SceneData | null;
  rgb: HTMLCanvasElement;
  name: string;
  georeferenced: boolean;
  formatLabel: string;
}

// ----------------------------------------------------------- toasts ---
/** Non-blocking toast notifications for actions, hints and errors. */
function toast(msg: string, kind: 'info' | 'ok' | 'warn' | 'err' = 'info', ms = 3400): void {
  const box = $('toasts');
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  // severity is conveyed by the small colored dot (see .t-dot in styles.css)
  el.innerHTML = `<span class="t-dot" aria-hidden="true"></span><span>${msg}</span>`;
  box.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 320);
  }, ms);
}

// ----------------------------------------------------- error screens ---
/** True when the browser can provide a WebGL context for the 3D viewer. */
function webglSupported(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}

/**
 * Designed full-screen fallback for fatal failures (no WebGL, viewer crash).
 * Renders above every screen as a fixed overlay; "Back to scene select" tears
 * down any half-built viewer and returns to the landing page.
 */
function showErrorScreen(title: string, msg: string): void {
  dismissErrorScreen();
  const ov = document.createElement('div');
  ov.id = 'err-screen';
  ov.className = 'err-screen';
  ov.setAttribute('role', 'alertdialog');
  ov.setAttribute('aria-modal', 'true');
  ov.setAttribute('aria-label', title);
  const card = document.createElement('div');
  card.className = 'err-card';
  const icon = document.createElement('div');
  icon.className = 'ec-icon';
  icon.innerHTML = uiIcon('satellite');
  const h = document.createElement('h2');
  h.textContent = title;
  const p = document.createElement('p');
  p.textContent = msg;
  const actions = document.createElement('div');
  actions.className = 'ec-actions';
  const back = document.createElement('button');
  back.className = 'cta-btn';
  back.innerHTML = uiIcon('arrow-left') + ' Back to scene select';
  back.onclick = () => {
    dismissErrorScreen();
    cancelAnimationFrame(raf);
    viewer?.dispose(); viewer = null;
    $('screen-viewer').classList.add('hidden');
    $('screen-processing').classList.add('hidden');
    $('screen-landing').classList.remove('hidden');
    window.scrollTo(0, 0);
  };
  const reload = document.createElement('button');
  reload.className = 'tb-btn ghost';
  reload.innerHTML = uiIcon('rotate-ccw') + ' Reload page';
  reload.onclick = () => location.reload();
  actions.append(back, reload);
  card.append(icon, h, p, actions);
  ov.append(card);
  document.body.appendChild(ov);
  back.focus();
}

function dismissErrorScreen(): void {
  document.getElementById('err-screen')?.remove();
}

/** Designed inline error state for the processing screen: toast + card, never alert(). */
function showProcessingError(msg: string): void {
  ($('proc-error-msg') as HTMLElement).textContent = msg;
  document.querySelector('#screen-processing .proc-grid')?.classList.add('hidden');
  $('proc-error').classList.remove('hidden');
  ($('btn-error-retry') as HTMLElement).focus();
}

function hideProcessingError(): void {
  document.querySelector('#screen-processing .proc-grid')?.classList.remove('hidden');
  $('proc-error').classList.add('hidden');
}

function wireProcessingError(): void {
  ($('btn-error-retry') as HTMLElement).onclick = () => { if (pendingJob) startJob(pendingJob); };
  ($('btn-error-back') as HTMLElement).onclick = () => {
    hideProcessingError();
    $('screen-processing').classList.add('hidden');
    $('screen-landing').classList.remove('hidden');
    window.scrollTo(0, 0);
  };
}

// ---------------------------------------------------------------- landing ---
// Demo pick order for the scene cards. SceneData carries no priority field,
// so the recommended first-look order lives here (urban first: georeferenced +
// all five land-cover classes → the strongest SIH demo path).
interface DemoPick { kind: LandscapeKind; fmt: string; rel: boolean; pick: string }
const DEMO_PICKS: DemoPick[] = [
  { kind: 'olympic', fmt: 'PNG · real photo', rel: true, pick: 'Best for demo' },
  { kind: 'urban', fmt: 'GeoTIFF · georeferenced', rel: false, pick: 'Procedural urban' },
  { kind: 'hilly', fmt: 'GeoTIFF · georeferenced', rel: false, pick: 'Relief showcase' },
  { kind: 'forest', fmt: 'PNG · non-georeferenced', rel: true, pick: 'Relative (rDSM) path' },
];

/** Scene-selection card with stat overlays read from the scene metadata. */
function sceneCard(kind: LandscapeKind, meta: { fmt: string; rel: boolean; pick: string }, order: number): HTMLButtonElement {
  const scene = generateScene(kind);
  const btn = document.createElement('button');
  btn.className = 'card';
  btn.setAttribute('aria-label', `Process scene: ${scene.name}`);
  const cv = document.createElement('canvas');
  cv.width = SIZE; cv.height = SIZE;
  cv.getContext('2d')!.drawImage(scene.rgb, 0, 0);
  const thumb = document.createElement('div');
  thumb.className = 'card-thumb';
  thumb.append(cv);
  const rib = document.createElement('span');
  rib.className = 'card-ribbon' + (order === 1 ? ' best' : '');
  rib.innerHTML = (order === 1 ? uiIcon('star') + ' ' : '') + meta.pick;
  const num = document.createElement('span');
  num.className = 'card-order';
  num.textContent = String(order);
  num.title = `Demo pick #${order}`;
  const hypso = document.createElement('div');
  hypso.className = 'card-hypso';
  hypso.title = 'Hypsometric tint (elevation)';
  thumb.append(rib, num, hypso);
  const stats = document.createElement('div');
  stats.className = 'card-stats';
  const km = (scene.worldSize / 1000).toFixed(1);
  const lo = Math.round(scene.baseElevation), hi = Math.round(scene.baseElevation + scene.relief);
  stats.innerHTML =
    `<span title="Min/max elevation (from scene metadata)">${uiIcon('mountain')} ${lo}–${hi} m</span>` +
    `<span title="Footprint">${uiIcon('grid')} ${km}×${km} km</span>` +
    `<span title="Ground sample distance">${uiIcon('ruler')} ${(scene.worldSize / SIZE).toFixed(0)} m/px</span>`;
  thumb.append(stats);
  const body = document.createElement('div');
  body.className = 'card-body';
  body.innerHTML = `<h3>${scene.name}</h3><p>${scene.blurb}</p>
    <span class="fmt${meta.rel ? ' rel' : ''}">${meta.fmt}</span>`;
  btn.append(thumb, body);
  btn.onclick = () => startJob({
    scene, rgb: scene.rgb, name: scene.name,
    georeferenced: !meta.rel, formatLabel: meta.fmt,
  });
  return btn;
}

/** Animated marquee of scene previews on the landing page (duplicated for a seamless loop). */
function stripItem(scene: SceneData, fmt: string): HTMLElement {
  const fig = document.createElement('div');
  fig.className = 'strip-item';
  const cv = document.createElement('canvas');
  cv.width = SIZE; cv.height = SIZE;
  cv.getContext('2d')!.drawImage(scene.rgb, 0, 0);
  const chip = document.createElement('span');
  chip.className = 'strip-chip';
  chip.textContent = fmt;
  const lbl = document.createElement('span');
  lbl.className = 'strip-lbl';
  lbl.textContent = scene.name;
  fig.append(cv, chip, lbl);
  return fig;
}

/** Animated marquee of scene previews on the landing page (duplicated for a seamless loop). */
function buildStrip(): void {
  const track = $('strip-track');
  if (!track) return;
  const kinds: { kind: LandscapeKind; fmt: string }[] = [
    { kind: 'urban', fmt: 'GeoTIFF' }, { kind: 'hilly', fmt: 'GeoTIFF' }, { kind: 'forest', fmt: 'PNG' },
    { kind: 'urban', fmt: 'GeoTIFF' }, { kind: 'hilly', fmt: 'GeoTIFF' }, { kind: 'forest', fmt: 'PNG' },
  ];
  const first = kinds.map(k => stripItem(generateScene(k.kind), k.fmt));
  const second = kinds.map(k => stripItem(generateScene(k.kind), k.fmt));
  // two identical halves so the -50% translate loops seamlessly
  track.append(...first, ...second);
}

/** Same easing the viewer's fly-to camera path uses (ease-in-out cubic). */
const easeInOutCubic = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** Animated "intro camera path" over a preview render of the hilly scene. */
function buildHeroPreview(): void {
  const cv = $('hero-preview') as HTMLCanvasElement | null;
  const fig = $('hero-visual');
  if (!cv || !fig) return;
  const scene = generateScene('hilly');
  const ctx = cv.getContext('2d')!;
  const W = cv.width, H = cv.height, SW = scene.rgb.width, SH = scene.rgb.height;
  const draw = (cx: number, cy: number, z: number) => {
    const sw = SW * z, sh = sw * (H / W);
    const sx = Math.min(SW - sw, Math.max(0, cx * SW - sw / 2));
    const sy = Math.min(SH - sh, Math.max(0, cy * SH - sh / 2));
    ctx.drawImage(scene.rgb, sx, sy, sw, sh, 0, 0, W, H);
  };
  // camera waypoints as fractions of the source frame: {x, y, zoom}
  const path = [
    { x: 0.30, y: 0.62, z: 0.42 }, // valley approach
    { x: 0.50, y: 0.45, z: 0.28 }, // push in toward the ridge
    { x: 0.66, y: 0.54, z: 0.46 }, // sweep across the ridgeline
  ];
  const legs = path.length - 1;
  const frame = (now: number) => {
    const cycle = 2 * legs * 5200; // ping-pong there and back, 5.2 s per leg
    const t = (now % cycle) / cycle; // 0..1
    let pt = (t % 0.5) / 0.5; // 0..1 along the full path
    if (t >= 0.5) pt = 1 - pt; // return leg
    const leg = Math.min(legs - 1, Math.floor(pt * legs));
    const k = easeInOutCubic(pt * legs - leg);
    const a = path[leg], b = path[leg + 1];
    draw(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.z + (b.z - a.z) * k);
    requestAnimationFrame(frame);
  };
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    draw(path[0].x, path[0].y, path[0].z); // static frame, no motion
  } else {
    requestAnimationFrame(frame);
  }
  // clicking the preview loads the demo scene directly
  const go = () => startJob({
    scene, rgb: scene.rgb, name: scene.name,
    georeferenced: true, formatLabel: 'GeoTIFF · georeferenced',
  });
  fig.addEventListener('click', go);
  fig.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
  });
}

/** Designed empty state for the landing scene grid when no demo scene can be prepared. */
function showCardsEmpty(detail: string): void {
  const wrap = $('scene-cards');
  wrap.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'empty-card';
  div.setAttribute('role', 'status');
  const icon = document.createElement('div');
  icon.className = 'ec-icon';
  icon.innerHTML = uiIcon('satellite');
  const h = document.createElement('h3');
  h.textContent = 'No survey scenes available';
  const p = document.createElement('p');
  p.textContent = `We couldn't prepare the demo scenes (${detail}). You can still upload your own imagery below, or reload to try again.`;
  const btn = document.createElement('button');
  btn.className = 'cta-btn';
  btn.innerHTML = uiIcon('rotate-ccw') + ' Reload page';
  btn.onclick = () => location.reload();
  div.append(icon, h, p, btn);
  wrap.append(div);
}

function buildLanding(): void {
  try {
    const wrap = $('scene-cards');
    DEMO_PICKS.forEach((p, i) => wrap.append(sceneCard(p.kind, p, i + 1)));
  } catch (err) {
    console.error('[aero3d] scene cards failed:', err);
    showCardsEmpty(err instanceof Error ? err.message : 'unknown error');
    toast('The demo scenes could not be prepared — you can still upload your own imagery.', 'warn');
  }
  try { buildStrip(); } catch (err) { console.error('[aero3d] preview strip failed:', err); }
  try { buildHeroPreview(); } catch (err) { console.error('[aero3d] hero preview failed:', err); }
  const demoBtn = $('hero-demo-btn');
  if (demoBtn) demoBtn.addEventListener('click', (e) => {
    e.preventDefault(); // anchor href is only a fallback
    const first = DEMO_PICKS[0];
    const scene = generateScene(first.kind);
    startJob({
      scene, rgb: scene.rgb, name: scene.name,
      georeferenced: !first.rel, formatLabel: first.fmt,
    });
  });
  const dz = $('dropzone');
  const fi = $('file-input') as HTMLInputElement;
  dz.onclick = () => fi.click();
  dz.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); }
  };
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('over'); };
  dz.ondragleave = () => dz.classList.remove('over');
  dz.ondrop = (e) => {
    e.preventDefault(); dz.classList.remove('over');
    const f = e.dataTransfer?.files?.[0];
    if (f) handleUpload(f);
  };
  fi.onchange = () => { const f = fi.files?.[0]; if (f) handleUpload(f); fi.value = ''; };
}

/**
 * Upload flow: every upload goes through the processing screen first, so a bad
 * file or a decode failure surfaces as toast + inline error card (never alert()).
 */
async function handleUpload(file: File): Promise<void> {
  const okType = /\.(png|jpe?g|tif|tiff)$/i.test(file.name) || file.type.startsWith('image/');
  // Route uploads through the processing screen so failures get the designed
  // inline error state.
  pendingJob = null;
  hideProcessingError();
  $('screen-landing').classList.add('hidden');
  $('screen-viewer').classList.add('hidden');
  $('screen-processing').classList.remove('hidden');
  $('proc-scene-name').textContent = '— ' + file.name;
  buildStageList();
  ($('proc-eta') as HTMLElement).textContent = '';
  $('proc-log').innerHTML = '';
  log(`upload: <span class="inf">${file.name}</span> (${(file.size / 1024).toFixed(0)} KB)`, 'inf');
  if (!okType) {
    const msg = `"${file.name}" is not a supported image — please upload a PNG, JPG or TIFF.`;
    log('upload rejected: unsupported file type', 'err');
    toast(msg, 'err', 5000);
    showProcessingError(msg);
    return;
  }
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file);
  } catch {
    const msg = 'Could not decode that image in this browser. Try exporting it as a PNG or JPG first.';
    log('upload rejected: image decode failed', 'err');
    toast(msg, 'err', 5000);
    showProcessingError(msg);
    return;
  }
  log('image decoded — handing off to the pipeline', 'ok');
  const cv = document.createElement('canvas');
  cv.width = SIZE; cv.height = SIZE;
  const ctx = cv.getContext('2d')!;
  // cover-fit crop
  const s = Math.max(SIZE / bmp.width, SIZE / bmp.height);
  const w = bmp.width * s, h = bmp.height * s;
  ctx.drawImage(bmp, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  const geo = /\.tif?$/i.test(file.name);
  await startJob({
    scene: null, rgb: cv, name: file.name,
    georeferenced: geo,
    formatLabel: geo ? 'GeoTIFF · georeferenced' : 'Image · non-georeferenced',
  });
}

// -------------------------------------------------------------- processing ---
const STAGES = [
  { title: 'Ingesting optical imagery', sub: 'decode → 256 m grid resample → cloud mask (simulated)', estMs: 900 },
  { title: 'Monocular depth inference', sub: 'mono-depth v2.1 backbone · relative surface model', estMs: 1500 },
  { title: 'Scale calibration', sub: 'SRTM-30m anchor / relative datum · land-cover head', estMs: 2600 },
  { title: 'Mesh generation', sub: 'heightfield → 130k-tri mesh · texture projection', estMs: 1100 },
  { title: 'Compiling 3D scene', sub: 'lighting · shadows · navigation rigs', estMs: 1200 },
];

const STAGE_TIMES_KEY = 'aero3d-stage-times';

function stageEstimates(): number[] {
  let saved: number[] = [];
  try {
    const raw = localStorage.getItem(STAGE_TIMES_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    if (Array.isArray(arr)) saved = arr.filter((x) => typeof x === 'number' && x > 0);
  } catch { /* ignore */ }
  return STAGES.map((s, i) => saved[i] ?? s.estMs);
}

function saveStageTime(i: number, ms: number): void {
  const est = stageEstimates();
  est[i] = Math.round(ms);
  try { localStorage.setItem(STAGE_TIMES_KEY, JSON.stringify(est)); } catch { /* ignore */ }
}

const fmtS = (ms: number) => ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : Math.round(ms) + 'ms';

function log(msg: string, cls = ''): void {
  const el = $('proc-log');
  const t = new Date().toISOString().slice(11, 19);
  const div = document.createElement('div');
  div.innerHTML = `<span class="t">${t}</span><span class="${cls}">${msg}</span>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function setStage(i: number, state: 'active' | 'done'): void {
  const lis = $('stage-list').children;
  for (let k = 0; k < lis.length; k++) {
    lis[k].classList.remove('active');
    if (k < i) lis[k].classList.add('done');
  }
  if (lis[i]) {
    lis[i].classList.add(state);
    (lis[i].querySelector('.st-ic') as HTMLElement).innerHTML = state === 'done' ? uiIcon('check') : '';
  }
}

function setProgress(i: number, pct: number): void {
  const li = $('stage-list').children[i];
  if (!li) return;
  const bar = li.querySelector('.st-bar i') as HTMLElement;
  const lbl = li.querySelector('.st-pct') as HTMLElement;
  if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (lbl) lbl.textContent = Math.round(Math.max(0, Math.min(100, pct))) + '%';
}

/** Reveal a preview pane with an animated transition once its canvas has content. */
function revealPane(phId: string): void {
  const ph = $(phId);
  const fig = ph.closest('figure');
  ph.classList.add('hidden');
  fig?.classList.add('reveal');
  // allow re-animation on the next job
  if (fig) { void fig.offsetWidth; }
}

/** Run one pipeline stage: animate progress toward its estimate, do the work, then record timing. */
async function runStage(i: number, estMs: number, body: () => Promise<void>): Promise<void> {
  setStage(i, 'active');
  const t0 = performance.now();
  let finished = false;
  const tick = () => {
    if (finished) return;
    setProgress(i, Math.min(95, ((performance.now() - t0) / estMs) * 100));
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  await body();
  finished = true;
  setProgress(i, 100);
  setStage(i, 'done');
  saveStageTime(i, performance.now() - t0);
}

/** Update the header ETA line with the remaining estimated time. */
function updateEta(doneThrough: number): void {
  const est = stageEstimates();
  const remain = est.slice(doneThrough + 1).reduce((a, b) => a + b, 0);
  const el = $('proc-eta');
  if (el) el.textContent = doneThrough >= STAGES.length - 1 ? 'done' : `≈ ${fmtS(remain)} remaining`;
}

function buildStageList(): void {
  $('stage-list').innerHTML = STAGES.map(s =>
    `<li><span class="st-ic"></span>
     <span class="st-main">${s.title}<span class="st-sub">${s.sub}</span>
     <span class="st-bar"><i></i></span></span>
     <span class="st-pct">0%</span></li>`).join('');
}

/** The job the "Try again" button on the processing error card re-runs. */
let pendingJob: Job | null = null;

/** Guarded entry point for every scene-load path (cards, hero, demo button, uploads). */
async function startJob(job: Job): Promise<void> {
  pendingJob = job;
  if (!webglSupported()) {
    toast('WebGL is not available in this browser — the 3D viewer needs it.', 'err', 5000);
    showErrorScreen(
      'WebGL unavailable',
      'Aero3D needs WebGL to render the 3D flythrough, and this browser could not provide it. ' +
      'Try a recent Chrome, Edge, Firefox or Safari with hardware acceleration enabled, then reload.'
    );
    return;
  }
  hideProcessingError();
  $('screen-landing').classList.add('hidden');
  $('screen-viewer').classList.add('hidden');
  $('screen-processing').classList.remove('hidden');
  try {
    await runPipeline(job);
  } catch (err) {
    console.error('[aero3d] pipeline failed:', err);
    const msg = err instanceof Error ? err.message : 'The pipeline stopped unexpectedly.';
    toast('Processing failed — see the message on screen.', 'err', 5000);
    showProcessingError(msg + ' You can try again or go back to the scene select.');
  }
}

async function runPipeline(job: Job): Promise<void> {
  $('proc-scene-name').textContent = '— ' + job.name;
  buildStageList();
  const est = stageEstimates();
  const total = est.reduce((a, b) => a + b, 0);
  $('proc-eta').textContent = `≈ ${fmtS(total)} remaining (simulated)`;
  $('proc-log').innerHTML = '';
  ($('pv-rgb') as HTMLCanvasElement).getContext('2d')!.drawImage(job.rgb, 0, 0);
  document.querySelectorAll('#screen-processing .proc-right figure').forEach(f => f.classList.remove('reveal'));
  $('pv-depth-ph').classList.remove('hidden');
  $('pv-dsm-ph').classList.remove('hidden');
  $('pv-class-ph').classList.remove('hidden');

  const img = job.rgb.getContext('2d')!.getImageData(0, 0, SIZE, SIZE);

  await runStage(0, est[0], async () => {
    log(`ingest: <span class="inf">${job.name}</span> (${job.formatLabel})`, 'inf');
    await sleep(650);
    log(`resampled to ${SIZE}×${SIZE} grid · GSD ≈ ${(job.scene?.worldSize ?? 1200) / SIZE | 0} m/px`, 'ok');
  });
  updateEta(0);

  let depth: DepthResult;
  await runStage(1, est[1], async () => {
    log('loading mono-depth v2.1 weights… (simulated)', 'inf');
    await sleep(900);
    depth = estimateDepth(job.scene, img, SIZE);
    log(`inference done in ${(depth.inferenceMs / 1000).toFixed(1)}s · output rDSM 0..1`, 'ok');
    const dMin = 0, dMax = 1;
    ($('pv-depth') as HTMLCanvasElement).getContext('2d')!.drawImage(renderColormap(depth.relative, SIZE, dMin, dMax), 0, 0);
    revealPane('pv-depth-ph');
    await sleep(350);
  });
  updateEta(1);

  let cal: Calibration;
  let cls: ClassificationResult;
  await runStage(2, est[2], async () => {
    const base = job.scene?.baseElevation ?? 480 + Math.random() * 40;
    const relief = job.scene?.relief ?? 60 + Math.random() * 30;
    if (job.georeferenced) {
      log('georeferenced input: EPSG:4326 metadata detected (simulated)', 'inf');
      await sleep(700);
      log('fetching SRTM 30m tile… bilinear anchor fit over 9 control windows', 'inf');
      await sleep(800);
      log('affine scale+offset solved · residual 1.8 m', 'ok');
      cal = calibrate(depth!.relative, SIZE, { georeferenced: true, baseElevation: Math.round(base), relief: Math.round(relief) });
    } else {
      log('non-georeferenced input: keeping relative datum (rDSM)', 'inf');
      await sleep(900);
      log('scene-level statistics → normalized relief model', 'ok');
      cal = calibrate(depth!.relative, SIZE, { georeferenced: false, baseElevation: Math.round(base), relief: Math.round(relief) });
    }
    $('pv-dsm-tag').textContent = job.georeferenced ? 'absolute, m MSL' : 'relative datum';
    ($('pv-dsm') as HTMLCanvasElement).getContext('2d')!.drawImage(renderColormap(cal!.absolute, SIZE, cal!.minH, cal!.maxH), 0, 0);
    revealPane('pv-dsm-ph');
    log(`DSM range ${cal!.minH.toFixed(1)} – ${cal!.maxH.toFixed(1)} m`, 'ok');

    // land-cover classification head (simulated)
    log('semantic head: seg-landcover v1.3 — 5 classes (simulated)', 'inf');
    await sleep(500);
    cls = job.scene ? classifyScene(job.scene) : classifyUpload(img, SIZE);
    ($('pv-class') as HTMLCanvasElement).getContext('2d')!.drawImage(renderLabels(cls!.labels, SIZE), 0, 0);
    revealPane('pv-class-ph');
    log(`land-cover mapped · mIoU ${cls!.meanIoU !== null ? cls!.meanIoU.toFixed(3) : 'n/a (heuristic)'}`, 'ok');
    await sleep(350);
  });
  updateEta(2);

  await runStage(3, est[3], async () => {
    log('triangulating 130,050 faces · projecting optical texture…', 'inf');
    await sleep(950);
    log('mesh ready · 65,536 vertices', 'ok');
  });
  updateEta(3);

  await runStage(4, est[4], async () => {
    await sleep(650);
    log('scene compiled — entering flythrough', 'ok');
    await sleep(400);
  });
  updateEta(4);

  enterViewer(job, cal!, depth!, cls!);
}

// ----------------------------------------------------------------- viewer ---
let viewer: TerrainViewer | null = null;
let raf = 0;
let activeJob: Job | null = null;
let activeCal: Calibration | null = null;
let activeCls: ClassificationResult | null = null;
let activeDepth: DepthResult | null = null;
let changeResult: PostEventResult | null = null; // post-monsoon DSM + Δ-erosion stats
let spotTool: SpotElevationTool | null = null; // Pix4D-style spot-elevation markers

const HINTS: Record<ViewMode, string> = {
  orbit: 'Drag to orbit · wheel to zoom · V to switch mode · enable Probe to read heights',
  drone: 'WASD fly · drag to look · R/F up/down · wheel = speed · auto-fly for cinema · V to switch',
  fpv: 'Click scene for mouse-look · WASD to walk (1.7 m eye height) · Shift to run · V to switch',
};

/** Crash-guarded wrapper: a viewer failure becomes the designed error screen, not a dead page. */
function enterViewer(job: Job, cal: Calibration, depth: DepthResult, cls: ClassificationResult): void {
  try {
    enterViewerInner(job, cal, depth, cls);
  } catch (err) {
    console.error('[aero3d] viewer failed:', err);
    cancelAnimationFrame(raf);
    viewer?.dispose(); viewer = null;
    toast('The 3D viewer failed to start — see the message on screen.', 'err', 5000);
    showErrorScreen(
      'Viewer failed to start',
      'Something went wrong while building the 3D scene. This input may be unusual — ' +
      'try another scene or your upload again, or reload the page.'
    );
  }
}

function enterViewerInner(job: Job, cal: Calibration, depth: DepthResult, cls: ClassificationResult): void {
  activeJob = job; activeCal = cal; activeCls = cls; activeDepth = depth;
  changeResult = null; // fresh scene → no change-detection stat yet
  $('screen-processing').classList.add('hidden');
  $('screen-viewer').classList.remove('hidden');

  viewer?.dispose();
  viewer = new TerrainViewer($('gl'));
  const scene = job.scene ?? synthSceneForUpload(job, depth);
  viewer.load(scene, cal, cls);
  viewer.setScan(true); // survey scan-wave on by default for the demo
  applyQuality(viewer, loadQuality()); // restore the saved render-quality preset
  // ---- Spot-elevation tool (Pix4D-style field verification) ----
  spotTool?.dispose();
  spotTool = new SpotElevationTool({
    layer: $('spot-layer'),
    projectToScreen: (x, y, z) => viewer!.projectToScreen(x, y, z),
    onChange: (markers) => renderSpots(markers),
  });
  spotTool.setTerrain({ heights: cal.absolute, size: SIZE, worldSize: scene.worldSize, exaggeration: viewer.getExaggeration() });
  viewer.onSpotPlace = (x, z) => spotTool?.clickWorld(x, z);
  wireToolbar();
  updateMetrics(job, cal, depth, cls);
  updateLegend(cls);
  setHints('orbit', false);
  if (pendingDeepLink) { // restore a shared view after the intro state is fully wired
    const st = pendingDeepLink;
    pendingDeepLink = null;
    applyDeepLinkState(viewer, st);
  }
  toast('Flythrough ready — press <b>?</b> for controls, or switch on <b>Probe</b> to read terrain heights.', 'ok');
  // first-run onboarding tour (once per browser; replays from the Help modal)
  if (!tourDone()) setTimeout(() => startTour(), 1200);

  viewer.onProbe = (p) => {
    const pop = $('probe-pop');
    if (!p) { pop.classList.add('hidden'); return; }
    pop.classList.remove('hidden');
    pop.innerHTML = `<div class="pe">${p.elevation.toFixed(1)} m</div>
      <div>slope <b>${p.slopeDeg.toFixed(1)}°</b></div>
      <div class="mono">x ${p.x.toFixed(0)} · z ${p.z.toFixed(0)}</div>`;
    // anchor popup near the click point (tracked globally)
    const r = ($('gl') as HTMLElement).getBoundingClientRect();
    pop.style.left = (lastProbeClient.x - r.left) + 'px';
    pop.style.top = (lastProbeClient.y - r.top) + 'px';
  };

  const clock = { last: performance.now() };
  const mm = { ctx: null as CanvasRenderingContext2D | null, last: 0 };
  const tele = { last: 0 }; // telemetry-strip throttle (~2 Hz)
  let dlLast = '', dlSeen = 0, dlTimer = 0; // deep-link URL writer state
  const loop = (t: number) => {
    const dt = Math.min(0.05, (t - clock.last) / 1000);
    clock.last = t;
    viewer?.tick(dt);
    spotTool?.update(); // project spot-elevation labels each frame
    // mini-map 2D overview inset, throttled to ~10 Hz
    if (t - mm.last > 100 && viewer) {
      mm.last = t;
      if (!mm.ctx) mm.ctx = ($('minimap') as HTMLCanvasElement).getContext('2d');
      if (mm.ctx) viewer.renderMinimap(mm.ctx);
    }
    // telemetry strip, throttled to ~2 Hz
    if (t - tele.last > 500) {
      tele.last = t;
      refreshTelemetry();
    }
    // deep-link: re-serialize the view state at ~2 Hz and, on change,
    // refresh the address bar (debounced so orbit drags don't spam history)
    if (viewer && t - dlSeen > 500) {
      dlSeen = t;
      const s = serializeState(viewer);
      if (s && s !== dlLast) {
        dlLast = s;
        window.clearTimeout(dlTimer);
        dlTimer = window.setTimeout(() => {
          try { history.replaceState(null, '', location.pathname + '?' + s); } catch { /* ignore */ }
        }, 400);
      }
    }
    raf = requestAnimationFrame(loop);
  };
  teleT0 = performance.now(); // mission-clock epoch = scene load
  raf = requestAnimationFrame(loop);
}

// uploads have no SceneData; build a minimal one so the viewer has world size etc.
function synthSceneForUpload(job: Job, _depth: DepthResult): SceneData {
  void _depth;
  return {
    id: 'upload', name: job.name, kind: 'urban', size: SIZE, worldSize: 1200,
    heights: new Float32Array(SIZE * SIZE), labels: new Uint8Array(SIZE * SIZE), rgb: job.rgb,
    baseElevation: 480, relief: 80,
    blurb: 'User upload — relief inferred from image luminance (simulated).',
  };
}

// ------------------------------------------- clean-view presentation mode ---
/**
 * Presentation mode: hides every UI chrome element (toolbar, panels, HUD,
 * minimap, toasts, hints, legend) while the WebGL canvas keeps rendering
 * untouched. A body-level `clean-view` class marks the state; cleanVisible
 * records exactly which elements were visible so restore is exact.
 */
const CLEAN_UI = [
  'toolbar-toggle', 'toolbar', 'metrics',
  'measure-panel', 'places-panel', 'viewshed-panel', 'profile-panel', 'spots-panel',
  'hud', 'minimap', 'toasts', 'hints', 'probe-pop', 'class-legend', 'tour-pop', 'help-modal',
  'spot-layer',
];
const cleanVisible = new Set<string>();

function setCleanView(on: boolean): void {
  const chip = $('clean-exit');
  if (on) {
    cleanVisible.clear();
    for (const id of CLEAN_UI) {
      const el = document.getElementById(id);
      if (el && !el.classList.contains('hidden')) { cleanVisible.add(id); el.style.display = 'none'; }
    }
    document.body.classList.add('clean-view');
    chip.classList.remove('hidden');
    $('btn-clean').classList.add('on');
    // slice is off in clean-view; spot placing exits so toolbar state stays honest
    if (viewer) {
      viewer.clearClipPlane();
      viewer.setSpotPlace(false);
    }
    spotTool?.setActive(false);
    $('btn-slice')?.classList.remove('on');
    $('slice-range')?.classList.add('hidden');
    $('slice-axis')?.classList.add('hidden');
    $('btn-spots')?.classList.remove('on');
  } else {
    for (const id of cleanVisible) {
      const el = document.getElementById(id);
      if (el) el.style.display = '';
    }
    cleanVisible.clear();
    document.body.classList.remove('clean-view');
    chip.classList.add('hidden');
    $('btn-clean').classList.remove('on');
  }
}

// ---------------------------------------------------- shareable deep-link ---
interface DeepLinkState {
  scene: string;
  pos?: [number, number, number];
  tgt?: [number, number, number];
  ex?: number;
  ov?: number;
  contours?: boolean;
  scan?: boolean;
  points?: boolean;
  flood?: number; // -1 / undefined = off, 0..1 = water level fraction
}

/** Deep-link restore state parsed from the URL, consumed once by enterViewerInner. */
let pendingDeepLink: DeepLinkState | null = null;

function parseNums(s: string | null): [number, number, number] | undefined {
  if (!s) return undefined;
  const parts = s.split(',').map(Number);
  if (parts.length !== 3 || parts.some((x) => !Number.isFinite(x))) return undefined;
  return [parts[0], parts[1], parts[2]];
}

/** Read ?scene=&pos=&tgt=&ex=&ov=&lyr=&flood= from the address bar. */
function parseDeepLink(): DeepLinkState | null {
  const q = new URLSearchParams(location.search);
  const scene = q.get('scene');
  if (!scene) return null;
  const lyr = q.get('lyr') ?? '';
  const flood = q.get('flood');
  const st: DeepLinkState = {
    scene,
    pos: parseNums(q.get('pos')),
    tgt: parseNums(q.get('tgt')),
    contours: lyr.includes('c'),
    scan: q.has('lyr') ? lyr.includes('s') : undefined,
    points: lyr.includes('p'),
    flood: flood !== null ? Number(flood) : undefined,
  };
  if (q.has('ex')) { const x = Number(q.get('ex')); if (Number.isFinite(x)) st.ex = x; }
  if (q.has('ov')) { const o = Number(q.get('ov')); if (o >= 0 && o <= 3) st.ov = o; }
  if (st.flood !== undefined && !(st.flood >= 0 && st.flood <= 1)) delete st.flood;
  return st;
}

/** Rebuild the demo job for a scene id (uploads can't be restored — no file to load). */
function jobForSceneId(id: string): Job | null {
  const pick = DEMO_PICKS.find((p) => p.kind === id);
  if (!pick) return null;
  const scene = generateScene(pick.kind);
  return { scene, rgb: scene.rgb, name: scene.name, georeferenced: !pick.rel, formatLabel: pick.fmt };
}

const r1 = (x: number): number => Math.round(x * 10) / 10;

/**
 * Encode the full view state into URL query params:
 * scene id, camera position/target, vertical exaggeration, overlay and layer toggles.
 */
function serializeState(v: TerrainViewer): string | null {
  if (!activeJob) return null;
  const cv = v.getCameraView();
  const q = new URLSearchParams();
  q.set('scene', activeJob.scene?.id ?? 'upload');
  q.set('pos', cv.pos.map(r1).join(','));
  q.set('tgt', cv.tgt.map(r1).join(','));
  q.set('ex', String(Number(($('exagg') as HTMLInputElement).value)));
  q.set('ov', ($('overlay-sel') as HTMLSelectElement).value);
  let lyr = '';
  if ($('btn-contours').classList.contains('on')) lyr += 'c';
  if ($('btn-scan').classList.contains('on')) lyr += 's';
  if ($('btn-points').classList.contains('on')) lyr += 'p';
  const floodOn = $('btn-flood').classList.contains('on');
  if (floodOn) {
    lyr += 'f';
    q.set('flood', String(Math.round(Number(($('flood-range') as HTMLInputElement).value)) / 100));
  }
  q.set('lyr', lyr);
  return q.toString();
}

/** Apply a restored deep-link onto the freshly built viewer (mirrors POI bookmark fly-to). */
function applyDeepLinkState(v: TerrainViewer, st: DeepLinkState): void {
  if (st.ov !== undefined) {
    ($('overlay-sel') as HTMLSelectElement).value = String(st.ov);
    v.setOverlay(st.ov as Overlay);
    $('class-legend').classList.toggle('hidden', st.ov !== 3);
  }
  if (st.ex !== undefined) {
    const x = Math.min(3, Math.max(0.5, st.ex));
    ($('exagg') as HTMLInputElement).value = String(x);
    $('exagg-val').textContent = x.toFixed(1) + '×';
    v.setExaggeration(x);
  }
  if (st.contours) { $('btn-contours').classList.add('on'); v.setContours(true); }
  if (st.scan === false) { $('btn-scan').classList.remove('on'); v.setScan(false); }
  if (st.points) { $('btn-points').classList.add('on'); v.setPoints(true); }
  if (st.flood !== undefined && st.flood >= 0 && st.flood <= 1) {
    $('btn-flood').classList.add('on');
    ($('flood-range') as HTMLInputElement).value = String(Math.round(st.flood * 100));
    $('flood-range').classList.remove('hidden');
    $('flood-val').classList.remove('hidden');
    v.setFlood(st.flood);
    const m = v.floodLevelM();
    $('flood-val').textContent = !Number.isNaN(m) ? '≈ ' + m.toFixed(0) + ' m' : '';
  }
  if (st.pos && st.tgt) {
    v.flyTo(new Vector3(...st.pos), new Vector3(...st.tgt));
    setHints('orbit', false); // flyTo settles in orbit mode; keep the mode buttons in sync
  }
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // fallback for non-secure contexts where the async clipboard API is blocked
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      ta.remove();
      return ok;
    } catch { return false; }
  }
}

async function shareDeepLink(v: TerrainViewer): Promise<void> {
  const qs = serializeState(v);
  if (!qs) { toast('Nothing to share yet — wait for the viewer to load.', 'warn'); return; }
  const url = location.origin + location.pathname + '?' + qs;
  const ok = await copyText(url);
  toast(ok ? 'Share link <b>copied</b> — it restores this exact view.' : 'Could not copy — grab the link from the address bar.', ok ? 'ok' : 'warn');
}

const lastProbeClient = { x: 0, y: 0 };
window.addEventListener('pointerdown', (e) => { lastProbeClient.x = e.clientX; lastProbeClient.y = e.clientY; }, true);

// ------------------------------------------- HUD telemetry strip (mission) ---
/**
 * Tactical telemetry strip above the HUD. Clock epoch (scene load) and the
 * last compass heading come from module state; altitude above terrain comes
 * from viewer.heightAt(); the strip itself refreshes at ~2 Hz from the app
 * tick loop in enterViewerInner. The strip hides with the rest of the HUD
 * chrome in presentation mode (body.clean-view).
 */
let teleT0 = 0;
let teleHeadingDeg = 0;

/** 16-wind compass point for a clockwise-from-north heading in degrees. */
function compassPoint(deg: number): string {
  const pts = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  return pts[((Math.round(deg / 22.5) % 16) + 16) % 16];
}

const pad2 = (n: number): string => String(n).padStart(2, '0');

function refreshTelemetry(): void {
  if (!viewer) return;
  const s = Math.max(0, Math.floor((performance.now() - teleT0) / 1000));
  $('tele-clock').textContent =
    `T+${pad2(Math.floor(s / 3600))}:${pad2(Math.floor(s / 60) % 60)}:${pad2(s % 60)}`;
  const cv = viewer.getCameraView();
  const agl = cv.pos[1] - viewer.heightAt(cv.pos[0], cv.pos[2]);
  $('tele-agl').textContent = agl.toFixed(0);
  $('tele-hdg').textContent = teleHeadingDeg.toFixed(0);
  $('tele-hdg-pt').textContent = compassPoint(teleHeadingDeg);
}

function setHints(m: ViewMode, announce = true): void {
  $('hints').textContent = HINTS[m];
  $('hud-mode').textContent = m.toUpperCase();
  document.querySelectorAll('#toolbar [data-mode]').forEach(b =>
    b.classList.toggle('active', (b as HTMLElement).dataset.mode === m));
  if (announce) toast(`Mode: <b>${m === 'fpv' ? 'walk' : m}</b> — ${HINTS[m]}`, 'info', 2600);
}

/** Render the finished-measurements list into the measure panel. */
function renderMeasureList(v: TerrainViewer): void {
  const list = $('measure-list');
  const items = v.measureList();
  list.innerHTML = items.length === 0
    ? '<div class="mnote">No finished measurements yet.</div>'
    : items.map((m, i) =>
        `<div class="mrow"><span><span class="cdot" style="background:${m.color}"></span>${m.pts} pts</span>` +
        `<b>${m.total}</b><button class="mini-del" data-i="${i}" title="Delete" aria-label="Delete measurement ${i + 1}">${uiIcon('x')}</button></div>`).join('');
  list.querySelectorAll('.mini-del').forEach(b =>
    (b as HTMLElement).onclick = () => { v.deleteMeasure(Number((b as HTMLElement).dataset.i)); });
}

/** Custom camera bookmarks, persisted across sessions. */
const PLACES_KEY = 'aero3d-bookmarks';

function loadPlaces(): Bookmark[] {
  try {
    const raw = localStorage.getItem(PLACES_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr.filter(p =>
      p && typeof p.name === 'string' && Array.isArray(p.pos) && Array.isArray(p.tgt)) : [];
  } catch { return []; }
}

function savePlace(v: TerrainViewer): void {
  const mine = loadPlaces();
  const cv = v.getCameraView();
  mine.push({ name: 'View ' + (mine.length + 1), pos: cv.pos, tgt: cv.tgt });
  localStorage.setItem(PLACES_KEY, JSON.stringify(mine));
}

/** Render POI presets + saved views into the places panel. */
function renderPlaces(v: TerrainViewer): void {
  const list = $('places-list');
  const rows: { b: Bookmark; custom: number }[] = [
    ...v.bookmarkPresets().map(b => ({ b, custom: -1 })),
    ...loadPlaces().map((b, i) => ({ b, custom: i })),
  ];
  list.innerHTML = rows.map((r, i) =>
    `<div class="mrow prow"><button class="pfly" data-i="${i}">${r.b.name}</button>` +
    (r.custom >= 0 ? `<button class="mini-del" data-del="${r.custom}" title="Delete" aria-label="Delete saved view ${r.b.name}">${uiIcon('x')}</button>` : '<span></span>') +
    '</div>').join('');
  list.querySelectorAll('.pfly').forEach(b =>
    (b as HTMLElement).onclick = () => {
      const r = rows[Number((b as HTMLElement).dataset.i)];
      v.flyTo(new Vector3(...r.b.pos), new Vector3(...r.b.tgt));
    });
  list.querySelectorAll('.mini-del').forEach(b =>
    (b as HTMLElement).onclick = () => {
      const mine = loadPlaces();
      mine.splice(Number((b as HTMLElement).dataset.del), 1);
      localStorage.setItem(PLACES_KEY, JSON.stringify(mine));
      renderPlaces(v);
    });
}

/** First-run onboarding tour wiring: replay button lives in the Help modal. */
function wireTourReplay(): void {
  const b = document.getElementById('btn-tour-replay');
  if (b) b.onclick = () => { $('help-modal').classList.add('hidden'); startTour(true); };
}

// ------------------------------------------------------ onboarding tour ---
/** First-run tooltip walkthrough. Auto-starts once per browser (TOUR_KEY). */
const TOUR_KEY = 'aero3d-tour-done';

function tourDone(): boolean {
  try { return !!localStorage.getItem(TOUR_KEY); } catch { return true; }
}

/** Find the toolbar cluster whose .tb-label matches; assigns a stable id on first use. */
function tbGroupSel(label: string): string {
  const groups = Array.from(document.querySelectorAll('#toolbar .tb-group'));
  const g = groups.find(el => el.querySelector('.tb-label')?.textContent?.trim() === label);
  if (!g) return '#toolbar';
  if (!g.id) g.id = 'tb-group-' + label.toLowerCase();
  return '#' + g.id;
}

interface TourStep {
  sel: () => string;
  title: string;
  body: string;
  hint: string;
  onEnter?: () => void;
}

const TOUR_STEPS: TourStep[] = [
  {
    sel: () => tbGroupSel('View'),
    title: '1 · View — camera modes',
    body: `Switch <b>Orbit</b>, <b>Drone</b> and <b>Walk</b> — or press <kbd>V</kbd>. <b>${uiIcon('clapper')} Auto-fly</b> runs the cinematic drone path, <b>${uiIcon('bookmark')} Places</b> flies to saved viewpoints, <b>?</b> reopens the help panel.`,
    hint: 'Step 1 of 6 — View: press V to switch Orbit → Drone → Walk.',
  },
  {
    sel: () => tbGroupSel('Analyze'),
    title: '2 · Analyze — survey tools',
    body: `<b>${uiIcon('pin')} Probe</b> reads height &amp; slope, <b>${uiIcon('ruler')} Measure</b> drapes distances over the terrain, <b>${uiIcon('mountain')} Profile</b> opens a 2D elevation chart of a traverse line, <b>${uiIcon('eye')} Viewshed</b> tints visible vs occluded terrain from an observer, <b>${uiIcon('layers')}</b> adds contours, <b>${uiIcon('compare')} Compare</b> swipes optical RGB ↔ DSM.`,
    hint: 'Step 2 of 6 — Analyze: probe, measure, contours, compare.',
  },
  {
    sel: () => '#overlay-sel',
    title: '3 · Display — overlay layers',
    body: `Swap the surface layer: <b>Optical</b>, <b>Slope</b>, <b>Hypsometric tint</b> or <b>Land-cover classes</b> (with the coverage legend). ${uiIcon('sun')} sliders drive sun shadows · ${uiIcon('move')} exaggerates relief.`,
    hint: 'Step 3 of 6 — Display: 4 surface layers + sun &amp; exaggeration sliders.',
  },
  {
    sel: () => '#btn-probe',
    title: '4 · Height probe — click the terrain',
    body: 'Probe is switched <b>on</b> for you — <b>click anywhere on the terrain</b> and a popup reads the elevation in metres plus the local slope.',
    hint: 'Step 4 of 6 — Probe is ON: click the terrain to read height &amp; slope.',
    onEnter: () => { const bp = $('btn-probe'); if (!bp.classList.contains('on')) bp.click(); },
  },
  {
    sel: () => '#toolbar [data-mode="drone"]',
    title: '5 · Drone-mode keys',
    body: 'In Drone mode: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly · <kbd>R</kbd>/<kbd>F</kbd> up &amp; down · drag to look · wheel = speed · <kbd>Shift</kbd> = boost.',
    hint: 'Step 5 of 6 — Drone keys: WASD + R/F + drag + Shift.',
  },
  {
    sel: () => tbGroupSel('Export'),
    title: '6 · Export — take it with you',
    body: `<b>${uiIcon('download')} DSM</b> downloads the elevation map as a PNG · <b>${uiIcon('file')} Report</b> saves the full survey report with metrics · <b>${uiIcon('camera')} Snapshot</b> saves the current 3D view as a PNG (Alt+P) · <b>${uiIcon('arrow-left')} New</b> returns to scene select.`,
    hint: 'Step 6 of 6 — Export the DSM PNG, the survey report and the 3D snapshot.',
  },
];

let tourIdx = -1;
let tourPop: HTMLElement | null = null;
let tourTarget: Element | null = null;
let tourEscHandler: ((e: KeyboardEvent) => void) | null = null;

function tourShow(i: number): void {
  if (!tourPop) return;
  tourIdx = i;
  tourTarget?.classList.remove('tour-target');
  const step = TOUR_STEPS[i];
  const el = document.querySelector(step.sel());
  if (!el) { tourNext(); return; } // target missing (e.g. collapsed toolbar) — advance
  tourTarget = el;
  el.classList.add('tour-target');
  step.onEnter?.();
  (tourPop.querySelector('.tour-title') as HTMLElement).innerHTML =
    `${step.title} <span class="tour-step-n">${i + 1} / ${TOUR_STEPS.length}</span>`;
  (tourPop.querySelector('.tour-body') as HTMLElement).innerHTML = step.body;
  (tourPop.querySelector('.tour-back') as HTMLButtonElement).disabled = i === 0;
  (tourPop.querySelector('.tour-next') as HTMLButtonElement).innerHTML =
    i === TOUR_STEPS.length - 1 ? 'Finish ' + uiIcon('check') : 'Next ' + uiIcon('arrow-right');
  tourPlace();
  toast(step.hint, 'info', 4600);
}

function tourPlace(): void {
  if (!tourPop || !tourTarget) return;
  const r = tourTarget.getBoundingClientRect();
  tourPop.style.visibility = 'hidden';
  tourPop.style.left = '0px';
  tourPop.style.top = '0px';
  const pw = tourPop.offsetWidth, ph = tourPop.offsetHeight;
  const x = Math.min(Math.max(8, r.left + r.width / 2 - pw / 2), Math.max(8, window.innerWidth - pw - 8));
  let y = r.bottom + 12;
  if (y + ph > window.innerHeight - 8) y = Math.max(8, r.top - ph - 12);
  tourPop.style.left = x + 'px';
  tourPop.style.top = y + 'px';
  tourPop.style.visibility = 'visible';
}

function tourNext(): void {
  if (tourIdx >= TOUR_STEPS.length - 1) tourEnd(true);
  else tourShow(tourIdx + 1);
}

/** End the tour. markDone also suppresses the first-run auto-start. */
function tourEnd(markDone: boolean): void {
  if (markDone) { try { localStorage.setItem(TOUR_KEY, '1'); } catch { /* storage unavailable */ } }
  tourTarget?.classList.remove('tour-target');
  tourTarget = null;
  tourIdx = -1;
  tourPop?.remove();
  tourPop = null;
  if (tourEscHandler) { window.removeEventListener('keydown', tourEscHandler); tourEscHandler = null; }
}

/** Start the walkthrough. force=true replays it even if the first-run flag is set. */
function startTour(force = false): void {
  if (!force && tourDone()) return;
  if (document.body.classList.contains('clean-view')) return; // don't pop a tour over presentation mode
  if ($('screen-viewer').classList.contains('hidden')) return; // tour only makes sense in the viewer
  if (!tourPop) {
    tourPop = document.createElement('div');
    tourPop.id = 'tour-pop';
    tourPop.className = 'tour-pop';
    tourPop.innerHTML = `
      <h4 class="tour-title"></h4>
      <p class="tour-body"></p>
      <div class="tour-actions">
        <button class="tb-btn ghost tour-skip">Skip</button>
        <span class="spacer"></span>
        <button class="tb-btn tour-back">${uiIcon('arrow-left')} Back</button>
        <button class="tb-btn active tour-next">Next ${uiIcon('arrow-right')}</button>
      </div>`;
    document.body.appendChild(tourPop);
    (tourPop.querySelector('.tour-back') as HTMLButtonElement).onclick = () => tourShow(Math.max(0, tourIdx - 1));
    (tourPop.querySelector('.tour-next') as HTMLButtonElement).onclick = () => tourNext();
    (tourPop.querySelector('.tour-skip') as HTMLButtonElement).onclick = () => tourEnd(true);
    window.addEventListener('resize', () => { if (tourIdx >= 0) tourPlace(); });
  }
  if (tourEscHandler) window.removeEventListener('keydown', tourEscHandler);
  tourEscHandler = (e: KeyboardEvent) => { if (e.key === 'Escape') tourEnd(true); };
  window.addEventListener('keydown', tourEscHandler);
  tourShow(0);
}

// ---- Spot-elevation panel: marker list with land-cover class + fly-to ----
function renderSpots(markers: SpotMarker[]): void {
  const stats = $('spots-stats');
  const list = $('spots-list');
  const s = spotStats(markers);
  stats.textContent = s
    ? `${s.count} marker${s.count === 1 ? '' : 's'} · ${s.minH.toFixed(1)} – ${s.maxH.toFixed(1)} m (Δ ${s.rangeM.toFixed(1)} m)`
    : 'No markers yet.';
  list.innerHTML = '';
  if (!viewer) return;
  const ex = viewer.getExaggeration();
  const worldSize = activeJob?.scene?.worldSize ?? 1200;
  for (const m of markers) {
    const li = document.createElement('li');
    li.title = 'Fly to this marker';
    const fx = ((m.x + worldSize / 2) / worldSize) * (SIZE - 1);
    const fz = ((m.z + worldSize / 2) / worldSize) * (SIZE - 1);
    const clsName = activeCls ? (LCLASS_NAMES[classAt(activeCls.labels, activeCls.size, fx, fz)] ?? '—') : '';
    li.innerHTML = `<span>${fmtSpot(m)}</span><span class="cls">${clsName}</span>`;
    li.onclick = () => {
      const surfY = m.h * ex;
      viewer!.flyTo(new Vector3(m.x, surfY + 180, m.z + 1), new Vector3(m.x, surfY, m.z));
    };
    list.appendChild(li);
  }
}

// --------------------------------------------- render quality presets ---
/**
 * Render-quality presets (Display toolbar cluster). The effective
 * Viewer.setQuality hook lives here in the UI area (viewer.ts is outside this
 * specialist's file scope): it adjusts renderer pixelRatio, the sun shadow-map
 * size, and the UI-layer film FX (vignette + grain) intensity. If a future
 * TerrainViewer revision provides its own setQuality() method it takes
 * precedence via the duck-type check below.
 */
export type QualityLevel = 'performance' | 'balanced' | 'cinematic';

interface QualityPreset {
  pixelRatioCap: number; // cap applied over window.devicePixelRatio
  shadowSize: number;    // sun shadow-map texel resolution
  vignette: number;      // CSS vignette overlay opacity (0 = off)
  grain: number;         // film-grain overlay opacity (0 = off)
  title: string;         // toolbar tooltip
}

const QUALITY_PRESETS: Record<QualityLevel, QualityPreset> = {
  performance: {
    pixelRatioCap: 1, shadowSize: 512, vignette: 0, grain: 0,
    title: 'Performance — pixel ratio ≤1 · 512px shadows · no film FX. Fastest on weak GPUs / judges\u2019 laptops.',
  },
  balanced: {
    pixelRatioCap: 1.5, shadowSize: 1024, vignette: 0.16, grain: 0.05,
    title: 'Balanced — pixel ratio ≤1.5 · 1024px shadows · light film FX. The default.',
  },
  cinematic: {
    pixelRatioCap: 2, shadowSize: 2048, vignette: 0.32, grain: 0.09,
    title: 'Cinematic — pixel ratio ≤2 · 2048px shadows · full film FX. Best visuals, needs a strong GPU.',
  },
};
const QUALITY_ORDER: QualityLevel[] = ['performance', 'balanced', 'cinematic'];
const QUALITY_ICON: Record<QualityLevel, string> = { performance: 'zap', balanced: 'balance', cinematic: 'video' };
const QUALITY_NAME: Record<QualityLevel, string> = { performance: 'Performance', balanced: 'Balanced', cinematic: 'Cinematic' };
const QUALITY_KEY = 'aero3d-quality';

function loadQuality(): QualityLevel {
  try {
    const q = localStorage.getItem(QUALITY_KEY);
    if (q === 'performance' || q === 'balanced' || q === 'cinematic') return q;
  } catch { /* storage unavailable */ }
  return 'balanced';
}

/** Apply a quality preset to the viewer: renderer pixel ratio, shadow-map size, film FX. */
function applyQuality(v: TerrainViewer, level: QualityLevel): void {
  const p = QUALITY_PRESETS[level];
  // Prefer a native viewer hook when the viewer provides one.
  const native = (v as unknown as { setQuality?: (l: QualityLevel) => void }).setQuality;
  if (typeof native === 'function') {
    native.call(v, level);
  } else {
    const inner = v as unknown as {
      renderer?: { setPixelRatio(n: number): void };
      _sun?: { shadow: { mapSize: { set(x: number, y: number): void }; map?: { dispose(): void } | null } };
    };
    // setPixelRatio re-allocates the drawing buffer internally (three r170)
    inner.renderer?.setPixelRatio(Math.min(p.pixelRatioCap, window.devicePixelRatio || 1));
    const shadow = inner._sun?.shadow;
    if (shadow) {
      shadow.mapSize.set(p.shadowSize, p.shadowSize);
      // force three.js to re-allocate the shadow map at the new resolution
      if (shadow.map) { shadow.map.dispose(); shadow.map = null; }
    }
  }
  // UI-layer film FX (vignette + grain) — intensity scales with the preset.
  const vig = document.getElementById('fx-vignette');
  const grn = document.getElementById('fx-grain');
  if (vig) vig.style.opacity = String(p.vignette);
  if (grn) grn.style.opacity = String(p.grain);
  document.querySelectorAll('#q-presets [data-q]').forEach(b =>
    (b as HTMLElement).classList.toggle('active', (b as HTMLElement).dataset.q === level));
}

/** Persist the choice, then apply it (toolbar click path). */
function setQualityLevel(v: TerrainViewer, level: QualityLevel, announce = true): void {
  try { localStorage.setItem(QUALITY_KEY, level); } catch { /* storage unavailable */ }
  applyQuality(v, level);
  if (announce) toast(`Render quality set to <b>${QUALITY_NAME[level]}</b> — saved for this browser.`, 'info', 2200);
}

/** Segmented Performance / Balanced / Cinematic control in the Display cluster. */
function wireQualityPresets(v: TerrainViewer): void {
  document.getElementById('q-presets')?.remove(); // idempotent across scene loads
  const dispGroup = document.querySelector(tbGroupSel('Display'));
  if (!dispGroup) return;
  const wrap = document.createElement('span');
  wrap.id = 'q-presets';
  wrap.className = 'q-seg';
  wrap.setAttribute('role', 'group');
  wrap.setAttribute('aria-label', 'Render quality preset');
  const qlbl = document.createElement('span');
  qlbl.className = 'q-lbl';
  qlbl.textContent = 'Quality';
  wrap.appendChild(qlbl);
  for (const level of QUALITY_ORDER) {
    const p = QUALITY_PRESETS[level];
    const b = document.createElement('button');
    b.className = 'tb-btn q-btn';
    b.dataset.q = level;
    b.title = p.title;
    b.setAttribute('aria-label', `Quality preset: ${QUALITY_NAME[level]}`);
    b.innerHTML = `<span class="ic" aria-hidden="true">${uiIcon(QUALITY_ICON[level])}</span><span class="lbl">${QUALITY_NAME[level]}</span>`;
    b.onclick = () => setQualityLevel(v, level);
    wrap.appendChild(b);
  }
  dispGroup.appendChild(wrap);
  applyQuality(v, loadQuality()); // sync the active button with the saved choice
}

function wireToolbar(): void {
  if (!viewer) return;
  const v = viewer;
  // collapsible toolbar on small screens
  const tbtn = $('toolbar-toggle');
  tbtn.onclick = () => {
    const tb = $('toolbar');
    const collapsed = tb.classList.toggle('collapsed');
    tbtn.setAttribute('aria-expanded', String(!collapsed));
  };
  // reset one-shot toggle buttons for the fresh viewer session
  for (const id of ['btn-points', 'btn-flood', 'btn-places', 'btn-autoorbit', 'btn-viewshed', 'btn-profile', 'btn-compare', 'btn-change', 'btn-spots', 'btn-slice']) $(id).classList.remove('on');
  $('flood-range').classList.add('hidden');
  $('flood-val').classList.add('hidden');
  $('slice-range').classList.add('hidden');
  $('slice-axis').classList.add('hidden');
  $('places-panel').classList.add('hidden');
  $('viewshed-panel').classList.add('hidden');
  $('spots-panel').classList.add('hidden');
  $('profile-panel').classList.add('hidden');
  document.querySelectorAll('#toolbar [data-mode]').forEach(b =>
    (b as HTMLElement).onclick = () => { v.setMode((b as HTMLElement).dataset.mode as ViewMode); setHints(v.mode); });
  ($('overlay-sel') as HTMLSelectElement).onchange = (e) => {
    const o = Number((e.target as HTMLSelectElement).value) as Overlay;
    v.setOverlay(o);
    $('class-legend').classList.toggle('hidden', o !== 3);
    toast(`Surface layer: <b>${(e.target as HTMLSelectElement).selectedOptions[0].textContent}</b>`, 'info', 2000);
  };
  ($('exagg') as HTMLInputElement).oninput = (e) => {
    const x = Number((e.target as HTMLInputElement).value);
    $('exagg-val').textContent = x.toFixed(1) + '×';
    v.setExaggeration(x);
    spotTool?.setExaggeration(x); // keep spot labels glued to the rendered surface
  };
  const bc = $('btn-contours');
  bc.onclick = () => { const on = bc.classList.toggle('on'); v.setContours(on); toast(on ? 'Contour lines on' : 'Contour lines off', 'info', 1800); };
  const bcmp = $('btn-compare');
  const cmpRange = $('compare-range') as HTMLInputElement;
  const bchg = $('btn-change');
  bcmp.onclick = () => {
    const on = bcmp.classList.toggle('on');
    if (on && v.isChangeDetectionOn()) { // mutually exclusive with change-detection compare
      bchg.classList.remove('on'); v.setChangeDetection(false);
      changeResult = null; refreshMetrics();
    }
    cmpRange.classList.toggle('hidden', !on && !v.isChangeDetectionOn());
    v.setCompare(on ? Number(cmpRange.value) / 100 : -1);
    if (on) toast('Drag the slider to compare optical RGB with the DSM.', 'info', 2600);
  };
  bchg.onclick = () => {
    if (!activeCal) return;
    const on = bchg.classList.toggle('on');
    if (on) { // change detection: baseline DSM ↔ post-monsoon DSM via the compare split-shader
      if (bcmp.classList.contains('on')) bcmp.classList.remove('on');
      const r = synthesizePostEventDSM(activeCal.absolute, SIZE);
      v.setChangeTexture(packChangeTexture(r), SIZE);
      v.setChangeDetection(true);
      cmpRange.classList.remove('hidden');
      v.setCompare(Number(cmpRange.value) / 100);
      changeResult = r; refreshMetrics();
      toast('<b>Change detection</b> — swipe the slider: left is the pre-monsoon baseline, right the post-monsoon DSM.', 'info', 3000);
    } else {
      v.setChangeDetection(false);
      cmpRange.classList.add('hidden');
      changeResult = null; refreshMetrics();
      toast('Change detection off', 'info', 1600);
    }
  };
  cmpRange.oninput = () => v.setCompare(Number(cmpRange.value) / 100);
  const saz = $('sun-az') as HTMLInputElement, sel2 = $('sun-el') as HTMLInputElement;
  const updSun = () => {
    v.setSun(Number(saz.value), Number(sel2.value));
    $('sun-val').textContent = `${saz.value}°/${sel2.value}°`;
  };
  saz.oninput = updSun; sel2.oninput = updSun;
  const bf = $('btn-cine');
  bf.onclick = () => {
    if (v.mode !== 'drone') { v.setMode('drone'); setHints('drone'); }
    const on = !v.isCinematic();
    v.setCinematic(on); bf.classList.toggle('on', on);
    toast(on ? 'Cinematic auto-fly started' : 'Cinematic auto-fly stopped', 'info', 2200);
  };
  // auto-orbit kiosk mode: toolbar toggle <-> Viewer.setAutoOrbit; any user
  // input pauses the spin and the viewer reports back via onAutoOrbit so the
  // button stays in sync (also after internal mode changes like V cycling)
  const bao = $('btn-autoorbit');
  bao.onclick = () => {
    const on = !v.isAutoOrbitOn();
    v.setAutoOrbit(on);
    bao.classList.toggle('on', on);
    if (on) setHints('orbit', false); // setAutoOrbit(true) switched to orbit mode; sync mode buttons silently
    toast(on ? 'Auto-orbit on — hands-free spin; any input pauses it.' : 'Auto-orbit off', 'info', 2200);
  };
  v.onAutoOrbit = (on) => bao.classList.toggle('on', on);
  const bp = $('btn-probe');
  bp.onclick = () => {
    const on = bp.classList.toggle('on');
    v.enableProbe(on);
    if (on && v.isMeasureOn()) { v.setMeasure(false); $('btn-measure').classList.remove('on'); $('measure-panel').classList.add('hidden'); resetVolumeMode(); }
    toast(on ? 'Probe on — click the terrain to read height and slope.' : 'Probe off', 'info', 2400);
  };
  const bsc = $('btn-scan');
  bsc.onclick = () => { const on = bsc.classList.toggle('on'); v.setScan(on); toast(on ? 'Survey scan wave and grid on' : 'Scan overlay off', 'info', 1800); };
  const bm = $('btn-measure');
  const bmv = $('btn-measure-volume');
  const resetVolumeMode = () => {
    if (bmv.classList.contains('on')) {
      bmv.classList.remove('on');
      v.setMeasureMode('distance');
      $('measure-hint').textContent = 'Click terrain to add points · double-click to finish · Esc cancels';
    }
  };
  bm.onclick = () => {
    const on = !v.isMeasureOn();
    v.setMeasure(on);
    bm.classList.toggle('on', on);
    $('measure-panel').classList.toggle('hidden', !on);
    if (!on) resetVolumeMode();
    if (on && bp.classList.contains('on')) { bp.classList.remove('on'); v.enableProbe(false); }
    toast(on ? 'Measure on — click the terrain to add points, double-click to finish.' : 'Measure off', 'info', 2600);
  };
  bmv.onclick = () => {
    const on = !bmv.classList.contains('on');
    bmv.classList.toggle('on', on);
    v.setMeasureMode(on ? 'volume' : 'distance');
    $('measure-hint').textContent = on
      ? 'Click terrain to add polygon corners · double-click to finish · Esc cancels'
      : 'Click terrain to add points · double-click to finish · Esc cancels';
    if (on) {
      if (!v.isMeasureOn()) {
        v.setMeasure(true);
        bm.classList.add('on');
        $('measure-panel').classList.remove('hidden');
      }
      if (bp.classList.contains('on')) { bp.classList.remove('on'); v.enableProbe(false); }
    }
    toast(on ? 'Volume on — draw a polygon, then finish to get cut and fill volumes in m³.' : 'Volume off — distance mode', 'info', 2600);
  };
  ($('btn-measure-finish') as HTMLElement).onclick = () => { v.finishMeasure(); };
  ($('btn-measure-clear') as HTMLElement).onclick = () => { v.clearMeasures(); toast('Measurements cleared', 'info', 1600); };

  // ---- Viewshed / line-of-sight tool ----
  const bvs = $('btn-viewshed');
  const vspanel = $('viewshed-panel');
  const vstower = $('viewshed-tower') as HTMLInputElement;
  const viewshedOff = () => {
    v.clearViewshed();
    bvs.classList.remove('on');
    vspanel.classList.add('hidden');
  };
  bvs.onclick = () => {
    const on = !v.isViewshedPlaceMode() && !v.isViewshedActive();
    if (on) {
      // mutual exclusion with the other terrain-click tools
      if (v.isMeasureOn()) { v.setMeasure(false); $('btn-measure').classList.remove('on'); $('measure-panel').classList.add('hidden'); resetVolumeMode(); profileArmed = false; bpf.classList.remove('on'); }
      if (bp.classList.contains('on')) { bp.classList.remove('on'); v.enableProbe(false); }
      v.setViewshedPlace(true);
      bvs.classList.add('on');
      vspanel.classList.remove('hidden');
      toast('Viewshed — click the terrain to place the observer.', 'info', 2600);
    } else viewshedOff();
  };
  vstower.oninput = () => {
    $('viewshed-tower-val').textContent = `${vstower.value} m`;
    v.setViewshedTower(Number(vstower.value));
  };
  ($('btn-viewshed-clear') as HTMLElement).onclick = () => viewshedOff();
  v.onViewshed = (s) => {
    const stats = $('viewshed-stats');
    if (!s) { stats.textContent = 'Click the terrain to place the observer…'; return; }
    stats.innerHTML = `Intervisible: <b>${s.pct.toFixed(1)}%</b> of terrain · observer ground ${s.groundM.toFixed(1)} m + ${s.towerH} m mast`;
  };

  // ---- Elevation-profile traverse tool (reuses the measure click flow) ----
  const bpf = $('btn-profile');
  let profileArmed = false;
  let profileSeen = 0; // finished-measurement count seen while armed
  let profileResult: ProfileResult | null = null;
  const PROFILE_W = 464, PROFILE_H = 200;
  const closeProfilePanel = () => {
    $('profile-panel').classList.add('hidden');
    v.clearProfileMarker();
    profileResult = null;
  };
  const scrubProfile = (i: number) => {
    const r = profileResult; if (!r || r.samples.length === 0) return;
    const n = r.samples.length;
    i = Math.max(0, Math.min(n - 1, i));
    const { x, y } = profileCursorXY(r, i, PROFILE_W, PROFILE_H);
    const cursor = document.querySelector('#profile-cursor');
    if (cursor) {
      cursor.setAttribute('transform', `translate(${x},0)`);
      cursor.removeAttribute('visibility');
      const dot = cursor.querySelector('circle');
      if (dot) dot.setAttribute('cy', String(y));
    }
    const s = r.samples[i];
    if (s) {
      v.setProfileMarker(s.x, s.z);
      ($('profile-scrub') as HTMLInputElement).value = String(i);
      $('profile-read').textContent = `${Math.round(s.dist)} m along · ${s.h.toFixed(1)} m elevation`;
    }
  };
  const openProfileChart = () => {
    const pts = v.lastMeasurePts();
    const dsm = v.getDSM();
    if (!pts || pts.length < 2 || !dsm) { toast('Profile needs a finished traverse line.', 'err'); return; }
    const r = computeProfile(pts, dsm.heights, dsm.gsdMeters);
    if (r.samples.length < 2) { toast('Not enough samples for a profile.', 'err'); return; }
    profileResult = r;
    $('profile-stats').innerHTML = `<b>${fmtProfile(r)}</b>`;
    const chart = $('profile-chart');
    chart.innerHTML = profileChartSVG(r, { width: PROFILE_W, height: PROFILE_H });
    const scrub = $('profile-scrub') as HTMLInputElement;
    scrub.max = String(r.samples.length - 1);
    scrub.value = '0';
    scrub.oninput = () => scrubProfile(Number(scrub.value));
    const svg = chart.querySelector('svg');
    if (svg) {
      svg.onmousemove = (e) => {
        const rect = svg.getBoundingClientRect();
        const { padL, padR } = PROFILE_CHART;
        const ux = ((e.clientX - rect.left) / Math.max(1, rect.width)) * PROFILE_W;
        const t = Math.max(0, Math.min(1, (ux - padL) / Math.max(1, PROFILE_W - padL - padR)));
        scrubProfile(Math.round(t * (r.samples.length - 1)));
      };
    }
    $('profile-panel').classList.remove('hidden');
    scrubProfile(0);
    toast('Elevation profile chart open — scrub the slider or hover the chart.', 'ok', 2400);
  };
  bpf.onclick = () => {
    const on = !profileArmed;
    profileArmed = on;
    bpf.classList.toggle('on', on);
    if (on) {
      if (bp.classList.contains('on')) { bp.classList.remove('on'); v.enableProbe(false); }
      if (bvs.classList.contains('on') || v.isViewshedActive()) viewshedOff();
      resetVolumeMode();
      v.setMeasureMode('distance');
      if (!v.isMeasureOn()) { v.setMeasure(true); bm.classList.add('on'); $('measure-panel').classList.remove('hidden'); }
      $('measure-hint').textContent = 'Profile traverse: click terrain for points · double-click to finish · chart opens on finish';
      profileSeen = v.measureList().length;
      toast('Profile on — draw a traverse line; the profile chart opens when you finish.', 'info', 2800);
    } else {
      closeProfilePanel();
      $('measure-hint').textContent = 'Click terrain to add points · double-click to finish · Esc cancels';
    }
  };
  ($('btn-profile-close') as HTMLElement).onclick = () => closeProfilePanel();
  v.onMeasure = () => {
    renderMeasureList(v);
    if (profileArmed) {
      const n = v.measureList().length;
      if (n > profileSeen) { profileSeen = n; openProfileChart(); }
      else profileSeen = n; // deletions / clears stay in sync
    }
  };
  const bpts = $('btn-points');
  bpts.onclick = () => { const on = bpts.classList.toggle('on'); v.setPoints(on); toast(on ? 'LiDAR-style point cloud on' : 'Point cloud off', 'info', 1800); };

  // ---- Spot elevations: Pix4D-style field-verification markers ----
  const bspots = $('btn-spots');
  const spotsPanel = $('spots-panel');
  const spotsOff = () => {
    spotTool?.setActive(false);
    v.setSpotPlace(false);
    bspots.classList.remove('on');
    spotsPanel.classList.add('hidden');
  };
  bspots.onclick = () => {
    const on = !bspots.classList.contains('on');
    if (on) {
      // mutual exclusion with the other terrain-click tools
      if (v.isMeasureOn()) { v.setMeasure(false); $('btn-measure').classList.remove('on'); $('measure-panel').classList.add('hidden'); resetVolumeMode(); profileArmed = false; bpf.classList.remove('on'); }
      if (bp.classList.contains('on')) { bp.classList.remove('on'); v.enableProbe(false); }
      if (v.isViewshedPlaceMode() || v.isViewshedActive()) viewshedOff();
      bspots.classList.add('on');
      spotsPanel.classList.remove('hidden');
      v.setSpotPlace(true);
      spotTool?.setActive(true);
      toast('Spots on — click the terrain to drop spot-height markers, or use Smart grid. Esc clears.', 'info', 2800);
    } else {
      spotsOff();
      toast('Spots off', 'info', 1600);
    }
  };
  ($('btn-spot-grid') as HTMLElement).onclick = () => {
    const n = spotTool?.generateSmartGrid().length ?? 0;
    toast(n > 0 ? `Smart grid placed — <b>${n}</b> spot heights.` : 'Smart grid needs a loaded scene.', 'info', 2200);
  };
  ($('btn-spot-clear') as HTMLElement).onclick = () => { spotTool?.clear(); toast('Spot markers cleared', 'info', 1600); };

  // ---- Elevation slice: clip-plane cross-section inspection ----
  const bslice = $('btn-slice');
  const sliceRange = $('slice-range') as HTMLInputElement;
  const sliceAxis = $('slice-axis') as HTMLButtonElement;
  const sliceAxisVal = () => (sliceAxis.textContent === 'Z' ? 'y' as const : 'x' as const);
  const sliceOff = () => {
    v.clearClipPlane();
    bslice.classList.remove('on');
    sliceRange.classList.add('hidden');
    sliceAxis.classList.add('hidden');
  };
  bslice.onclick = () => {
    const on = !bslice.classList.contains('on');
    if (on) {
      bslice.classList.add('on');
      sliceRange.classList.remove('hidden');
      sliceAxis.classList.remove('hidden');
      v.setClipPlane(sliceAxisVal(), Number(sliceRange.value) / 100);
      toast('Slice on — drag the slider to move the cut plane; X/Z toggles the axis.', 'info', 2600);
    } else {
      sliceOff();
      toast('Slice off', 'info', 1600);
    }
  };
  sliceRange.oninput = () => v.setClipPlane(sliceAxisVal(), Number(sliceRange.value) / 100);
  sliceAxis.onclick = () => {
    sliceAxis.textContent = sliceAxis.textContent === 'X' ? 'Z' : 'X';
    v.setClipPlane(sliceAxisVal(), Number(sliceRange.value) / 100);
  };
  const bfl = $('btn-flood');
  const frng = $('flood-range') as HTMLInputElement;
  const updFlood = () => {
    const on = bfl.classList.contains('on');
    v.setFlood(on ? Number(frng.value) / 100 : -1);
    const m = v.floodLevelM();
    $('flood-val').textContent = on && !Number.isNaN(m) ? '≈ ' + m.toFixed(0) + ' m' : '';
  };
  bfl.onclick = () => {
    const on = bfl.classList.toggle('on');
    frng.classList.toggle('hidden', !on);
    $('flood-val').classList.toggle('hidden', !on);
    updFlood();
    toast(on ? 'Flood simulation on — drag the slider to raise the water level.' : 'Flood simulation off', 'info', 2600);
  };
  frng.oninput = updFlood;
  // ---- NVG night-vision + FLIR thermal sensor modes (Display cluster cross-hook) ----
  // The buttons are built here (not in index.html) so this wiring needs no markup
  // change; removing stale copies first keeps wireToolbar() idempotent across scenes.
  for (const id of ['btn-nvg', 'btn-thermal']) $(id)?.remove();
  const dispGroup = document.querySelector(tbGroupSel('Display'));
  if (dispGroup) {
    const mkSensorBtn = (id: string, ic: string, lbl: string, title: string) => {
      const b = document.createElement('button');
      b.id = id; b.className = 'tb-btn toggle'; b.title = title;
      b.innerHTML = `<span class="ic" aria-hidden="true">${ic}</span><span class="lbl">${lbl}</span>`;
      dispGroup.appendChild(b);
      return b;
    };
    const bnv = mkSensorBtn('btn-nvg', uiIcon('moon'), 'NVG', 'NVG night-vision: green-phosphor render mode');
    const bth = mkSensorBtn('btn-thermal', uiIcon('therm'), 'Thermal', 'FLIR thermal: ironbow gradient shaded from DSM height');
    // viewer setters are mutually exclusive; the getters are the source of truth
    // so both button states stay in sync when one mode kicks the other out.
    bnv.onclick = () => {
      v.setNightVision(!v.isNightVisionOn());
      bnv.classList.toggle('on', v.isNightVisionOn());
      bth.classList.toggle('on', v.isThermalOn());
      toast(v.isNightVisionOn() ? 'Night-vision on — green-phosphor render.' : 'Night-vision off', 'info', 1800);
    };
    bth.onclick = () => {
      v.setThermal(!v.isThermalOn());
      bth.classList.toggle('on', v.isThermalOn());
      bnv.classList.toggle('on', v.isNightVisionOn());
      toast(v.isThermalOn() ? 'Thermal (FLIR) on — ironbow gradient over DSM height.' : 'Thermal off', 'info', 1800);
    };
  }
  // ---- Render quality presets (Performance / Balanced / Cinematic) ----
  wireQualityPresets(v);
  const bpl = $('btn-places');
  bpl.onclick = () => {
    const on = bpl.classList.toggle('on');
    $('places-panel').classList.toggle('hidden', !on);
    if (on) renderPlaces(v);
  };
  ($('btn-place-save') as HTMLElement).onclick = () => { savePlace(v); renderPlaces(v); toast('View saved — select it in Places to fly back.', 'ok', 2200); };
  ($('btn-place-clear') as HTMLElement).onclick = () => { localStorage.removeItem(PLACES_KEY); renderPlaces(v); };
  $('btn-help').onclick = () => { $('help-modal').classList.remove('hidden'); ($('btn-help-close') as HTMLElement).focus(); };
  $('btn-help-close').onclick = () => { $('help-modal').classList.add('hidden'); ($('btn-help') as HTMLElement).focus(); };
  ($('btn-cheat-close') as HTMLElement).onclick = () => { $('cheat-modal').classList.add('hidden'); ($('btn-help') as HTMLElement).focus(); };
  wireTourReplay();
  ($('btn-clean') as HTMLElement).onclick = () => setCleanView(!document.body.classList.contains('clean-view'));
  ($('clean-exit') as HTMLElement).onclick = () => setCleanView(false);
  ($('btn-share') as HTMLElement).onclick = () => shareDeepLink(v);
  // "New" strips the deep-link query so a shared view doesn't auto-reload
  $('btn-new').onclick = () => { location.href = location.pathname; };
  $('btn-dsm').onclick = exportDSM;
  $('btn-report').onclick = exportReport;
  $('btn-snapshot').onclick = () => exportSnapshot(v);
  window.onkeydown = (e) => {
    if (e.altKey && (e.key === 'p' || e.key === 'P')) {
      e.preventDefault();
      exportSnapshot(v);
      return;
    }
    if ((e.key === 'p' || e.key === 'P') && !e.altKey && !e.ctrlKey && !e.metaKey) {
      const tag = ((e.target as HTMLElement | null)?.tagName) ?? '';
      if (!/^(INPUT|SELECT|TEXTAREA)$/.test(tag) && !$('screen-viewer').classList.contains('hidden')) {
        e.preventDefault();
        setCleanView(!document.body.classList.contains('clean-view'));
        return;
      }
    }
    const typingNow = (ev: KeyboardEvent) => {
      const tag = ((ev.target as HTMLElement | null)?.tagName) ?? '';
      return /^(INPUT|SELECT|TEXTAREA)$/.test(tag);
    };
    if (e.key === 'Escape' && document.body.classList.contains('clean-view')) {
      setCleanView(false);
      return;
    }
    if (e.key === 'Escape' && !$('cheat-modal').classList.contains('hidden')) {
      $('cheat-modal').classList.add('hidden');
      ($('btn-help') as HTMLElement).focus();
      return;
    }
    if (e.key === 'Escape' && !$('help-modal').classList.contains('hidden')) {
      $('help-modal').classList.add('hidden');
      ($('btn-help') as HTMLElement).focus();
      return;
    }
    // Spot tool: Esc exits placing mode and clears markers (like the other annotation tools)
    if (spotTool && spotTool.handleKey(e)) {
      v.setSpotPlace(false);
      ($('btn-spots') as HTMLElement).classList.remove('on');
      $('spots-panel').classList.add('hidden');
      return;
    }
    // H opens the cheat-sheet overlay; ? toggles the Help panel. Both skip while typing.
    if (!typingNow(e) && !e.ctrlKey && !e.metaKey) {
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        const cm = $('cheat-modal');
        if (cm.classList.contains('hidden')) {
          $('help-modal').classList.add('hidden'); // avoid stacking two modal backdrops
          cm.classList.remove('hidden');
          ($('btn-cheat-close') as HTMLElement).focus();
        } else {
          cm.classList.add('hidden');
          ($('btn-help') as HTMLElement).focus();
        }
        return;
      }
      if (e.key === '?') {
        const hm = $('help-modal');
        if (hm.classList.contains('hidden')) {
          $('cheat-modal').classList.add('hidden');
          hm.classList.remove('hidden');
          ($('btn-help-close') as HTMLElement).focus();
        } else {
          hm.classList.add('hidden');
          ($('btn-help') as HTMLElement).focus();
        }
        return;
      }
    }
  };
  const NICE = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
  v.onHud = (fps, alt, spd, heading, mPer100px) => {
    teleHeadingDeg = heading; // telemetry strip reuses this compass heading source
    $('hud-fps').textContent = fps.toFixed(0);
    $('hud-alt').textContent = alt.toFixed(0);
    $('hud-spd').textContent = spd.toFixed(0);
    $('hud-heading').textContent = heading.toFixed(0);
    ($('compass-rose') as unknown as SVGGElement).style.transform = `rotate(${-heading}deg)`;
    const nice = NICE.reduce((a, b) => (b <= mPer100px && b > a ? b : a), NICE[0]);
    const bar = $('scalebar-bar');
    bar.style.width = Math.max(14, Math.min(140, 100 * nice / mPer100px)) + 'px';
    $('scalebar-lbl').textContent = nice >= 1000 ? (nice / 1000) + ' km' : nice + ' m';
  };
}

/** Bucketed elevation histogram (subsampled for speed) for the sparkline. */
function histogramBins(abs: Float32Array, min: number, max: number, bins = 28): number[] {
  const out = new Array<number>(bins).fill(0);
  const span = Math.max(1e-6, max - min);
  const stride = Math.max(1, Math.floor(abs.length / 20000));
  for (let i = 0; i < abs.length; i += stride) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor((abs[i] - min) / span * bins)));
    out[b]++;
  }
  return out;
}

function drawHistogram(cv: HTMLCanvasElement, bins: number[]): void {
  const ctx = cv.getContext('2d')!;
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const peak = Math.max(1, ...bins);
  const bw = W / bins.length;
  bins.forEach((v, i) => {
    const h = Math.max(1.5, (v / peak) * (H - 8));
    const g = ctx.createLinearGradient(0, H - h, 0, H);
    g.addColorStop(0, '#6f9cc8'); g.addColorStop(1, '#3d5a76');
    ctx.fillStyle = g;
    ctx.fillRect(i * bw + 0.5, H - h - 2, bw - 1, h);
  });
}

function updateMetrics(job: Job, cal: Calibration, depth: DepthResult, cls: ClassificationResult): void {
  const body = $('metrics-body');
  const clsRows = cls.meanIoU !== null
    ? `<div class="mrow"><span>Mean IoU</span><b class="mgood">${cls.meanIoU.toFixed(3)}</b></div>`
      + cls.perClassIoU!.map((v, c) =>
        `<div class="mrow sub"><span>${LCLASS_NAMES[c]}</span><b>${v.toFixed(3)}</b></div>`).join('')
    : `<div class="mrow"><span>Land-cover</span><b>heuristic</b></div>`;
  const lcCard = `<div class="mc"><div class="mc-h">Land cover <span class="sim-tag">simulated</span></div>${clsRows}</div>`;
  const changeCard = changeResult
    ? `<div class="mc"><div class="mc-h">Change detection <span class="sim-tag">simulated</span></div>
      <div class="mrow"><span>Δ erosion (pre→post monsoon)</span><b>${fmtChangeDetection(changeResult)}</b></div></div>`
    : '';
  const elevCard = `<div class="mc"><div class="mc-h">Elevation distribution</div>
      <canvas id="hist-spark" class="hist" width="232" height="52"></canvas>
      <div class="legend-lbl"><span>${cal.minH.toFixed(0)} m</span><span>${cal.maxH.toFixed(0)} m</span></div>
      <div class="legend"></div>
      <div class="legend-lbl"><span>low</span><span>hypsometric</span><span>high</span></div></div>`;
  if (job.scene) {
    // prediction in meters vs. rescaled truth in meters
    const n = SIZE * SIZE;
    const st = gridStats(job.scene.heights);
    const span = Math.max(1e-6, st.max - st.min);
    const pred = new Float32Array(n), truth = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pred[i] = cal.baseElevation + depth.relative[i] * cal.relief;
      truth[i] = cal.baseElevation + ((job.scene.heights[i] - st.min) / span) * cal.relief;
    }
    const m = computeMetrics(pred, truth);
    body.innerHTML = `
      <div class="mc"><div class="mc-h">DSM accuracy <span class="sim-tag">simulated</span></div>
      <div class="mrow"><span>RMSE vs LiDAR ref</span><b class="mgood">${m.rmse.toFixed(2)} m</b></div>
      <div class="mrow"><span>MAE</span><b>${m.mae.toFixed(2)} m</b></div>
      <div class="mrow"><span>Correlation</span><b>${m.corr.toFixed(3)}</b></div>
      <div class="mrow"><span>Bias</span><b>${m.bias >= 0 ? '+' : ''}${m.bias.toFixed(2)} m</b></div>
      <div class="mrow"><span>DSM range</span><b>${cal.minH.toFixed(0)}–${cal.maxH.toFixed(0)} m</b></div></div>
      ${elevCard}
      ${lcCard}
      ${changeCard}
      <div class="mnote">All values <b>simulated</b> — not survey-grade. ${cal.method}. Reference: simulated airborne LiDAR, 0.5 m GSD.</div>`;
  } else {
    body.innerHTML = `
      <div class="mc"><div class="mc-h">Elevation</div>
      <div class="mrow"><span>Path</span><b>${job.georeferenced ? 'Absolute DSM' : 'Relative rDSM'}</b></div>
      <div class="mrow"><span>DSM range</span><b>${cal.minH.toFixed(0)}–${cal.maxH.toFixed(0)} m</b></div>
      <div class="mrow"><span>Reference</span><b>—</b></div></div>
      ${elevCard}
      ${lcCard}
      ${changeCard}
      <div class="mnote">All values <b>simulated</b> — no reference capture for uploads; heights are simulated from image cues. ${cal.method}.</div>`;
  }
  drawHistogram($('hist-spark') as HTMLCanvasElement, histogramBins(cal.absolute, cal.minH, cal.maxH));
}

/** Re-render the metrics panel with the current scene args (keeps the Δ-erosion card in sync). */
function refreshMetrics(): void {
  if (activeJob && activeCal && activeDepth && activeCls) updateMetrics(activeJob, activeCal, activeDepth, activeCls);
}

/** Floating land-cover legend with per-class coverage. */
function updateLegend(cls: ClassificationResult): void {
  const el = $('class-legend');
  el.innerHTML = `<h4>${uiIcon('leaf')} Land cover <span class="sim-tag">simulated</span></h4>` +
    LCLASS_NAMES.map((name, c) =>
      `<div class="lrow"><span class="chip" style="background:${cssColor(LCLASS_COLORS[c])}"></span>
       <span>${name}</span><b>${(cls.coverage[c] * 100).toFixed(1)}%</b></div>`).join('') +
    `<div class="mnote">${cls.model}</div>`;
}

function download(url: string, name: string): void {
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function exportDSM(): void {
  if (!activeCal) return;
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(SIZE, SIZE);
  const span = Math.max(1e-6, activeCal.maxH - activeCal.minH);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const g = Math.round(((activeCal.absolute[i] - activeCal.minH) / span) * 255);
    img.data[i * 4] = g; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  c.toBlob(b => b && download(URL.createObjectURL(b), 'aero3d_dsm.png'));
  toast('DSM PNG exported', 'ok');
}

function exportReport(): void {
  if (!activeJob || !activeCal) return;
  const j = activeJob, c = activeCal;
  const txt =
`=====================================================
Aero3D GEOSPATIAL TERRAIN & ELEVATION SUMMARY REPORT
Generated: ${new Date().toLocaleString('en-GB')}
EPSG: 4326 (WGS 84)${c.georeferenced ? '' : ' — RELATIVE DATUM (non-georeferenced input)'}
=====================================================

INPUT:
  Source: ${j.name}
  Type: ${j.formatLabel}
  Grid: ${SIZE}x${SIZE} @ ~${((j.scene?.worldSize ?? 1200) / SIZE).toFixed(1)} m GSD

ELEVATION ANALYTICS:
  Minimum Elevation: ${c.minH.toFixed(1)} m
  Maximum Elevation: ${c.maxH.toFixed(1)} m
  Relief: ${(c.maxH - c.minH).toFixed(1)} m
  Calibration: ${c.method}

PIPELINE (SIMULATED PROTOTYPE):
  Backbone: mono-depth v2.1 (simulated inference)
  Calibration: ${c.georeferenced ? 'SRTM-30m anchored affine fit' : 'scene statistics, relative datum'}
  Mesh: 65,536 verts / 130,050 tris, optical texture projected

NOTE: Prototype build — elevation values are simulated for
demonstration and are NOT survey-grade measurements.
=====================================================
Aero3D Cartographic Analytics Engine`;
  download(URL.createObjectURL(new Blob([txt], { type: 'text/plain' })), 'Aero3D_Elevation_Report.txt');
  toast('Survey report exported', 'ok');
}

/**
 * 3D view snapshot (Pix4Dinspect-style "3D screenshot" inspection output).
 * Calls Viewer.captureFrame() for the raw WebGL frame (dataURL), then
 * composites it onto a 2D canvas with a scene-name + timestamp title bar and
 * triggers a PNG download. All compositing, filename generation and the
 * download trigger live here in the UI area.
 */
function exportSnapshot(v: TerrainViewer): void {
  const dataURL = v.captureFrame(); // renders one frame and returns a dataURL (preserveDrawingBuffer-safe)
  const img = new Image();
  img.onerror = () => toast('Snapshot failed — could not read the 3D frame.', 'err');
  img.onload = () => {
    const BAR = Math.max(56, Math.round(img.width * 0.055)); // title-bar height scales with frame width
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height + BAR;
    const ctx = c.getContext('2d')!;
    // title bar
    ctx.fillStyle = '#0b1526';
    ctx.fillRect(0, 0, c.width, BAR);
    ctx.fillStyle = '#6f9cc8';
    ctx.fillRect(0, BAR - 3, c.width, 3);
    const sceneName = activeJob?.name ?? 'Aero3D scene';
    const stamp = new Date().toLocaleString('en-GB', { hour12: false });
    const t1 = Math.round(BAR * 0.34), t2 = Math.round(BAR * 0.28);
    const pad = Math.round(BAR * 0.28);
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#eaf2ff';
    ctx.font = `600 ${t1}px system-ui, -apple-system, sans-serif`;
    ctx.fillText(sceneName, pad, BAR * 0.32);
    ctx.fillStyle = '#9fb3cc';
    ctx.font = `400 ${t2}px system-ui, -apple-system, sans-serif`;
    ctx.fillText(`${stamp} · Aero3D 3D snapshot (simulated)`, pad, BAR * 0.68);
    // right-aligned source format label
    if (activeJob) {
      ctx.fillStyle = '#7d90ab';
      ctx.font = `400 ${t2}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillText(activeJob.formatLabel, c.width - pad, BAR * 0.32);
      ctx.textAlign = 'left';
    }
    ctx.drawImage(img, 0, BAR);
    const slug = sceneName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'scene';
    const fileStamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
    c.toBlob((b) => {
      if (!b) { toast('Snapshot failed to render.', 'err'); return; }
      download(URL.createObjectURL(b), `aero3d_3dview_${slug}_${fileStamp}.png`);
      toast('3D view snapshot exported', 'ok');
    }, 'image/png');
  };
  img.src = dataURL;
}

export function init(): void {
  buildLanding();
  wireProcessingError();
  // deep-link restore: a shared ?scene=… URL skips the landing page and runs
  // the demo scene straight through the pipeline, then applies camera/layers.
  pendingDeepLink = parseDeepLink();
  if (pendingDeepLink) {
    const job = jobForSceneId(pendingDeepLink.scene);
    if (job) { startJob(job); return; }
    pendingDeepLink = null; // unknown scene id — land normally
  }
}
