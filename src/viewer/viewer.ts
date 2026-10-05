// Three.js terrain viewer: orbit / drone / first-person + cinematic path,
// analysis overlays (slope, hypsometric, contours) and a height probe.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildTerrainGeometry, applyExaggeration, sampleHeightWorld, computeExGBands, VEG_BAND_NAMES, VEG_BAND_COLORS, ironbowColor, THERMAL_BAND_NAMES, THERMAL_BAND_STOPS, createClipPlaneUniforms, CLIP_PLANE_GLSL_COMMON, CLIP_PLANE_GLSL_DISCARD, type ExGBands } from '../lib/mesh';
import { computeViewshed } from '../lib/viewshed';
import type { SceneData } from '../lib/scenes';
import { calcVolume, fmtVolume, type Calibration } from '../lib/pipeline';
import type { ClassificationResult } from '../lib/classify';

export type ViewMode = 'orbit' | 'drone' | 'fpv';
export type Overlay = 0 | 1 | 2 | 3 | 4; // texture | slope | hypsometric | land-cover | vegetation-index

export interface ProbeInfo {
  x: number; y: number; z: number;
  elevation: number; slopeDeg: number;
}

/** Named camera viewpoint for the POI/bookmark tour. */
export interface Bookmark {
  name: string;
  pos: [number, number, number];
  tgt: [number, number, number];
}

/** Hypsometric ramp in TS, mirroring the terrain shader's hypso(). */
export function hypsoColor(t: number, out: THREE.Color): THREE.Color {
  const c1 = [0.16, 0.45, 0.20], c2 = [0.85, 0.78, 0.35], c3 = [0.55, 0.38, 0.24], c4 = [0.92, 0.92, 0.94];
  const mix3 = (a: number[], b: number[], k: number) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
  const c = t < 0.33 ? mix3(c1, c2, t / 0.33) : t < 0.66 ? mix3(c2, c3, (t - 0.33) / 0.33) : mix3(c3, c4, (t - 0.66) / 0.34);
  return out.setRGB(c[0], c[1], c[2]);
}

function skyDome(): THREE.Mesh {
  const geo = new THREE.SphereGeometry(9000, 24, 12);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      top: { value: new THREE.Color(0x2f6fd0) },
      mid: { value: new THREE.Color(0x9cc8ee) },
      bot: { value: new THREE.Color(0xe8f2f8) },
    },
    vertexShader: `varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `varying vec3 vP; uniform vec3 top,mid,bot;
      void main(){ float h = normalize(vP).y;
        vec3 c = h > 0.0 ? mix(mid, top, pow(h, 0.6)) : mix(mid, bot, pow(-h, 0.5));
        gl_FragColor = vec4(c, 1.0); }`,
  });
  return new THREE.Mesh(geo, mat);
}

export class TerrainViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private orbit: OrbitControls;
  private container: HTMLElement;
  private terrain: THREE.Mesh | null = null;
  private classTex: THREE.DataTexture | null = null;
  private changeTex: THREE.DataTexture | null = null; // post-event DSM for change-detection compare
  // Excess Green vegetation-index band coverage, computed once per scene in load()
  private vegBands: ExGBands | null = null;
  private terrainUniforms = {
    uOverlay: { value: 0 },
    uContours: { value: 0 },
    uMinH: { value: 0 },
    uMaxH: { value: 100 },
    uCompareX: { value: -1 }, // screen-fraction split; <0 = off
    uViewportW: { value: 1 },
    uScanOn: { value: 0 },
    uScanR: { value: 0 },
    uScanA: { value: 0 },
    uScanI: { value: 0.85 },
    uFloodY: { value: -1e9 }, // flood water level in world units; <0 levels = off
    uClassMap: { value: null as THREE.DataTexture | null }, // land-cover class ids
    uViewshedOn: { value: 0 }, // 1 = viewshed tint active
    uViewshedTex: { value: null as THREE.DataTexture | null }, // visibility mask (red channel)
    uNight: { value: 0 },   // NVG green-phosphor sensor render mode
    uThermal: { value: 0 }, // FLIR ironbow thermal render mode (height-shaded)
    uChangeMode: { value: 0 }, // 1 = change-detection compare (baseline ↔ post-event DSM)
    uChangeTex: { value: null as THREE.DataTexture | null }, // post-event DSM heights (float, row-flipped)
    uChangeExagg: { value: 1 }, // exaggeration factor applied to uChangeTex heights
    ...createClipPlaneUniforms(), // elevation slice / clip-plane inspection (off by default)
  };
  private probeMarker: THREE.Group;
  private raycaster = new THREE.Raycaster();
  // viewshed / line-of-sight analysis state
  private viewshedPlace = false;   // modal click-to-place mode for the observer
  private viewshedActive = false;  // a viewshed overlay is currently displayed
  private viewshedTex: THREE.DataTexture | null = null;
  private viewshedTowerH = 15;     // mast/tower height in meters above ground
  private viewshedObs = { x: 0, z: 0, groundM: 0 }; // unexaggerated meters
  private observerMarker: THREE.Group | null = null;
  // profile-traverse chart cursor pin (synced 3D marker for the 2D chart scrubber)
  private profileMarker: THREE.Mesh | null = null;
  // LiDAR-style point-cloud layer
  private points: THREE.Points | null = null;
  private pointsOn = false;
  // flood simulation state
  private floodPlane: THREE.Mesh | null = null;
  private floodFrac = -1; // <0 = off, else 0..1 of scene relief

  mode: ViewMode = 'orbit';
  private drone = { yaw: 0.6, pitch: -0.42, speed: 26 };
  private keys = new Set<string>();
  private dragging = false; private lastPX = 0; private lastPY = 0;
  private fpv = { yaw: 0, pitch: 0 };
  private cinematic = false; private cineT = 0;
  private cineCurve: THREE.CatmullRomCurve3 | null = null;
  // auto-orbit kiosk mode: slow azimuth spin around the orbit target
  private autoOrbitOn = false;
  private autoOrbitP = 0; // 0..1 eased-in-out spin progress (smooth spin-up/down)
  private readonly autoOrbitSpeed = 0.08; // rad/s ≈ 78 s per revolution
  private probeEnabled = false;
  private exaggeration = 1;
  private scanOn = false;
  private scanT = 0;
  private surveyGrid: THREE.Group | null = null;
  // measurement tool state
  private measureOn = false;
  private measureMode: 'distance' | 'volume' = 'distance';
  private measurePts: THREE.Vector3[] = [];
  private measureDone: { pts: THREE.Vector3[]; group: THREE.Group; color: number; kind: 'distance' | 'volume'; total: string }[] = [];
  private measureLive: THREE.Group | null = null;
  private downX = 0; private downY = 0;
  private cal: Calibration | null = null;
  private data: SceneData | null = null;
  // mini-map 2D overview cache (offscreen hypsometric blit, rebuilt on scene load / exaggeration change)
  private minimapCache: HTMLCanvasElement | null = null;
  private minimapDir = new THREE.Vector3();

  // camera fly-to animation state
  private fly = {
    active: false, t: 0, dur: 2.4,
    fromPos: new THREE.Vector3(), toPos: new THREE.Vector3(),
    fromTgt: new THREE.Vector3(), toTgt: new THREE.Vector3(),
  };

  onProbe: (p: ProbeInfo | null) => void = () => {};
  /** Fires when a viewshed is computed, or null when it is cleared. */
  onViewshed: (v: { x: number; z: number; groundM: number; towerH: number; pct: number } | null) => void = () => {};
  /** fps, camera altitude (m), speed (m/s), camera heading (deg, 0 = north/-Z), meters per 100 px at view distance */
  onHud: (fps: number, alt: number, spd: number, headingDeg: number, mPer100px: number) => void = () => {};
  onMeasure: () => void = () => {};
  /** Fires on every auto-orbit state change — incl. input-cancelled pauses — so the toolbar can sync its toggle. */
  onAutoOrbit: (on: boolean) => void = () => {};
  /** Spot-elevation tool: fires with world (x, z) when a terrain click lands while spot placing is on. */
  onSpotPlace: (x: number, z: number) => void = () => {};
  private spotPlace = false;
  private fpsAcc = 0; private fpsN = 0; private fpsT = 0;
  // analog FPV look for drone mode: Betaflight-style OSD overlay canvas +
  // CSS vignette (cheap screen-space barrel-distortion feel) + FOV breathing
  private osdWrap: HTMLDivElement | null = null;
  private osdCanvas: HTMLCanvasElement | null = null;
  private osdCtx: CanvasRenderingContext2D | null = null;
  private fpvBreathT = 0;   // FOV-breathing clock, reset on drone-mode entry
  private fpvBattery = 100; // OSD battery %, drains while in drone mode
  private readonly fpvBaseFov = 58; // camera fov restored when leaving drone mode
  private readonly osdFwd = new THREE.Vector3();
  private readonly osdRight = new THREE.Vector3();

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.appendChild(this.renderer.domElement);

    // analog FPV overlay stack: OSD canvas + vignette, hidden unless drone mode
    const wrap = document.createElement('div');
    wrap.style.cssText = 'position:absolute;inset:0;pointer-events:none;display:none;z-index:5;';
    const osd = document.createElement('canvas');
    osd.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';
    const vig = document.createElement('div');
    vig.style.cssText = 'position:absolute;inset:0;pointer-events:none;' +
      'background:radial-gradient(ellipse at center, rgba(0,0,0,0) 50%, rgba(0,0,0,0.26) 80%, rgba(0,0,0,0.52) 100%);' +
      'box-shadow:inset 0 0 160px rgba(0,0,0,0.42);';
    wrap.append(osd, vig);
    container.appendChild(wrap);
    this.osdWrap = wrap;
    this.osdCanvas = osd;
    this.osdCtx = osd.getContext('2d');

    this.camera = new THREE.PerspectiveCamera(58, 1, 0.5, 20000);
    this.camera.position.set(600, 500, 600);

    this.scene.fog = new THREE.Fog(0xd8e8f2, 2500, 9000);
    const skyMesh = skyDome();
    this.scene.add(skyMesh);
    (this as { _skyU?: { [k: string]: { value: THREE.Color } } })._skyU =
      (skyMesh.material as THREE.ShaderMaterial).uniforms as { [k: string]: { value: THREE.Color } };

    const hemi = new THREE.HemisphereLight(0xbfd9ff, 0x8a7a5f, 0.75);
    this.scene.add(hemi);
    (this as { _hemi?: THREE.HemisphereLight })._hemi = hemi;
    const sun = new THREE.DirectionalLight(0xfff2dd, 2.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);
    this.scene.add(sun.target);
    (this as { _sun?: THREE.DirectionalLight })._sun = sun;
    this.setSun(27, 49); // default sun position, matches old hard-coded look

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.06;
    this.orbit.maxPolarAngle = Math.PI * 0.495;
    this.orbit.addEventListener('start', () => { this.cancelFly(); this.pauseAutoOrbit(); });

    // probe marker: ring + pin
    const ring = new THREE.RingGeometry(6, 9, 40);
    const pin = new THREE.CylinderGeometry(1.2, 1.2, 26, 10);
    const mMat = new THREE.MeshBasicMaterial({ color: 0xff3b30, side: THREE.DoubleSide, depthTest: false, transparent: true });
    const g = new THREE.Group();
    const r1 = new THREE.Mesh(ring, mMat); r1.rotation.x = -Math.PI / 2;
    const p1 = new THREE.Mesh(pin, mMat); p1.position.y = 13;
    g.add(r1, p1);
    this.probeMarker = g;
    this.probeMarker.visible = false;
    this.probeMarker.renderOrder = 999;
    this.scene.add(this.probeMarker);

    this.bindInput();
    this.resize();
  }

  // ------------------------------------------------------------- loading ---
  load(data: SceneData, cal: Calibration, cls: ClassificationResult | null): void {
    this.data = data; this.cal = cal;
    if (this.terrain) { this.scene.remove(this.terrain); this.terrain.geometry.dispose(); }
    if (this.classTex) { this.classTex.dispose(); this.classTex = null; }
    // drop any stale change-detection state from the previous scene
    if (this.changeTex) { this.changeTex.dispose(); this.changeTex = null; }
    this.terrainUniforms.uChangeTex.value = null;
    this.terrainUniforms.uChangeMode.value = 0;
    this.terrainUniforms.uCompareX.value = -1;
    this.clearViewshed(); // drop any stale viewshed overlay / observer marker
    this.clearProfileMarker();

    const geo = buildTerrainGeometry(cal.absolute, data.size, data.worldSize, this.exaggeration);
    const tex = new THREE.CanvasTexture(data.rgb);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;

    // land-cover class id texture (nearest-sampled)
    if (cls) {
      const ct = new THREE.DataTexture(new Uint8Array(cls.labels), cls.size, cls.size, THREE.RedFormat, THREE.UnsignedByteType);
      ct.magFilter = THREE.NearestFilter;
      ct.minFilter = THREE.NearestFilter;
      ct.needsUpdate = true;
      this.classTex = ct;
    }
    this.terrainUniforms.uClassMap.value = this.classTex;
    this.vegBands = computeExGBands(data.rgb); // vegetation-index legend, once per scene

    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.94, metalness: 0.0 });
    const U = this.terrainUniforms;
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNormal;')
        .replace('#include <project_vertex>',
          `#include <project_vertex>
           vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
           vWNormal = normalize(mat3(modelMatrix) * objectNormal);`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>',
          `#include <common>
           varying vec3 vWPos; varying vec3 vWNormal;
           uniform int uOverlay; uniform float uContours; uniform float uMinH; uniform float uMaxH;
           uniform float uCompareX; uniform float uViewportW;
           uniform float uScanOn; uniform float uScanR; uniform float uScanA; uniform float uScanI;
           uniform float uFloodY;
           uniform sampler2D uClassMap;
           uniform float uViewshedOn;
           uniform sampler2D uViewshedTex;
           uniform float uNight;
           uniform float uThermal;
           uniform float uChangeMode;
           uniform sampler2D uChangeTex;
           uniform float uChangeExagg;
           ` + CLIP_PLANE_GLSL_COMMON + `vec3 hypso(float t){
             vec3 c1=vec3(0.16,0.45,0.20), c2=vec3(0.85,0.78,0.35), c3=vec3(0.55,0.38,0.24), c4=vec3(0.92,0.92,0.94);
             return t<0.33 ? mix(c1,c2,t/0.33) : t<0.66 ? mix(c2,c3,(t-0.33)/0.33) : mix(c3,c4,(t-0.66)/0.34); }
           vec3 ironbow(float t){
             vec3 c0=vec3(0.00,0.00,0.00), c1=vec3(0.45,0.05,0.50), c2=vec3(0.95,0.10,0.05),
                  c3=vec3(1.00,0.65,0.05), c4=vec3(1.00,1.00,0.90);
             return t<0.25 ? mix(c0,c1,t/0.25) : t<0.50 ? mix(c1,c2,(t-0.25)/0.25)
                  : t<0.75 ? mix(c2,c3,(t-0.50)/0.25) : mix(c3,c4,(t-0.75)/0.25); }`)
        .replace('#include <map_fragment>',
          `#include <map_fragment>
          {
            ` + CLIP_PLANE_GLSL_DISCARD + `
            vec3 optCol = diffuseColor.rgb; // untouched optical texture (compare mode)
            vec3 nrm = normalize(vWNormal);
            float slopeDeg = degrees(acos(clamp(nrm.y, -1.0, 1.0)));
            if (uOverlay == 1) {
              float t = clamp(slopeDeg / 45.0, 0.0, 1.0);
              vec3 sc = t < 0.5 ? mix(vec3(0.18,0.68,0.30), vec3(0.95,0.82,0.18), t*2.0)
                                : mix(vec3(0.95,0.82,0.18), vec3(0.88,0.16,0.12), (t-0.5)*2.0);
              diffuseColor.rgb = mix(diffuseColor.rgb, sc, 0.88);
            } else if (uOverlay == 2) {
              float t = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 0.001), 0.0, 1.0);
              diffuseColor.rgb = mix(diffuseColor.rgb, hypso(t), 0.88);
            } else if (uOverlay == 3) {
              float cid = floor(texture2D(uClassMap, vMapUv).r * 255.0 + 0.5);
              vec3 cc = vec3(0.66, 0.56, 0.39); // bare ground
              if (cid > 0.5 && cid < 1.5) cc = vec3(0.91, 0.30, 0.24);      // building
              else if (cid > 1.5 && cid < 2.5) cc = vec3(0.95, 0.77, 0.06); // road
              else if (cid > 2.5 && cid < 3.5) cc = vec3(0.18, 0.80, 0.44); // vegetation
              else if (cid > 3.5) cc = vec3(0.20, 0.60, 0.86);              // water
              diffuseColor.rgb = mix(diffuseColor.rgb, cc, 0.92);
            } else if (uOverlay == 4) {
              // Excess Green (ExG) vegetation index, shader-side: ExG = 2G - R - B
              // from the optical texel (linear). Green = healthy, yellow = moderate,
              // red = stressed; thresholds mirror computeExGBands() in lib/mesh.
              float exg = 2.0 * diffuseColor.g - diffuseColor.r - diffuseColor.b;
              float t = clamp((exg + 0.3) / 1.2, 0.0, 1.0);
              vec3 ec = t < 0.5 ? mix(vec3(0.88, 0.16, 0.12), vec3(0.95, 0.82, 0.18), t * 2.0)
                                : mix(vec3(0.95, 0.82, 0.18), vec3(0.18, 0.68, 0.30), (t - 0.5) * 2.0);
              diffuseColor.rgb = mix(diffuseColor.rgb, ec, 0.9);
            }
            if (uContours > 0.5) {
              float e = vWPos.y / 10.0;
              float w = fwidth(e) * 1.4 + 1e-4;
              float minorL = 1.0 - smoothstep(0.0, w, abs(fract(e + 0.5) - 0.5) - 0.465);
              float e2 = vWPos.y / 50.0;
              float w2 = fwidth(e2) * 1.4 + 1e-4;
              float majorL = 1.0 - smoothstep(0.0, w2, abs(fract(e2 + 0.5) - 0.5) - 0.45);
              float line = clamp(minorL * 0.4 + majorL * 0.75, 0.0, 0.85);
              diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.08, 0.09, 0.10), line);
            }
            if (uCompareX >= 0.0) {
              // swipe compare: change-detection mode → left = baseline DSM hypsometric, right = post-event DSM hypsometric
              //                standard mode       → left = optical RGB,           right = DSM hypsometric + major contours
              float cf = gl_FragCoord.x / max(uViewportW, 1.0);
              float ct = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 0.001), 0.0, 1.0);
              if (cf <= uCompareX) {
                diffuseColor.rgb = uChangeMode > 0.5 ? hypso(ct) : optCol;
              } else {
                float cy = vWPos.y; // comparison height, world units
                if (uChangeMode > 0.5) cy = texture2D(uChangeTex, vMapUv).r * uChangeExagg;
                float qt = clamp((cy - uMinH) / max(uMaxH - uMinH, 0.001), 0.0, 1.0);
                diffuseColor.rgb = hypso(qt);
                float ce = cy / 50.0;
                float cw = fwidth(ce) * 1.4 + 1e-4;
                float cm = 1.0 - smoothstep(0.0, cw, abs(fract(ce + 0.5) - 0.5) - 0.45);
                diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.08, 0.09, 0.10), clamp(cm * 0.75, 0.0, 0.85));
              }
              if (abs(cf - uCompareX) < 0.0025) diffuseColor.rgb = vec3(1.0);
            }
            // flood simulation: submerged terrain reads as shallow water
            if (vWPos.y < uFloodY) {
              float depthT = clamp((uFloodY - vWPos.y) / 60.0, 0.0, 1.0);
              vec3 water = mix(vec3(0.28, 0.58, 0.78), vec3(0.06, 0.28, 0.52), depthT);
              diffuseColor.rgb = mix(diffuseColor.rgb, water, 0.78);
            }
            // viewshed tint: green = intervisible, red = occluded (Esri/Supergeo convention)
            if (uViewshedOn > 0.5) {
              float vsv = texture2D(uViewshedTex, vMapUv).r;
              vec3 vc = vsv > 0.5 ? vec3(0.16, 0.72, 0.28) : vec3(0.84, 0.20, 0.14);
              diffuseColor.rgb = mix(diffuseColor.rgb, vc, 0.60);
            }
            // survey scan-wave: expanding sonar ring + rotating radar wedge
            if (uScanOn > 0.5) {
              vec2 sp = vWPos.xz;
              float rr = length(sp);
              float ring = 1.0 - smoothstep(0.0, 90.0, abs(rr - uScanR));
              float ang = atan(sp.y, sp.x);
              float dw = abs(mod(ang - uScanA + 3.14159265, 6.2831853) - 3.14159265);
              float wedge = (1.0 - smoothstep(0.0, 0.55, dw)) * step(rr, uScanR + 60.0) * 0.4;
              float glow = clamp(ring + wedge, 0.0, 1.2);
              diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.13, 0.84, 1.0), clamp(glow * uScanI, 0.0, 0.8));
            }
            // NVG night-vision: green-phosphor amplification of the sensor luminance
            if (uNight > 0.5) {
              float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
              float nl = pow(clamp(lum * 2.8, 0.0, 1.0), 1.15); // phosphor gain + soft roll-off
              vec3 nvg = vec3(0.02, 0.28, 0.07) + vec3(0.30, 1.05, 0.32) * nl;
              float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
              diffuseColor.rgb = nvg * (0.93 + 0.07 * grain); // intensifier grain
            }
            // FLIR thermal: ironbow gradient shaded from the DSM height
            if (uThermal > 0.5) {
              float tt = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 0.001), 0.0, 1.0);
              diffuseColor.rgb = ironbow(tt);
            }
          }`);
    };

    this.terrain = new THREE.Mesh(geo, mat);
    this.terrain.castShadow = true;
    this.terrain.receiveShadow = true;
    this.scene.add(this.terrain);
    this.terrainUniforms.uMinH.value = cal.minH * this.exaggeration;
    this.terrainUniforms.uMaxH.value = cal.maxH * this.exaggeration;
    this.terrainUniforms.uChangeExagg.value = this.exaggeration;
    // clip-plane world extent for the elevation-slice tool; disabled per scene by default
    this.terrainUniforms.uClipMin.value = -data.worldSize / 2;
    this.terrainUniforms.uClipMax.value = data.worldSize / 2;
    this.clearClipPlane();
    this.rebuildSurveyGrid();
    if (this.pointsOn) this.rebuildPoints();
    this.setFlood(this.floodFrac); // re-anchor water plane to the new scene/calibration

    // diorama base
    const baseH = 40;
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(data.worldSize, baseH, data.worldSize),
      new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.9 }),
    );
    base.position.y = cal.minH * this.exaggeration - baseH / 2 - 2;
    base.receiveShadow = true;
    this.scene.add(base);
    (this as { _base?: THREE.Mesh })._base = base;

    // fit sun shadow camera
    const sun = (this as { _sun?: THREE.DirectionalLight })._sun!;
    const R = data.worldSize * 0.75;
    sun.shadow.camera.left = -R; sun.shadow.camera.right = R;
    sun.shadow.camera.top = R; sun.shadow.camera.bottom = -R;
    sun.shadow.camera.far = 6000;
    sun.target.position.set(0, 0, 0);
    sun.shadow.camera.updateProjectionMatrix();

    this.buildCinematicPath(data, cal);
    this.buildMinimapCache();
    this.flyToOverview(true); // animated intro fly-in
  }

  private buildCinematicPath(data: SceneData, cal: Calibration): void {
    const W = data.worldSize, hMax = cal.maxH * this.exaggeration;
    const pts = [
      new THREE.Vector3(-W * 0.42, hMax + 260, -W * 0.42),
      new THREE.Vector3(W * 0.30, hMax + 150, -W * 0.30),
      new THREE.Vector3(W * 0.44, hMax + 90, W * 0.28),
      new THREE.Vector3(-W * 0.10, hMax + 60, W * 0.44),
      new THREE.Vector3(-W * 0.44, hMax + 170, W * 0.05),
    ];
    this.cineCurve = new THREE.CatmullRomCurve3(pts, true, 'centripetal', 0.6);
  }

  cancelFly(): void {
    if (this.fly.active) {
      this.fly.active = false;
      this.orbit.enabled = this.mode === 'orbit';
    }
  }

  /** Animated camera flight (ease-in-out cubic) to pos/tgt, orbit mode. */
  flyTo(pos: THREE.Vector3, tgt: THREE.Vector3): void {
    this.setMode('orbit');
    this.fly.fromPos.copy(this.camera.position);
    this.fly.toPos.copy(pos);
    this.fly.fromTgt.copy(this.orbit.target);
    this.fly.toTgt.copy(tgt);
    this.fly.t = 0;
    this.fly.active = true;
    this.orbit.enabled = false;
  }

  flyToOverview(animate = false): void {
    if (!this.data || !this.cal) return;
    const W = this.data.worldSize, h = (this.cal.maxH - this.cal.minH) * this.exaggeration;
    const pos = new THREE.Vector3(W * 0.55, h + W * 0.55, W * 0.55);
    const tgt = new THREE.Vector3(0, this.cal.minH * this.exaggeration + h * 0.3, 0);
    if (animate) this.flyTo(pos, tgt);
    else {
      this.setMode('orbit');
      this.fly.active = false;
      this.camera.position.copy(pos);
      this.orbit.target.copy(tgt);
      this.orbit.enabled = true;
      this.orbit.update();
    }
  }

  /** Current camera viewpoint, for saved bookmarks. */
  getCameraView(): { pos: [number, number, number]; tgt: [number, number, number] } {
    return {
      pos: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      tgt: [this.orbit.target.x, this.orbit.target.y, this.orbit.target.z],
    };
  }

  /** Built-in POI presets computed from the scene heights. */
  bookmarkPresets(): Bookmark[] {
    if (!this.data || !this.cal) return [];
    const n = this.data.size, W = this.data.worldSize, half = W / 2;
    const cell = W / (n - 1), ex = this.exaggeration;
    const hgt = this.cal.absolute;
    let iMax = 0, iMin = 0;
    for (let i = 1; i < hgt.length; i++) {
      if (hgt[i] > hgt[iMax]) iMax = i;
      if (hgt[i] < hgt[iMin]) iMin = i;
    }
    const wx = (i: number) => -half + (i % n) * cell;
    const wz = (i: number) => -half + Math.floor(i / n) * cell;
    const minY = this.cal.minH * ex, range = (this.cal.maxH - this.cal.minH) * ex;
    return [
      {
        name: '⛰ Highest peak', pos: [wx(iMax) + W * 0.28, hgt[iMax] * ex + range * 0.9, wz(iMax) + W * 0.28],
        tgt: [wx(iMax), hgt[iMax] * ex, wz(iMax)],
      },
      {
        name: '🏞 Valley floor', pos: [wx(iMin) - W * 0.2, hgt[iMin] * ex + range * 0.7, wz(iMin) - W * 0.2],
        tgt: [wx(iMin), hgt[iMin] * ex, wz(iMin)],
      },
      {
        name: '🛰 Overview', pos: [W * 0.55, range + W * 0.55, W * 0.55],
        tgt: [0, minY + range * 0.3, 0],
      },
      {
        name: '🌅 Low oblique', pos: [W * 0.52, minY + range * 0.45, 0],
        tgt: [0, minY + range * 0.25, 0],
      },
    ];
  }

  // ---------------------------------------------------------------- input ---
  private bindInput(): void {
    const el = this.renderer.domElement;
    window.addEventListener('keydown', (e) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT') return;
      if (tag !== 'BUTTON') this.pauseAutoOrbit(); // any key pauses the kiosk spin (toolbar-button keys manage the toggle themselves)
      this.keys.add(e.code);
      if (e.code === 'Escape' && this.measureOn) this.cancelMeasure();
      if (e.code === 'Escape' && this.viewshedPlace) this.setViewshedPlace(false);
      this.cancelFly();
      if (this.cinematic && this.mode === 'drone') this.setCinematic(false);
      if (e.code === 'KeyV') this.cycleMode();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    el.addEventListener('pointerdown', (e) => {
      this.cancelFly();
      if (this.mode === 'fpv') { el.requestPointerLock(); return; }
      this.dragging = true; this.lastPX = e.clientX; this.lastPY = e.clientY;
    });
    window.addEventListener('pointerup', () => { this.dragging = false; });
    window.addEventListener('pointermove', (e) => {
      if (this.mode === 'fpv' && document.pointerLockElement === el) {
        this.fpv.yaw -= e.movementX * 0.0022;
        this.fpv.pitch = Math.max(-1.45, Math.min(1.45, this.fpv.pitch - e.movementY * 0.0022));
        return;
      }
      if (this.dragging && this.mode === 'drone') {
        this.drone.yaw -= (e.clientX - this.lastPX) * 0.0032;
        this.drone.pitch = Math.max(-1.4, Math.min(1.2, this.drone.pitch - (e.clientY - this.lastPY) * 0.0032));
        this.lastPX = e.clientX; this.lastPY = e.clientY;
      }
    });
    el.addEventListener('wheel', (e) => {
      if (this.mode === 'drone') {
        e.preventDefault();
        this.drone.speed = Math.max(4, Math.min(160, this.drone.speed * (e.deltaY > 0 ? 0.88 : 1.14)));
      }
    }, { passive: false });
    el.addEventListener('click', (e) => {
      if (this.viewshedPlace && this.mode !== 'fpv') this.placeViewshed(e.clientX, e.clientY);
      else if (this.probeEnabled && this.mode !== 'fpv') this.probe(e.clientX, e.clientY);
      else if (this.spotPlace && this.mode !== 'fpv') this.spotPlaceClick(e.clientX, e.clientY);
      else if (this.measureOn && this.mode === 'orbit') {
        // ignore drags: only treat as a measurement click if the pointer barely moved
        const dx = e.clientX - this.downX, dy = e.clientY - this.downY;
        if (dx * dx + dy * dy < 49) this.measureClick(e.clientX, e.clientY);
      }
    });
    el.addEventListener('pointerdown', (e) => { this.downX = e.clientX; this.downY = e.clientY; });
    el.addEventListener('dblclick', () => { if (this.measureOn && this.mode === 'orbit') this.finishMeasure(); });
    window.addEventListener('resize', () => this.resize());
  }

  cycleMode(): void {
    this.setMode(this.mode === 'orbit' ? 'drone' : this.mode === 'drone' ? 'fpv' : 'orbit');
  }

  setMode(m: ViewMode): void {
    this.mode = m;
    this.setCinematic(false);
    if (m !== 'orbit') this.setAutoOrbit(false); // kiosk spin only makes sense in orbit mode
    this.fly.active = false;
    this.orbit.enabled = m === 'orbit';
    if (m === 'drone' && this.data && this.cal) {
      const W = this.data.worldSize;
      this.camera.position.set(-W * 0.3, this.cal.maxH * this.exaggeration + 220, -W * 0.3);
      // face the scene center: yaw such that forward = toward origin
      this.drone.yaw = Math.atan2(-(0 - this.camera.position.x), -(0 - this.camera.position.z));
      this.drone.pitch = -0.35;
    }
    if (m === 'fpv' && this.data && this.cal) {
      const W = this.data.worldSize;
      const gx = -W * 0.25, gz = -W * 0.25;
      const gy = sampleHeightWorld(this.cal.absolute, this.data.size, W, gx, gz) * this.exaggeration;
      this.camera.position.set(gx, gy + 1.7, gz);
      const dx = 0 - gx, dz = 0 - gz;
      this.fpv.yaw = Math.atan2(-dx, -dz);
      this.fpv.pitch = -0.05;
    }
    // analog FPV look follows the drone-mode lifecycle: overlay + vignette on
    // entry, everything hidden and the FOV restored on exit
    if (m === 'drone') this.showFpvOsd();
    else this.hideFpvOsd();
  }

  /** Show the Betaflight-style OSD + vignette on drone-mode entry. */
  private showFpvOsd(): void {
    if (this.osdWrap) this.osdWrap.style.display = 'block';
    this.fpvBattery = 100;
    this.fpvBreathT = 0;
    this.sizeOsd();
  }

  /** Hide the FPV overlay and restore the base FOV on drone-mode exit. */
  private hideFpvOsd(): void {
    if (this.osdWrap) this.osdWrap.style.display = 'none';
    this.camera.fov = this.fpvBaseFov;
    this.camera.updateProjectionMatrix();
  }

  /** Match the OSD canvas to the renderer size (device pixels). */
  private sizeOsd(): void {
    if (!this.osdCanvas) return;
    const pr = this.renderer.getPixelRatio();
    const bw = Math.max(2, Math.round((this.container.clientWidth || 800) * pr));
    const bh = Math.max(2, Math.round((this.container.clientHeight || 600) * pr));
    if (this.osdCanvas.width !== bw || this.osdCanvas.height !== bh) {
      this.osdCanvas.width = bw;
      this.osdCanvas.height = bh;
    }
  }

  /** Per-frame analog FPV look: FOV breathing + OSD redraw (drone mode only). */
  private updateFpv(dt: number): void {
    if (!this.osdCtx || !this.osdCanvas) return;
    // slow battery drain while "flying" + subtle FOV breathing that widens
    // slightly with speed (cheap screen-space analog-camera feel, no deps)
    this.fpvBattery = Math.max(0, this.fpvBattery - dt * 0.25);
    this.fpvBreathT += dt;
    const spdN = Math.min(1, this.currentSpeed() / 160);
    this.camera.fov = this.fpvBaseFov + Math.sin(this.fpvBreathT * 1.9) * 0.45 + spdN * 2.0;
    this.camera.updateProjectionMatrix();
    this.sizeOsd();
    const pr = this.renderer.getPixelRatio();
    const ctx = this.osdCtx;
    ctx.setTransform(pr, 0, 0, pr, 0, 0);
    this.drawOsd(ctx, this.osdCanvas.width / pr, this.osdCanvas.height / pr);
  }

  /** Camera height above the terrain in meters (0 when no scene loaded). */
  private fpvAgl(): number {
    const p = this.camera.position;
    if (!this.data) return Math.max(0, p.y);
    const W = this.data.worldSize, half = W / 2;
    const gx = Math.max(-half, Math.min(half, p.x));
    const gz = Math.max(-half, Math.min(half, p.z));
    return Math.max(0, p.y - this.heightAt(gx, gz));
  }

  /**
   * Betaflight-style analog OSD, drawn in CSS pixels: center crosshair,
   * roll-tilted artificial horizon with a pitch ladder, right-hand altitude
   * sidebar, ARMED/mode/speed readout, link-quality bars and a battery glyph,
   * plus faint analog static streaks. Attitude comes from the real camera
   * quaternion, so it tracks both manual and cinematic drone flight.
   */
  private drawOsd(ctx: CanvasRenderingContext2D, w: number, h: number): void {
    ctx.clearRect(0, 0, w, h);
    const q = this.camera.quaternion;
    const fwd = this.osdFwd.set(0, 0, -1).applyQuaternion(q);
    const right = this.osdRight.set(1, 0, 0).applyQuaternion(q);
    const cl = THREE.MathUtils.clamp;
    const pitchDeg = THREE.MathUtils.radToDeg(Math.asin(cl(fwd.y, -1, 1))); // + = nose up
    const rollDeg = -THREE.MathUtils.radToDeg(Math.asin(cl(right.y, -1, 1))); // + = rolled right
    const rollRad = THREE.MathUtils.degToRad(rollDeg);
    const cx = w / 2, cy = h / 2;
    const PX_PER_DEG = Math.max(3, h / 160); // horizon sensitivity scales with viewport
    const fg = 'rgba(245,245,245,0.92)';
    const dim = 'rgba(245,245,245,0.55)';
    const faint = 'rgba(245,245,245,0.18)';
    const warn = 'rgba(255,82,82,0.95)';
    const FONT = 'ui-monospace, Menlo, monospace';

    // ---- artificial horizon: rotated by roll, scrolled by pitch, ladder ticks
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-rollRad);
    const ladMax = Math.ceil((h / 2 - 26) / PX_PER_DEG / 10) * 10 + 10;
    ctx.lineWidth = 2;
    for (let d = -ladMax; d <= ladMax; d += 10) {
      const y = (pitchDeg - d) * PX_PER_DEG;
      if (Math.abs(y) > h / 2 - 26) continue;
      const isZero = d === 0;
      ctx.strokeStyle = isZero ? fg : dim;
      const halfLen = isZero ? Math.max(140, w / 2 - 80) : 30;
      ctx.beginPath();
      ctx.moveTo(-halfLen, y); ctx.lineTo(-12, y);
      ctx.moveTo(12, y); ctx.lineTo(halfLen, y);
      ctx.stroke();
      if (!isZero) {
        ctx.fillStyle = dim;
        ctx.font = `600 11px ${FONT}`;
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(String(Math.abs(d)), 36, y);
        ctx.textAlign = 'right';
        ctx.fillText(String(Math.abs(d)), -36, y);
      }
    }
    ctx.restore();

    // ---- center crosshair
    ctx.strokeStyle = fg;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx - 26, cy); ctx.lineTo(cx - 7, cy);
    ctx.moveTo(cx + 7, cy); ctx.lineTo(cx + 26, cy);
    ctx.moveTo(cx, cy - 26); ctx.lineTo(cx, cy - 7);
    ctx.moveTo(cx, cy + 7); ctx.lineTo(cx, cy + 26);
    ctx.stroke();
    ctx.fillStyle = fg;
    ctx.fillRect(cx - 1.5, cy - 1.5, 3, 3);

    // ---- altitude sidebar (right): AGL scale + current-altitude box
    const agl = this.fpvAgl();
    const sx = w - 56;
    const sy0 = cy - 120, sy1 = cy + 120;
    const PX_PER_M = 2.2;
    ctx.strokeStyle = dim;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy0); ctx.lineTo(sx, sy1);
    ctx.stroke();
    const step = 25;
    const aLo = Math.floor((agl - (sy1 - cy) / PX_PER_M) / step) * step;
    const aHi = agl + (cy - sy0) / PX_PER_M;
    ctx.font = `600 11px ${FONT}`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let a = aLo; a <= aHi; a += step) {
      if (a < 0) continue;
      const y = cy + (agl - a) * PX_PER_M;
      if (y < sy0 || y > sy1) continue;
      ctx.beginPath();
      ctx.moveTo(sx - 8, y); ctx.lineTo(sx, y);
      ctx.stroke();
      ctx.fillStyle = dim;
      ctx.fillText(String(a), sx + 6, y);
    }
    const altLabel = Math.max(0, agl).toFixed(0) + 'm';
    ctx.font = `700 14px ${FONT}`;
    const tw = ctx.measureText(altLabel).width + 14;
    ctx.fillStyle = 'rgba(5,10,20,0.72)';
    ctx.fillRect(sx - tw - 10, cy - 12, tw, 24);
    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(sx - tw - 10, cy - 12, tw, 24);
    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.fillText(altLabel, sx - 10 - tw / 2, cy + 1);
    ctx.fillStyle = dim;
    ctx.font = `600 10px ${FONT}`;
    ctx.fillText('ALT', sx, sy0 - 14);

    // ---- top-left: ARMED / mode / speed
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.font = `700 14px ${FONT}`;
    ctx.fillStyle = fg;
    ctx.fillText('● ARMED', 18, 30);
    ctx.font = `600 12px ${FONT}`;
    ctx.fillStyle = dim;
    ctx.fillText(this.cinematic ? 'DRONE · CINEMA' : 'DRONE · ACRO', 18, 50);
    ctx.fillStyle = fg;
    ctx.fillText('SPD ' + this.currentSpeed().toFixed(0) + ' m/s', 18, 70);

    // ---- top-right: link-quality bars + battery glyph
    let lq = 100;
    if (this.data) {
      const d = Math.hypot(this.camera.position.x, this.camera.position.z) / (this.data.worldSize * 0.7);
      lq = Math.max(8, 100 - d * 70);
    }
    lq = Math.max(5, Math.min(100, lq + (Math.random() - 0.5) * 10)); // analog flicker
    const bars = Math.ceil(lq / 25);
    const bx = w - 156;
    ctx.fillStyle = fg;
    ctx.font = `600 12px ${FONT}`;
    ctx.textAlign = 'right';
    ctx.fillText('LQ', bx - 8, 30);
    for (let i = 0; i < 4; i++) {
      const bh2 = 8 + i * 5;
      ctx.fillStyle = i < bars ? fg : faint;
      ctx.fillRect(bx + i * 12, 34 - bh2, 9, bh2);
    }
    const batt = this.fpvBattery;
    const battLow = batt < 20;
    const batX = w - 100, batY = 18, batW = 46, batH = 18;
    ctx.strokeStyle = battLow ? warn : fg;
    ctx.lineWidth = 2;
    ctx.strokeRect(batX, batY, batW, batH);
    ctx.fillStyle = battLow ? warn : fg;
    ctx.fillRect(batX + 3, batY + 3, (batW - 6) * batt / 100, batH - 6);
    ctx.fillRect(batX + batW, batY + 5, 4, batH - 10); // cap nub
    const blinkOn = !battLow || Math.floor(this.fpvBreathT * 3) % 2 === 0;
    ctx.fillStyle = battLow && blinkOn ? warn : fg;
    ctx.textAlign = 'left';
    ctx.fillText(batt.toFixed(0) + '%', batX + batW + 12, 32);

    // ---- bottom-center band label + faint analog static streaks
    ctx.fillStyle = dim;
    ctx.textAlign = 'center';
    ctx.font = `600 11px ${FONT}`;
    ctx.fillText('ANALOG · 5.8G', cx, h - 18);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    for (let i = 0; i < 5; i++) {
      ctx.fillRect(0, Math.random() * h, w, 1 + Math.random() * 2);
    }
  }

  setCinematic(on: boolean): void {
    this.cinematic = on && this.mode === 'drone' && !!this.cineCurve;
    if (this.cinematic) this.cineT = 0;
  }
  isCinematic(): boolean { return this.cinematic; }

  /**
   * Auto-orbit kiosk/demo mode: slow constant azimuth rotation around the
   * current orbit target. Enabling switches to orbit mode (cancelling any
   * fly-to in progress) and keeps the existing orbit target. Any user input
   * (canvas drag/wheel via OrbitControls, keyboard) pauses the spin via
   * onAutoOrbit(false) until re-toggled with setAutoOrbit(true).
   */
  setAutoOrbit(on: boolean): void {
    if (on) this.setMode('orbit'); // exits cinematic/drone/fpv, keeps the orbit target
    if (this.autoOrbitOn === on) return;
    this.autoOrbitOn = on;
    this.onAutoOrbit(on);
  }
  isAutoOrbitOn(): boolean { return this.autoOrbitOn; }

  /** Pause the kiosk spin after user input; re-enable with setAutoOrbit(true). */
  private pauseAutoOrbit(): void {
    if (!this.autoOrbitOn) return;
    this.autoOrbitOn = false;
    this.onAutoOrbit(false);
  }

  /** One frame of auto-orbit: rotate the camera around the orbit target's vertical axis. */
  private spinOrbit(dt: number, k: number): void {
    const angle = this.autoOrbitSpeed * k * dt;
    const tgt = this.orbit.target;
    const dx = this.camera.position.x - tgt.x;
    const dz = this.camera.position.z - tgt.z;
    const cosA = Math.cos(angle), sinA = Math.sin(angle);
    this.camera.position.x = tgt.x + dx * cosA - dz * sinA;
    this.camera.position.z = tgt.z + dx * sinA + dz * cosA;
    // orbit.update() (called right after) re-aims the camera at the target
  }

  setOverlay(o: Overlay): void { this.terrainUniforms.uOverlay.value = o; }

  /**
   * Excess Green (ExG) vegetation-index overlay toggle: the toolbar calls this
   * (e.g. from a "crop health" button). On = shader-side ExG heatmap
   * (overlay slot 4), off = back to the plain optical texture.
   */
  setVegetationIndex(on: boolean): void {
    this.terrainUniforms.uOverlay.value = on ? 4 : 0;
  }
  isVegetationIndexOn(): boolean { return this.terrainUniforms.uOverlay.value === 4; }

  /**
   * Legend rows for the ExG overlay, mirroring the land-cover legend:
   * Healthy / Moderate / Stressed bands with per-scene coverage %.
   * Null before a scene loads.
   */
  vegIndexLegend(): { name: string; color: string; pct: number }[] | null {
    if (!this.vegBands) return null;
    const b = this.vegBands;
    const pcts = [b.healthy * 100, b.moderate * 100, b.stressed * 100];
    return VEG_BAND_NAMES.map((name, i) => ({ name, color: VEG_BAND_COLORS[i], pct: pcts[i] }));
  }

  /**
   * NVG night-vision sensor mode: green-phosphor amplification render of the
   * scene. Mutually exclusive with the thermal mode (one sensor at a time).
   */
  setNightVision(on: boolean): void {
    this.terrainUniforms.uNight.value = on ? 1 : 0;
    if (on) this.terrainUniforms.uThermal.value = 0;
  }
  isNightVisionOn(): boolean { return this.terrainUniforms.uNight.value === 1; }

  /**
   * FLIR thermal sensor mode: ironbow gradient shaded from the DSM height.
   * Mutually exclusive with the night-vision mode.
   */
  setThermal(on: boolean): void {
    this.terrainUniforms.uThermal.value = on ? 1 : 0;
    if (on) this.terrainUniforms.uNight.value = 0;
  }
  isThermalOn(): boolean { return this.terrainUniforms.uThermal.value === 1; }

  /**
   * Legend rows for the thermal ironbow scale: per-stop elevation labels
   * with the share of the DSM in each band. Mirrors vegIndexLegend().
   * Null before a scene loads.
   */
  thermalLegend(): { name: string; color: string; pct: number }[] | null {
    if (!this.data || !this.cal) return null;
    const cal = this.cal;
    const span = Math.max(cal.maxH - cal.minH, 1e-6);
    // band i spans the quarter-interval around stop i (edge bands run to 0/1)
    const counts = new Array(THERMAL_BAND_STOPS.length).fill(0);
    const stride = Math.max(1, Math.floor(cal.absolute.length / 20000));
    let n = 0;
    for (let k = 0; k < cal.absolute.length; k += stride) {
      const t = Math.min(1, Math.max(0, (cal.absolute[k] - cal.minH) / span));
      const b = Math.min(counts.length - 1, Math.floor(t * counts.length));
      counts[b]++; n++;
    }
    const c = new THREE.Color();
    return THERMAL_BAND_NAMES.map((name, i) => ({
      name: `${name} · ${(cal.minH + THERMAL_BAND_STOPS[i] * span).toFixed(0)} m`,
      color: '#' + ironbowColor(THERMAL_BAND_STOPS[i], c).getHexString(),
      pct: n > 0 ? (counts[i] / n) * 100 : 0,
    }));
  }
  setContours(on: boolean): void { this.terrainUniforms.uContours.value = on ? 1 : 0; }
  /** Survey scan-wave + draped grid overlay. */
  setScan(on: boolean): void {
    this.scanOn = on;
    this.terrainUniforms.uScanOn.value = on ? 1 : 0;
    if (this.surveyGrid) this.surveyGrid.visible = on;
  }
  isScanOn(): boolean { return this.scanOn; }

  // ----------------------------------------------------- point cloud ---
  /** LiDAR-style point-cloud layer: hypsometric-colored THREE.Points over the mesh. */
  setPoints(on: boolean): void {
    this.pointsOn = on;
    if (on && !this.points && this.terrain) this.rebuildPoints();
    if (this.points) this.points.visible = on;
  }
  isPointsOn(): boolean { return this.pointsOn; }

  private rebuildPoints(): void {
    if (this.points) {
      this.scene.remove(this.points);
      disposeGroup(this.points as unknown as THREE.Group);
      this.points = null;
    }
    if (!this.terrain || !this.data) return;
    const pos = this.terrain.geometry.getAttribute('position') as THREE.BufferAttribute;
    const n = pos.count;
    const colors = new Float32Array(n * 3);
    const minH = this.terrainUniforms.uMinH.value, maxH = this.terrainUniforms.uMaxH.value;
    const span = Math.max(maxH - minH, 1e-6);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const t = Math.min(1, Math.max(0, (pos.getY(i) - minH) / span));
      hypsoColor(t, c);
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    const geo = new THREE.BufferGeometry();
    const p2 = new THREE.Float32BufferAttribute(pos.array.slice(), 3);
    geo.setAttribute('position', p2);
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const lift = Math.max(0.4, this.data.worldSize * 0.0009);
    for (let i = 0; i < n; i++) p2.setY(i, p2.getY(i) + lift);
    const mat = new THREE.PointsMaterial({
      size: Math.max(0.8, this.data.worldSize / 512),
      vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.95,
    });
    this.points = new THREE.Points(geo, mat);
    this.points.visible = this.pointsOn;
    this.points.renderOrder = 4;
    this.scene.add(this.points);
  }

  // ------------------------------------------------------ flooding ---
  /** Flood simulation: frac 0..1 of scene relief as water level, <0 disables. */
  setFlood(frac: number): void {
    this.floodFrac = frac;
    if (frac < 0 || !this.data || !this.cal) {
      this.terrainUniforms.uFloodY.value = -1e9;
      if (this.floodPlane) this.floodPlane.visible = false;
      return;
    }
    const minY = this.cal.minH * this.exaggeration;
    const maxY = this.cal.maxH * this.exaggeration;
    const y = minY + frac * (maxY - minY);
    this.terrainUniforms.uFloodY.value = y;
    if (!this.floodPlane) {
      const W = this.data.worldSize;
      const geo = new THREE.PlaneGeometry(W * 1.04, W * 1.04);
      const mat = new THREE.MeshStandardMaterial({
        color: 0x2f8fd6, transparent: true, opacity: 0.42,
        roughness: 0.12, metalness: 0.05, side: THREE.DoubleSide,
      });
      this.floodPlane = new THREE.Mesh(geo, mat);
      this.floodPlane.rotation.x = -Math.PI / 2;
      this.floodPlane.renderOrder = 5;
      this.scene.add(this.floodPlane);
    }
    this.floodPlane.visible = true;
    this.floodPlane.position.y = y + 0.5;
  }
  /** Current water level in meters (non-exaggerated), NaN when off. */
  floodLevelM(): number {
    if (this.floodFrac < 0 || !this.cal) return NaN;
    return this.cal.minH + this.floodFrac * (this.cal.maxH - this.cal.minH);
  }

  // ---------------------------------------------------- viewshed ---
  /** Modal click-to-place mode for the viewshed observer (next canvas click places it). */
  setViewshedPlace(on: boolean): void {
    this.viewshedPlace = on;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
  }
  isViewshedPlaceMode(): boolean { return this.viewshedPlace; }
  /** Tower/mast height (m) added to the terrain at the observer point; recomputes live. */
  setViewshedTower(h: number): void {
    this.viewshedTowerH = Math.max(0, Math.min(500, h));
    if (this.viewshedActive) {
      this.recomputeViewshed();
      this.rebuildObserverMarker();
    }
  }
  viewshedTower(): number { return this.viewshedTowerH; }
  isViewshedActive(): boolean { return this.viewshedActive; }

  /** Turn the viewshed overlay off and clear the observer. */
  clearViewshed(): void {
    this.viewshedActive = false;
    this.viewshedPlace = false;
    this.renderer.domElement.style.cursor = '';
    this.terrainUniforms.uViewshedOn.value = 0;
    this.terrainUniforms.uViewshedTex.value = null;
    if (this.viewshedTex) { this.viewshedTex.dispose(); this.viewshedTex = null; }
    if (this.observerMarker) this.observerMarker.visible = false;
    this.onViewshed(null);
  }

  private placeViewshed(cx: number, cy: number): void {
    if (!this.terrain || !this.data || !this.cal) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hit = this.raycaster.intersectObject(this.terrain)[0];
    if (!hit) return; // keep placement mode active until a valid click lands
    this.viewshedObs.x = hit.point.x;
    this.viewshedObs.z = hit.point.z;
    this.viewshedObs.groundM = hit.point.y / this.exaggeration;
    this.recomputeViewshed();
    this.rebuildObserverMarker();
    this.viewshedActive = true;
    this.setViewshedPlace(false); // one-shot placement: click places the observer, then exits
  }

  /** Recompute the 360° visibility grid over the unexaggerated DSM and re-upload the mask. */
  private recomputeViewshed(): void {
    if (!this.data || !this.cal) return;
    const n = this.data.size, W = this.data.worldSize;
    const o = this.viewshedObs;
    const eyeH = o.groundM + this.viewshedTowerH;
    const res = computeViewshed(this.cal.absolute, n, W, o.x, o.z, eyeH);
    // mesh uv.y = 1 - row/(n-1); DataTexture row 0 sits at v=0 -> flip rows into the texture
    const texData = new Uint8Array(n * n);
    for (let row = 0; row < n; row++)
      for (let col = 0; col < n; col++)
        texData[(n - 1 - row) * n + col] = res.vis[row * n + col] ? 255 : 0;
    if (this.viewshedTex) this.viewshedTex.dispose();
    const t = new THREE.DataTexture(texData, n, n, THREE.RedFormat, THREE.UnsignedByteType);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    this.viewshedTex = t;
    this.terrainUniforms.uViewshedTex.value = t;
    this.terrainUniforms.uViewshedOn.value = 1;
    this.onViewshed({ x: o.x, z: o.z, groundM: o.groundM, towerH: this.viewshedTowerH, pct: res.pct });
  }

  /** Observer marker: ground ring + mast pole (tower height) + eye point. */
  private rebuildObserverMarker(): void {
    if (!this.data) return;
    if (!this.observerMarker) {
      this.observerMarker = new THREE.Group();
      this.observerMarker.renderOrder = 999;
      this.scene.add(this.observerMarker);
    }
    const g = this.observerMarker;
    disposeGroup(g);
    g.clear();
    const W = this.data.worldSize;
    const o = this.viewshedObs;
    const gy = o.groundM * this.exaggeration;
    const ey = (o.groundM + this.viewshedTowerH) * this.exaggeration;
    const ringR = Math.max(3, W * 0.006);
    const orange = new THREE.MeshBasicMaterial({ color: 0xff9f1a, depthTest: false, transparent: true });
    const ring = new THREE.Mesh(new THREE.RingGeometry(ringR, ringR * 1.45, 40), orange);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(o.x, gy + 1, o.z);
    const poleH = Math.max(ey - gy, 0.1);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(ringR * 0.14, ringR * 0.14, poleH, 10), orange);
    pole.position.set(o.x, gy + poleH / 2, o.z);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(ringR * 0.55, 16, 12), orange);
    eye.position.set(o.x, ey, o.z);
    g.add(ring, pole, eye);
    g.visible = true;
  }

  private rebuildSurveyGrid(): void {
    if (this.surveyGrid) {
      this.scene.remove(this.surveyGrid);
      disposeGroup(this.surveyGrid);
      this.surveyGrid = null;
    }
    if (!this.data) return;
    this.surveyGrid = buildSurveyGrid(this.data, (x, z) => this.heightAt(x, z));
    this.surveyGrid.visible = this.scanOn;
    this.scene.add(this.surveyGrid);
  }
  /** Compare swipe: x in [0,1] = split fraction, <0 disables. */
  setCompare(x: number): void { this.terrainUniforms.uCompareX.value = x; }

  /**
   * Elevation slice / clip-plane inspection (Cesium 'slicing' analysis
   * pattern): discard the terrain on one side of a vertical plane through the
   * scene so the mesh renders split at the plane, with a glowing edge at the
   * cut so it reads as a cross-section wall.
   *
   * axis 'x' puts the plane at world x (east–west slice), 'y' at world z
   * (north–south slice). frac is 0..1 across the scene extent. Disabled by
   * default and off in clean-view; clearClipPlane() turns it back off.
   */
  setClipPlane(axis: 'x' | 'y', frac: number): void {
    const U = this.terrainUniforms;
    U.uClipOn.value = 1;
    U.uClipAxis.value = axis === 'x' ? 0 : 1;
    U.uClipFrac.value = Math.min(1, Math.max(0, frac));
  }
  clearClipPlane(): void { this.terrainUniforms.uClipOn.value = 0; }
  isClipPlaneOn(): boolean { return this.terrainUniforms.uClipOn.value === 1; }
  /** Current slice state for the UI (slider position + axis toggle). */
  clipPlaneState(): { axis: 'x' | 'y'; frac: number } {
    return {
      axis: this.terrainUniforms.uClipAxis.value < 0.5 ? 'x' : 'y',
      frac: this.terrainUniforms.uClipFrac.value,
    };
  }
  /**
   * Upload a post-event DSM (packed row-flipped via packChangeTexture) as the
   * change-detection texture. Float format, nearest-sampled (R32F is not
   * filterable without OES_texture_float_linear).
   */
  setChangeTexture(data: Float32Array, size: number): void {
    if (this.changeTex) this.changeTex.dispose();
    const t = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.FloatType);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    this.changeTex = t;
    this.terrainUniforms.uChangeTex.value = t;
    this.terrainUniforms.uChangeExagg.value = this.exaggeration;
  }
  /**
   * Change-detection compare: reuses the compare split-shader — left = baseline
   * DSM hypsometric, right = post-event DSM hypsometric (from uChangeTex).
   * Turning off also kills the split so a stale uCompareX can't sample a dead texture.
   */
  setChangeDetection(on: boolean): void {
    this.terrainUniforms.uChangeMode.value = on ? 1 : 0;
    if (!on) this.terrainUniforms.uCompareX.value = -1;
  }
  isChangeDetectionOn(): boolean { return this.terrainUniforms.uChangeMode.value === 1; }
  /** Sun position for shadow analysis: azimuth deg (0=N..360), elevation deg (8..80). */
  setSun(azDeg: number, elDeg: number): void {
    const sun = (this as { _sun?: THREE.DirectionalLight })._sun;
    if (!sun) return;
    const hemi = (this as { _hemi?: THREE.HemisphereLight })._hemi;
    const skyU = (this as { _skyU?: { [k: string]: { value: THREE.Color } } })._skyU;
    const elR = THREE.MathUtils.degToRad(elDeg), azR = THREE.MathUtils.degToRad(azDeg);
    const R = 2400;
    sun.position.set(
      Math.cos(elR) * Math.cos(azR) * R,
      Math.sin(elR) * R,
      Math.cos(elR) * Math.sin(azR) * R,
    );
    const t = Math.min(1, Math.max(0, (elDeg - 8) / 52)); // 0 = low sun, 1 = high sun
    sun.intensity = 0.45 + 1.65 * t;
    sun.color.setHex(0xff9a3c).lerp(new THREE.Color(0xfff2dd), t);
    if (hemi) hemi.intensity = 0.25 + 0.55 * t;
    if (skyU) {
      skyU.top.value.setHex(0x27406e).lerp(new THREE.Color(0x2f6fd0), t);
      skyU.mid.value.setHex(0xe8a06a).lerp(new THREE.Color(0x9cc8ee), t);
      skyU.bot.value.setHex(0x5a4a52).lerp(new THREE.Color(0xe8f2f8), t);
    }
    this.renderer.toneMappingExposure = 0.92 + 0.13 * t;
  }
  setExaggeration(x: number): void {
    this.exaggeration = x;
    this.clearMeasures();
    if (this.terrain) {
      applyExaggeration(this.terrain.geometry, x);
      if (this.cal) {
        this.terrainUniforms.uMinH.value = this.cal.minH * x;
        this.terrainUniforms.uMaxH.value = this.cal.maxH * x;
      }
      this.terrainUniforms.uChangeExagg.value = x;
      const base = (this as { _base?: THREE.Mesh })._base;
      if (base && this.cal) base.position.y = this.cal.minH * x - 22;
      this.rebuildSurveyGrid();
      if (this.pointsOn) this.rebuildPoints();
      if (this.floodFrac >= 0) this.setFlood(this.floodFrac);
      if (this.viewshedActive) this.rebuildObserverMarker(); // viewshed mask is exaggeration-independent
      this.buildMinimapCache();
    }
  }

  enableProbe(on: boolean): void {
    this.probeEnabled = on;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
    if (!on) { this.probeMarker.visible = false; this.onProbe(null); }
  }

  // ---------------------------------------------------------- measuring ---
  setMeasure(on: boolean): void {
    this.measureOn = on;
    this.renderer.domElement.style.cursor = on ? 'crosshair' : '';
    if (!on) this.cancelMeasure();
  }
  isMeasureOn(): boolean { return this.measureOn; }

  /** Switch the measure tool between distance and polygon-volume modes. */
  setMeasureMode(m: 'distance' | 'volume'): void { this.measureMode = m; this.cancelMeasure(); }
  isVolumeMode(): boolean { return this.measureMode === 'volume'; }

  private measureClick(cx: number, cy: number): void {
    if (!this.terrain || !this.data) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hit = this.raycaster.intersectObject(this.terrain)[0];
    if (!hit) return;
    this.measurePts.push(hit.point.clone());
    this.rebuildMeasureLive();
    this.onMeasure();
  }

  /** Finish the in-progress measurement (double-click or panel button). */
  finishMeasure(): void {
    if (this.measurePts.length < 2 || !this.data) { this.cancelMeasure(); return; }
    const colors = [0x22d3ee, 0xfbbf24, 0x34d399, 0xf472b6, 0xa78bfa];
    const color = colors[this.measureDone.length % colors.length];
    const cal = this.cal;
    const isVol = this.measureMode === 'volume' && this.measurePts.length >= 3 && !!cal;
    let group: THREE.Group;
    let kind: 'distance' | 'volume' = 'distance';
    let total = '';
    if (isVol && cal) {
      kind = 'volume';
      const r = calcVolume(
        this.measurePts.map(p => ({ x: p.x, z: p.z })),
        cal.absolute,                                    // metric meters (unexaggerated)
        this.data.worldSize / (this.data.size - 1),      // gsdMeters
      );
      total = fmtVolume(r);
      group = this.buildMeasureGroup(this.measurePts, color, this.data.worldSize, { close: true, totalLabel: total });
    } else {
      group = this.buildMeasureGroup(this.measurePts, color, this.data.worldSize);
    }
    this.scene.add(group);
    this.measureDone.push({ pts: this.measurePts, group, color, kind, total });
    this.measurePts = [];
    this.rebuildMeasureLive();
    this.onMeasure();
  }

  cancelMeasure(): void {
    this.measurePts = [];
    this.rebuildMeasureLive();
    this.onMeasure();
  }

  deleteMeasure(i: number): void {
    const m = this.measureDone[i];
    if (!m) return;
    this.scene.remove(m.group);
    disposeGroup(m.group);
    this.measureDone.splice(i, 1);
    this.onMeasure();
  }

  clearMeasures(): void {
    for (const m of this.measureDone) { this.scene.remove(m.group); disposeGroup(m.group); }
    this.measureDone = [];
    this.measurePts = [];
    this.rebuildMeasureLive();
    this.onMeasure();
  }

  /**
   * Render one frame immediately and return it as a PNG data URL.
   * Renders synchronously first, so this works correctly even with
   * preserveDrawingBuffer=false (the drawing buffer is fresh in this task).
   */
  captureFrame(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  /** Unexaggerated DSM grid for geo consumers (profile traverse), or null before a scene loads. */
  getDSM(): { heights: Float32Array; size: number; gsdMeters: number } | null {
    if (!this.data || !this.cal) return null;
    return { heights: this.cal.absolute, size: this.data.size, gsdMeters: this.data.worldSize / (this.data.size - 1) };
  }

  /** World x/z points of the most recently finished measurement, for the profile tool. */
  lastMeasurePts(): { x: number; z: number }[] | null {
    const m = this.measureDone[this.measureDone.length - 1];
    return m ? m.pts.map((p) => ({ x: p.x, z: p.z })) : null;
  }

  /** 3D cursor pin synced with the elevation-profile chart scrubber. */
  setProfileMarker(x: number, z: number): void {
    if (!this.data) return;
    if (!this.profileMarker) {
      const r = Math.max(2, this.data.worldSize * 0.004);
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(r, 14, 10),
        new THREE.MeshBasicMaterial({ color: 0xfbbf24, depthTest: false, transparent: true, opacity: 0.95 }),
      );
      mesh.renderOrder = 999;
      this.scene.add(mesh);
      this.profileMarker = mesh;
    }
    this.profileMarker.position.set(x, this.heightAt(x, z) + 1, z);
    this.profileMarker.visible = true;
  }
  clearProfileMarker(): void {
    if (this.profileMarker) this.profileMarker.visible = false;
  }

  /** Snapshot for the HUD panel: totals, point counts, colors. */
  measureList(): { total: string; pts: number; color: string }[] {
    return this.measureDone.map((m) => {
      const color = '#' + m.color.toString(16).padStart(6, '0');
      if (m.kind === 'volume') return { total: m.total, pts: m.pts.length, color };
      let tot = 0;
      for (let i = 0; i < m.pts.length - 1; i++) tot += m.pts[i].distanceTo(m.pts[i + 1]);
      return { total: fmtDist(tot), pts: m.pts.length, color };
    });
  }

  private rebuildMeasureLive(): void {
    if (this.measureLive) {
      this.scene.remove(this.measureLive);
      disposeGroup(this.measureLive);
      this.measureLive = null;
    }
    if (this.measurePts.length === 0 || !this.data) return;
    this.measureLive = this.buildMeasureGroup(this.measurePts, 0x22d3ee, this.data.worldSize,
      this.measureMode === 'volume' && this.measurePts.length >= 3 ? { close: true } : undefined);
    this.scene.add(this.measureLive);
  }

  /** Draped polyline + point markers + distance sprite labels. close=true draws a closed
      polygon with a translucent draped fill (volume mode); totalLabel overrides the Σ distance label. */
  private buildMeasureGroup(pts: THREE.Vector3[], color: number, worldSize: number,
    opts?: { close?: boolean; totalLabel?: string }): THREE.Group {
    const g = new THREE.Group();
    const close = opts?.close ?? false;
    const lift = Math.max(1.5, worldSize * 0.0012);
    const markerR = Math.max(2.2, worldSize * 0.003);
    const centroid = (arr: THREE.Vector3[]) => ({
      x: arr.reduce((s, p) => s + p.x, 0) / arr.length,
      z: arr.reduce((s, p) => s + p.z, 0) / arr.length,
    });
    for (const p of pts) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(markerR, 12, 8),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }),
      );
      m.position.copy(p); m.position.y += lift * 0.5;
      m.renderOrder = 998;
      g.add(m);
    }
    const lp: number[] = [];
    const SEG = 24;
    const segCount = close ? pts.length : pts.length - 1; // closed loop in volume mode
    for (let i = 0; i < segCount; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      let prev: THREE.Vector3 | null = null;
      for (let s = 0; s <= SEG; s++) {
        const t = s / SEG;
        const x = a.x + (b.x - a.x) * t, z = a.z + (b.z - a.z) * t;
        const p = new THREE.Vector3(x, this.heightAt(x, z) + lift, z);
        if (prev) lp.push(prev.x, prev.y, prev.z, p.x, p.y, p.z);
        prev = p;
      }
      if (!close) {
        const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
        const lab = makeLabelSprite(fmtDist(a.distanceTo(b)), markerR * 4.4);
        lab.position.set(mid.x, this.heightAt(mid.x, mid.z) + lift + markerR * 2.4, mid.z);
        g.add(lab);
      }
    }
    if (lp.length > 0) {
      const lgeo = new THREE.BufferGeometry();
      lgeo.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
      const line = new THREE.LineSegments(lgeo,
        new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
      line.renderOrder = 997;
      g.add(line);
    }
    if (close && pts.length >= 3) {
      // translucent draped polygon fill: triangle fan around the centroid
      const c = centroid(pts);
      const cy = this.heightAt(c.x, c.z) + lift;
      const fp: number[] = [];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length];
        fp.push(c.x, cy, c.z,
          a.x, this.heightAt(a.x, a.z) + lift, a.z,
          b.x, this.heightAt(b.x, b.z) + lift, b.z);
      }
      const fgeo = new THREE.BufferGeometry();
      fgeo.setAttribute('position', new THREE.Float32BufferAttribute(fp, 3));
      const fill = new THREE.Mesh(fgeo, new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.25, depthWrite: false, side: THREE.DoubleSide,
      }));
      fill.renderOrder = 996;
      g.add(fill);
    }
    if (opts?.totalLabel) {
      const c = centroid(pts);
      const lab = makeLabelSprite(opts.totalLabel, markerR * 4.4, 'rgba(8,14,26,0.92)', '#eaf2ff', '#38bdf8');
      lab.position.set(c.x, this.heightAt(c.x, c.z) + lift + markerR * 7, c.z);
      g.add(lab);
    } else if (pts.length >= 2) {
      let tot = 0;
      for (let i = 0; i < pts.length - 1; i++) tot += pts[i].distanceTo(pts[i + 1]);
      const last = pts[pts.length - 1];
      const lab = makeLabelSprite('Σ ' + fmtDist(tot), markerR * 5.4, 'rgba(8,14,26,0.92)', '#eaf2ff', '#38bdf8');
      lab.position.set(last.x, this.heightAt(last.x, last.z) + lift + markerR * 7, last.z);
      g.add(lab);
    }
    return g;
  }

  private probe(cx: number, cy: number): void {
    if (!this.terrain || !this.data || !this.cal) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hit = this.raycaster.intersectObject(this.terrain)[0];
    if (!hit) return;
    const p = hit.point;
    this.probeMarker.visible = true;
    this.probeMarker.position.copy(p);
    // slope from face normal
    const nrm = hit.face ? hit.face.normal.clone().transformDirection(this.terrain.matrixWorld) : new THREE.Vector3(0, 1, 0);
    const slopeDeg = THREE.MathUtils.radToDeg(Math.acos(Math.max(-1, Math.min(1, nrm.y))));
    this.onProbe({
      x: p.x, y: p.y, z: p.z,
      elevation: p.y / this.exaggeration,
      slopeDeg,
    });
  }

  heightAt(x: number, z: number): number {
    if (!this.data || !this.cal) return 0;
    return sampleHeightWorld(this.cal.absolute, this.data.size, this.data.worldSize, x, z) * this.exaggeration;
  }

  /** Current vertical exaggeration factor (the render mesh's y-scale). */
  getExaggeration(): number { return this.exaggeration; }

  /** Spot-elevation placing mode: terrain clicks are forwarded to onSpotPlace. */
  setSpotPlace(on: boolean): void { this.spotPlace = on; }
  isSpotPlaceOn(): boolean { return this.spotPlace; }

  private spotPlaceClick(cx: number, cy: number): void {
    if (!this.terrain) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nd = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hit = this.raycaster.intersectObject(this.terrain)[0];
    if (!hit) return; // keep placing mode active until a valid click lands
    this.onSpotPlace(hit.point.x, hit.point.z);
  }

  /**
   * Project a world point to CSS-pixel coordinates over the canvas —
   * the cross-area hook for the spot-elevation label layer. Returns null
   * when the point is behind the camera (projection would flip).
   */
  projectToScreen(x: number, y: number, z: number): { x: number; y: number } | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const camSpace = new THREE.Vector3(x, y, z).applyMatrix4(this.camera.matrixWorldInverse);
    if (camSpace.z > -0.1) return null;
    const v = new THREE.Vector3(x, y, z).project(this.camera);
    return {
      x: (v.x * 0.5 + 0.5) * rect.width,
      y: (-v.y * 0.5 + 0.5) * rect.height,
    };
  }

  resize(): void {
    const w = this.container.clientWidth || 800, h = this.container.clientHeight || 600;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const sz = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.terrainUniforms.uViewportW.value = sz.x;
    this.sizeOsd();
  }

  // ------------------------------------------------------------ mini-map ---
  /** Rebuild the cached top-down hypsometric blit used by renderMinimap(). */
  private buildMinimapCache(): void {
    const S = 192;
    if (!this.minimapCache) this.minimapCache = document.createElement('canvas');
    const cv = this.minimapCache;
    cv.width = S; cv.height = S;
    const ctx = cv.getContext('2d')!;
    const data = this.data, cal = this.cal;
    if (!data || !cal) { ctx.clearRect(0, 0, S, S); return; }
    const n = data.size, W = data.worldSize, half = W / 2;
    const hgt = cal.absolute, minH = cal.minH;
    const span = Math.max(cal.maxH - cal.minH, 1e-6);
    const img = ctx.createImageData(S, S);
    const c = new THREE.Color();
    for (let j = 0; j < S; j++) {
      const z = -half + (j / (S - 1)) * W;
      for (let i = 0; i < S; i++) {
        const x = -half + (i / (S - 1)) * W;
        const t = Math.min(1, Math.max(0, (sampleHeightWorld(hgt, n, W, x, z) - minH) / span));
        hypsoColor(t, c);
        c.convertLinearToSRGB(); // linear shader colors -> 2D canvas sRGB
        const k = (j * S + i) * 4;
        img.data[k] = Math.round(c.r * 255);
        img.data[k + 1] = Math.round(c.g * 255);
        img.data[k + 2] = Math.round(c.b * 255);
        img.data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  /**
   * Draw the 2D overview inset into an external <canvas>: cached hypsometric
   * blit + live camera frustum wedge + north arrow. The UI layer supplies the
   * canvas (e.g. <canvas id=minimap>) and calls this from its update loop
   * throttled to ~10 Hz.
   */
  renderMinimap(ctx: CanvasRenderingContext2D): void {
    const cv = ctx.canvas, w = cv.width, h = cv.height;
    if (w <= 0 || h <= 0) return;
    if (!this.minimapCache) this.buildMinimapCache();
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    if (this.minimapCache) ctx.drawImage(this.minimapCache, 0, 0, w, h);
    if (!this.data) { ctx.restore(); return; }
    const W = this.data.worldSize, half = W / 2;
    const px = (x: number) => (x + half) / W * w;
    const pz = (z: number) => (z + half) / W * h;
    // terrain extent frame
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px(-half) + 0.5, pz(-half) + 0.5, px(half) - px(-half) - 1, pz(half) - pz(-half) - 1);
    // camera frustum wedge (top-down projection of the horizontal FOV)
    const p = this.camera.position;
    const cx = px(Math.max(-half, Math.min(half, p.x)));
    const cy = pz(Math.max(-half, Math.min(half, p.z)));
    this.camera.getWorldDirection(this.minimapDir);
    const ang = Math.atan2(this.minimapDir.x, -this.minimapDir.z); // 0 = north(-Z), clockwise
    const vFov = this.camera.fov * Math.PI / 180;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    const canvasAng = ang - Math.PI / 2; // canvas 0 rad = +x, clockwise-positive
    const wedgeLen = Math.max(w, h) * 0.55;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, wedgeLen, canvasAng - hFov / 2, canvasAng + hFov / 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(56,189,248,0.22)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(56,189,248,0.85)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // forward-direction tick
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.sin(ang) * wedgeLen * 0.7, cy - Math.cos(ang) * wedgeLen * 0.7);
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // camera position dot
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#f43f5e';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    // orbit target marker
    if (this.mode === 'orbit') {
      const t = this.orbit.target;
      ctx.beginPath();
      ctx.arc(px(t.x), pz(t.z), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fill();
    }
    // north arrow (scene north = -Z = map up)
    const nx = w - 16, ny = 18;
    ctx.beginPath();
    ctx.arc(nx, ny, 11, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(5,10,20,0.65)';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(nx, ny - 7); ctx.lineTo(nx + 4.5, ny + 4); ctx.lineTo(nx - 4.5, ny + 4);
    ctx.closePath();
    ctx.fillStyle = '#f8fafc';
    ctx.fill();
    ctx.font = '700 8px ui-monospace, Menlo, monospace';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('N', nx, ny + 5);
    ctx.restore();
  }

  /** Camera heading in degrees, 0 = looking toward -Z (scene north), clockwise. */
  private cameraHeadingDeg(): number {
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    let h = Math.atan2(dir.x, -dir.z) * 180 / Math.PI;
    if (h < 0) h += 360;
    return h;
  }

  /** Ground meters covered by 100 screen pixels at the current view distance. */
  private metersPer100px(): number {
    let d: number;
    if (this.mode === 'orbit') {
      d = this.camera.position.distanceTo(this.orbit.target);
    } else {
      d = Math.max(2, this.camera.position.y - (this.cal ? this.cal.minH * this.exaggeration : 0));
    }
    const hPx = this.container.clientHeight || 600;
    return 2 * d * Math.tan(this.camera.fov * Math.PI / 360) / hPx * 100;
  }

  // ----------------------------------------------------------------- loop ---
  tick(dt: number): void {
    // fps meter
    this.fpsAcc += dt; this.fpsN++; this.fpsT += dt;
    if (this.fpsT > 0.5) {
      this.onHud(this.fpsN / this.fpsAcc, this.camera.position.y, this.currentSpeed(),
        this.cameraHeadingDeg(), this.metersPer100px());
      this.fpsAcc = 0; this.fpsN = 0; this.fpsT = 0;
    }

    if (this.mode === 'orbit') {
      if (this.fly.active) {
        // fly-to animation (ease-in-out cubic)
        this.fly.t += dt / this.fly.dur;
        const tt = Math.min(1, this.fly.t);
        const k = tt < 0.5 ? 4 * tt * tt * tt : 1 - Math.pow(-2 * tt + 2, 3) / 2;
        this.camera.position.lerpVectors(this.fly.fromPos, this.fly.toPos, k);
        this.orbit.target.lerpVectors(this.fly.fromTgt, this.fly.toTgt, k);
        if (this.fly.t >= 1) { this.fly.active = false; this.orbit.enabled = true; }
        this.orbit.update();
      } else {
        // auto-orbit kiosk spin: eased spin-up/down (same ease-in-out cubic
        // curve as fly-to), constant azimuth rotation around the orbit target
        const aoTarget = this.autoOrbitOn ? 1 : 0;
        if (this.autoOrbitP < aoTarget) this.autoOrbitP = Math.min(aoTarget, this.autoOrbitP + dt / 1.6);
        else if (this.autoOrbitP > aoTarget) this.autoOrbitP = Math.max(aoTarget, this.autoOrbitP - dt / 1.6);
        if (this.autoOrbitP > 0.0005) {
          const p = this.autoOrbitP;
          const k = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
          this.spinOrbit(dt, k);
        }
        this.orbit.update();
      }
    } else if (this.mode === 'drone') {
      if (this.cinematic && this.cineCurve) {
        this.cineT = (this.cineT + dt * 0.014) % 1;
        const pos = this.cineCurve.getPointAt(this.cineT);
        const ahead = this.cineCurve.getPointAt((this.cineT + 0.025) % 1);
        this.camera.position.copy(pos);
        this.camera.lookAt(ahead.x, ahead.y - 40, ahead.z);
      } else {
        const sp = this.drone.speed * (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 2.6 : 1);
        const dir = new THREE.Vector3(
          -Math.sin(this.drone.yaw) * Math.cos(this.drone.pitch),
          Math.sin(this.drone.pitch),
          -Math.cos(this.drone.yaw) * Math.cos(this.drone.pitch),
        );
        const right = new THREE.Vector3(-Math.cos(this.drone.yaw), 0, Math.sin(this.drone.yaw));
        const mv = new THREE.Vector3();
        if (this.keys.has('KeyW')) mv.add(dir);
        if (this.keys.has('KeyS')) mv.sub(dir);
        if (this.keys.has('KeyD')) mv.add(right);
        if (this.keys.has('KeyA')) mv.sub(right);
        if (this.keys.has('KeyR') || this.keys.has('Space')) mv.y += 1;
        if (this.keys.has('KeyF')) mv.y -= 1;
        if (mv.lengthSq() > 0) mv.normalize();
        this.camera.position.addScaledVector(mv, sp * dt);
        // keep above terrain & inside world
        if (this.data) {
          const W = this.data.worldSize, half = W / 2;
          const p = this.camera.position;
          p.x = Math.max(-half * 1.6, Math.min(half * 1.6, p.x));
          p.z = Math.max(-half * 1.6, Math.min(half * 1.6, p.z));
          const minY = this.heightAt(Math.max(-half, Math.min(half, p.x)), Math.max(-half, Math.min(half, p.z))) + 4;
          if (p.y < minY) p.y = minY;
          if (p.y > 2500) p.y = 2500;
        }
        this.camera.quaternion.setFromEuler(new THREE.Euler(0, this.drone.yaw, 0, 'YXZ'));
        this.camera.rotateX(this.drone.pitch);
      }
      this.updateFpv(dt); // analog FPV look: OSD + vignette + FOV breathing
    } else { // fpv
      const sp = 9 * (this.keys.has('ShiftLeft') ? 2.4 : 1);
      const fwd = new THREE.Vector3(-Math.sin(this.fpv.yaw), 0, -Math.cos(this.fpv.yaw));
      const right = new THREE.Vector3(-Math.cos(this.fpv.yaw), 0, Math.sin(this.fpv.yaw));
      const mv = new THREE.Vector3();
      if (this.keys.has('KeyW')) mv.add(fwd);
      if (this.keys.has('KeyS')) mv.sub(fwd);
      if (this.keys.has('KeyD')) mv.add(right);
      if (this.keys.has('KeyA')) mv.sub(right);
      if (mv.lengthSq() > 0) mv.normalize().multiplyScalar(sp * dt);
      const p = this.camera.position;
      p.add(mv);
      if (this.data) {
        const W = this.data.worldSize, half = W / 2 - 4;
        p.x = Math.max(-half, Math.min(half, p.x));
        p.z = Math.max(-half, Math.min(half, p.z));
        p.y = this.heightAt(p.x, p.z) + 1.7;
      }
      this.camera.quaternion.setFromEuler(new THREE.Euler(0, this.fpv.yaw, 0, 'YXZ'));
      this.camera.rotateX(this.fpv.pitch);
    }

    // survey scan-wave animation: expanding sonar ring + rotating radar wedge
    if (this.scanOn && this.data) {
      this.scanT += dt;
      const maxR = this.data.worldSize * 0.75;
      this.terrainUniforms.uScanR.value = (this.scanT * 260) % (maxR + 120);
      this.terrainUniforms.uScanA.value = this.scanT * 1.1;
    }

    this.renderer.render(this.scene, this.camera);
  }

  private currentSpeed(): number {
    if (this.mode === 'drone' && !this.cinematic) return this.drone.speed;
    if (this.cinematic) return 34;
    return 0;
  }

  dispose(): void {
    this.orbit.dispose();
    this.renderer.dispose();
    if (this.changeTex) { this.changeTex.dispose(); this.changeTex = null; }
    this.container.innerHTML = '';
  }
}

// ------------------------------------------------------------- helpers ---

/** Draped survey grid: thin lines following the terrain surface. */
function buildSurveyGrid(data: SceneData, heightAt: (x: number, z: number) => number): THREE.Group {
  const W = data.worldSize, half = W / 2;
  const div = 12, samples = 40;
  const lift = Math.max(1.5, W * 0.0012);
  const pos: number[] = [];
  const push = (a: THREE.Vector3, b: THREE.Vector3) => { pos.push(a.x, a.y, a.z, b.x, b.y, b.z); };
  for (let d = 0; d <= div; d++) {
    const t = -half + (d / div) * W;
    let prev: THREE.Vector3 | null = null;
    for (let s = 0; s <= samples; s++) {
      const u = -half + (s / samples) * W;
      const p = new THREE.Vector3(t, heightAt(t, u) + lift, u);
      if (prev) push(prev, p);
      prev = p;
    }
    prev = null;
    for (let s = 0; s <= samples; s++) {
      const u = -half + (s / samples) * W;
      const p = new THREE.Vector3(u, heightAt(u, t) + lift, t);
      if (prev) push(prev, p);
      prev = p;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.22, depthWrite: false }));
  lines.renderOrder = 3;
  const g = new THREE.Group();
  g.add(lines);
  return g;
}

/** Canvas-textured billboard label. h = world-unit height of the sprite. */
function makeLabelSprite(
  text: string, h: number,
  bg = 'rgba(5,10,20,0.88)', fg = '#ffffff', border = '#22d3ee',
): THREE.Sprite {
  const fs = 44, pad = 24;
  const cv = document.createElement('canvas');
  let ctx = cv.getContext('2d')!;
  ctx.font = `600 ${fs}px ui-monospace, Menlo, monospace`;
  cv.width = Math.ceil(ctx.measureText(text).width) + pad * 2;
  cv.height = fs + pad * 2;
  ctx = cv.getContext('2d')!;
  const r = cv.height / 2;
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(3, 3, cv.width - 6, cv.height - 6, r);
  else ctx.rect(3, 3, cv.width - 6, cv.height - 6);
  ctx.fillStyle = bg; ctx.fill();
  ctx.lineWidth = 3; ctx.strokeStyle = border; ctx.stroke();
  ctx.font = `600 ${fs}px ui-monospace, Menlo, monospace`;
  ctx.fillStyle = fg; ctx.textBaseline = 'middle';
  ctx.fillText(text, pad, cv.height / 2 + 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(h * cv.width / cv.height, h, 1);
  sp.renderOrder = 999;
  return sp;
}

/** Human distance: meters, or km past 1 km. */
function fmtDist(m: number): string {
  return m >= 1000 ? (m / 1000).toFixed(2) + ' km' : m.toFixed(1) + ' m';
}

/** Deep-dispose a group created by the viewer (geometries, materials, textures). */
function disposeGroup(g: THREE.Group): void {
  g.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) (mesh.geometry as THREE.BufferGeometry).dispose();
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    const kill = (x: THREE.Material) => {
      const t = (x as THREE.SpriteMaterial).map as THREE.Texture | undefined;
      if (t) t.dispose();
      x.dispose();
    };
    if (Array.isArray(mat)) mat.forEach(kill);
    else if (mat) kill(mat);
  });
}
