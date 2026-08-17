"""
ui/scene3d.py
Builds a self-contained HTML document (Three.js via CDN) rendering the
current environment state in 3D. Pure presentation layer — takes a plain
dict describing state and returns HTML; never touches RL/env logic.

v2: realistic stations (shelter/bench/sign/lamp/display), a scripted
dwell state-machine for tram arrivals (decel -> dock -> doors -> alight
-> board -> doors close -> depart, no blinking anywhere), 5 camera
modes, and a fading WAIT/DISPATCH decision banner with reason text.

Backward compatible: 'prev_queues' and 'decision' are optional keys —
if app.py hasn't been upgraded yet, sensible defaults are used.

Called from app.py via streamlit's st.iframe(build_scene_html(...)).
"""

import json

import config as cfg


def _sky_interp_js():
    frames = json.dumps(cfg.SKY_KEYFRAMES)
    return f"""
const SKY_FRAMES = {frames};
function hexNum(h) {{ return parseInt(h, 16); }}
function lerpColor(a, b, t) {{
  const ar=(a>>16)&255, ag=(a>>8)&255, ab=a&255;
  const br=(b>>16)&255, bg=(b>>8)&255, bb=b&255;
  const r = Math.round(ar+(br-ar)*t), g = Math.round(ag+(bg-ag)*t), bl = Math.round(ab+(bb-ab)*t);
  return (r<<16)|(g<<8)|bl;
}}
function skyState(tFrac) {{
  for (let i=0;i<SKY_FRAMES.length-1;i++) {{
    const f0=SKY_FRAMES[i], f1=SKY_FRAMES[i+1];
    if (tFrac>=f0[0] && tFrac<=f1[0]) {{
      const localT = (tFrac-f0[0])/(f1[0]-f0[0]||1);
      return {{
        sky: lerpColor(hexNum(f0[1].slice(2)), hexNum(f1[1].slice(2)), localT),
        sun: lerpColor(hexNum(f0[2].slice(2)), hexNum(f1[2].slice(2)), localT),
        ambient: f0[3]+(f1[3]-f0[3])*localT,
        sunI: f0[4]+(f1[4]-f0[4])*localT
      }};
    }}
  }}
  const last = SKY_FRAMES[SKY_FRAMES.length-1];
  return {{sky: hexNum(last[1].slice(2)), sun: hexNum(last[2].slice(2)), ambient:last[3], sunI:last[4]}};
}}
"""


def build_scene_html(state, camera_mode='Tracking', weather='Clear', height=None):
    """
    state = {
        'minute': int,
        'queues': [q0,q1,q2,q3],                       # current (after this frame)
        'prev_queues': [q0,q1,q2,q3],                   # optional, before this frame
        'trams': [{'id','position','occupancy'}],
        'prev_trams': {id: {'position':int}},
        'dispatch_event': bool,
        'decision': {'action': 'WAIT'|'DISPATCH', 'reason': str} or None,  # optional
    }
    """
    height = height or cfg.SCENE_HEIGHT_PX
    spacing = cfg.STATION_SPACING
    max_fig = cfg.MAX_FIGURES_PER_STATION

    state = dict(state)
    state.setdefault('prev_queues', state['queues'])
    state.setdefault('decision', None)

    state_json = json.dumps(state)
    sky_js = _sky_interp_js()

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; overflow:hidden; background:#111; }}
  #canvas-wrap {{ width:100%; height:{height}px; position:relative; opacity:0; transition:opacity 0.25s ease-in; }}
  #decision-banner {{
    position:absolute; top:14px; left:50%; transform:translateX(-50%);
    padding:8px 22px; border-radius:8px; font:bold 20px sans-serif; color:#fff;
    opacity:0; transition:opacity 0.4s ease; pointer-events:none; letter-spacing:1px;
    box-shadow:0 2px 10px rgba(0,0,0,0.4);
  }}
  #reason-banner {{
    position:absolute; top:52px; left:50%; transform:translateX(-50%);
    padding:4px 14px; border-radius:6px; font:13px sans-serif; color:#eee;
    background:rgba(0,0,0,0.55); opacity:0; transition:opacity 0.4s ease; pointer-events:none;
  }}
</style>
</head>
<body>
<div id="canvas-wrap">
  <div id="decision-banner"></div>
  <div id="reason-banner"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
{sky_js}
const STATE = {state_json};
const SPACING = {spacing};
const MAX_FIG = {max_fig};
const WEATHER = "{weather}";
const CAMERA_MODE = "{camera_mode}";

const wrap = document.getElementById('canvas-wrap');
const W = wrap.clientWidth || 900, H = {height};

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
wrap.insertBefore(renderer.domElement, wrap.firstChild);

// ---------------------------------------------------------------------
// Audio: distinct stylised tones (no external assets)
// ---------------------------------------------------------------------
let audioCtx = null;
function chime(freq, dur, type) {{
  try {{
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type || 'sine'; osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.14, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(); osc.stop(audioCtx.currentTime + dur);
  }} catch(e) {{}}
}}
const CHIME_ARRIVAL  = () => chime(520, 0.15, 'sine');
const CHIME_DOOR      = () => chime(300, 0.10, 'square');
const CHIME_DISPATCH  = () => chime(880, 0.35, 'sine');

// ---------------------------------------------------------------------
// Ground + track
// ---------------------------------------------------------------------
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(220, 70),
  new THREE.MeshStandardMaterial({{color: {cfg.GROUND_COLOR}}})
);
ground.rotation.x = -Math.PI/2;
ground.receiveShadow = true;
scene.add(ground);

const track = new THREE.Mesh(
  new THREE.BoxGeometry(SPACING*4 + 6, 0.15, 1.6),
  new THREE.MeshStandardMaterial({{color: {cfg.TRACK_COLOR}}})
);
track.position.set(SPACING*1.5, 0.08, 0);
track.receiveShadow = true;
scene.add(track);

[-0.5, 0.5].forEach(off => {{
  const rail = new THREE.Mesh(
    new THREE.BoxGeometry(SPACING*4 + 6, 0.08, 0.08),
    new THREE.MeshStandardMaterial({{color: 0x888888, metalness:0.6}})
  );
  rail.position.set(SPACING*1.5, 0.2, off);
  scene.add(rail);
}});
// sleepers
for (let x=-2; x<SPACING*4+4; x+=1.2) {{
  const sleeper = new THREE.Mesh(
    new THREE.BoxGeometry(0.25, 0.06, 1.8),
    new THREE.MeshStandardMaterial({{color:0x4a3826}})
  );
  sleeper.position.set(x, 0.05, 0);
  scene.add(sleeper);
}}

// ---------------------------------------------------------------------
// Label sprite helper
// ---------------------------------------------------------------------
function makeLabelSprite(text, scale, bg) {{
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = bg || 'rgba(0,0,0,0.55)';
  ctx.fillRect(0,0,c.width,c.height);
  ctx.font = 'bold 32px sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.fillText(text, c.width/2, 42);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({{map:tex, transparent:true}});
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(scale, scale/4, 1);
  sprite.userData.canvas = c;
  sprite.userData.ctx = ctx;
  return sprite;
}}
function refreshLabel(sprite, text, bg) {{
  const c = sprite.userData.canvas, ctx = sprite.userData.ctx;
  ctx.fillStyle = bg || 'rgba(0,0,0,0.55)'; ctx.fillRect(0,0,c.width,c.height);
  ctx.font='bold 32px sans-serif'; ctx.fillStyle='#fff'; ctx.textAlign='center';
  ctx.fillText(text, c.width/2, 42);
  sprite.material.map.needsUpdate = true;
}}

// ---------------------------------------------------------------------
// Stations — platform, shelter, bench, sign, lamp, digital display
// ---------------------------------------------------------------------
const stationGroup = new THREE.Group();
const queueBars = [];
const queueLabels = [];
const displaySprites = [];
const stationAnchors = [];  // world positions for Station View camera
const platformMeshes = [];  // for live congestion tinting (boarding stations 0-3 only)

// Congestion heatmap colour: green (empty) -> yellow -> red (>= ~40 waiting)
function congestionColor(q) {{
  const t = Math.max(0, Math.min(1, q / 40));
  const c1 = t < 0.5 ? [0x2e,0xcc,0x71] : [0xf1,0xc4,0x0f]; // green->yellow
  const c2 = t < 0.5 ? [0xf1,0xc4,0x0f] : [0xe7,0x4c,0x3c]; // yellow->red
  const localT = t < 0.5 ? t/0.5 : (t-0.5)/0.5;
  const r = Math.round(c1[0]+(c2[0]-c1[0])*localT);
  const g = Math.round(c1[1]+(c2[1]-c1[1])*localT);
  const b = Math.round(c1[2]+(c2[2]-c1[2])*localT);
  return (r<<16)|(g<<8)|b;
}}

for (let i=0;i<5;i++) {{
  const color = i===0 ? {cfg.DEPOT_COLOR} : (i===4 ? {cfg.TERMINUS_COLOR} : {cfg.STATION_COLOR});
  const cx = i*SPACING;
  stationAnchors.push(cx);

  const platform = new THREE.Mesh(
    new THREE.BoxGeometry(3.2, 0.5, 3.2),
    new THREE.MeshStandardMaterial({{color}})
  );
  platform.position.set(cx, 0.25, 2.8);
  platform.castShadow = true; platform.receiveShadow = true;
  stationGroup.add(platform);
  platformMeshes.push(platform);  // index 4 (terminus) kept as role color, never tinted

  // Shelter: roof + 2 poles
  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(2.6, 0.1, 1.6),
    new THREE.MeshStandardMaterial({{color:0x445566}})
  );
  roof.position.set(cx, 2.4, 3.6);
  roof.castShadow = true;
  stationGroup.add(roof);
  [[-1.1,3.0],[1.1,3.0]].forEach(([dx,dz]) => {{
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,2.2,8),
      new THREE.MeshStandardMaterial({{color:0x333333}}));
    pole.position.set(cx+dx, 1.3, dz);
    stationGroup.add(pole);
  }});

  // Bench
  const bench = new THREE.Mesh(new THREE.BoxGeometry(1.2,0.35,0.4),
    new THREE.MeshStandardMaterial({{color:0x6b4a2f}}));
  bench.position.set(cx, 0.7, 3.9);
  stationGroup.add(bench);

  // Lamp post + light
  const lampPole = new THREE.Mesh(new THREE.CylinderGeometry(0.06,0.06,3,8),
    new THREE.MeshStandardMaterial({{color:0x222222}}));
  lampPole.position.set(cx-1.7, 1.5, 2.0);
  stationGroup.add(lampPole);
  const lampHead = new THREE.Mesh(new THREE.SphereGeometry(0.15,10,10),
    new THREE.MeshStandardMaterial({{color:0xfff2b3, emissive:0xfff2b3, emissiveIntensity:0.6}}));
  lampHead.position.set(cx-1.7, 3.0, 2.0);
  stationGroup.add(lampHead);
  const lampLight = new THREE.PointLight(0xfff2c0, 0.5, 6);
  lampLight.position.copy(lampHead.position);
  stationGroup.add(lampLight);

  // Station sign
  const name = i===0 ? 'DEPOT' : (i===4 ? 'TERMINUS' : ('STATION ' + i));
  const label = makeLabelSprite(name, 4);
  label.position.set(cx, 3.4, 2.8);
  stationGroup.add(label);

  // Digital display (next-arrival countdown, decorative)
  const display = makeLabelSprite('NEXT: --', 2.6, 'rgba(10,30,10,0.85)');
  display.position.set(cx, 2.0, 3.6);
  stationGroup.add(display);
  displaySprites.push(display);

  if (i < 4) {{
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(0.6, 1, 0.6),
      new THREE.MeshStandardMaterial({{color: 0xffcc00}})
    );
    bar.position.set(cx - 1.2, 0.5, 4.6);
    stationGroup.add(bar);
    queueBars.push(bar);

    const qLabel = makeLabelSprite('Q:0', 2.5);
    qLabel.position.set(cx - 1.2, 2.0, 4.6);
    stationGroup.add(qLabel);
    queueLabels.push(qLabel);

    const figs = [];
    for (let f=0; f<MAX_FIG; f++) {{
      const person = new THREE.Mesh(
        new THREE.CylinderGeometry(0.15,0.15,0.7,8),
        new THREE.MeshStandardMaterial({{color: 0xffffff}})
      );
      person.position.set(cx - 2.2 + (f%4)*0.35, 0.45, 5.0 + Math.floor(f/4)*0.4);
      person.visible = false;
      person.userData.homeX = person.position.x;
      person.userData.homeZ = person.position.z;
      stationGroup.add(person);
      figs.push(person);
    }}
    bar.userData.figs = figs;
    queueBars[queueBars.length-1] = bar;
  }}
}}
scene.add(stationGroup);

// ---------------------------------------------------------------------
// Tram builder
// ---------------------------------------------------------------------
function buildTram() {{
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(2.6, 1.2, 1.4),
    new THREE.MeshStandardMaterial({{color: {cfg.TRAM_BODY_COLOR}}})
  );
  body.position.y = 1.0;
  body.castShadow = true;
  g.add(body);

  // windows
  [[-0.7,0.72],[0,0.72],[0.7,0.72],[-0.7,-0.72],[0,-0.72],[0.7,-0.72]].forEach(([dx,dz]) => {{
    const win = new THREE.Mesh(new THREE.BoxGeometry(0.5,0.4,0.03),
      new THREE.MeshStandardMaterial({{color:0xbde3ff, transparent:true, opacity:0.55}}));
    win.position.set(dx, 1.2, dz);
    g.add(win);
  }});

  // interior passengers (visible dots, capped, scaled by occupancy)
  const interior = [];
  for (let k=0;k<10;k++) {{
    const p = new THREE.Mesh(new THREE.SphereGeometry(0.08,6,6),
      new THREE.MeshStandardMaterial({{color:0x333333}}));
    p.position.set(-0.9 + (k%5)*0.42, 1.0, (k<5?-0.3:0.3));
    p.visible = false;
    interior.push(p); g.add(p);
  }}
  g.userData.interior = interior;

  const wheels = [];
  [[-1,-0.6],[1,-0.6],[-1,0.6],[1,0.6]].forEach(([dx,dz]) => {{
    const wheel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.3,0.3,0.2,12),
      new THREE.MeshStandardMaterial({{color:0x111111}})
    );
    wheel.rotation.z = Math.PI/2;
    wheel.position.set(dx, 0.3, dz);
    g.add(wheel);
    wheels.push(wheel);
  }});
  g.userData.wheels = wheels;

  const doorL = new THREE.Mesh(new THREE.BoxGeometry(0.05,1.0,0.6),
    new THREE.MeshStandardMaterial({{color:0x222222}}));
  doorL.position.set(1.3, 1.0, -0.4);
  const doorR = doorL.clone(); doorR.position.z = 0.4;
  g.add(doorL); g.add(doorR);
  g.userData.doorL = doorL; g.userData.doorR = doorR;

  const headlight = new THREE.PointLight(0xffffee, 0.8, 8);
  headlight.position.set(1.4, 1.1, 0);
  g.add(headlight);
  const headlampMesh = new THREE.Mesh(new THREE.SphereGeometry(0.08,8,8),
    new THREE.MeshStandardMaterial({{color:0xffffff, emissive:0xffffee, emissiveIntensity:0.8}}));
  headlampMesh.position.set(1.35,1.0,0);
  g.add(headlampMesh);

  const brakeLight = new THREE.Mesh(new THREE.BoxGeometry(0.08,0.15,0.9),
    new THREE.MeshStandardMaterial({{color:0x660000, emissive:0xff0000, emissiveIntensity:0}}));
  brakeLight.position.set(-1.32, 1.0, 0);
  g.add(brakeLight);
  g.userData.brakeLight = brakeLight;

  const interiorLight = new THREE.PointLight(0xfff2cc, 0.35, 3);
  interiorLight.position.set(0, 1.3, 0);
  g.add(interiorLight);

  return g;
}}

// ---------------------------------------------------------------------
// Lighting / sky / weather
// ---------------------------------------------------------------------
const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(20, 30, 10);
sun.castShadow = true;
scene.add(sun);

const tFrac = STATE.minute / 1080;
const sky = skyState(tFrac);
scene.background = new THREE.Color(sky.sky);
ambient.intensity = sky.ambient;
sun.intensity = sky.sunI;
sun.color = new THREE.Color(sky.sun);
wrap.style.background = '#' + sky.sky.toString(16).padStart(6,'0');

if (WEATHER === 'Fog') {{ scene.fog = new THREE.Fog(sky.sky, 15, 70); }}
let rainPoints = null;
if (WEATHER === 'Rain') {{
  const n = 800;
  const positions = new Float32Array(n*3);
  for (let i=0;i<n;i++) {{
    positions[i*3]=(Math.random()-0.5)*90; positions[i*3+1]=Math.random()*20; positions[i*3+2]=(Math.random()-0.5)*40;
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  rainPoints = new THREE.Points(geo, new THREE.PointsMaterial({{color:0x88aaff,size:0.15,transparent:true,opacity:0.6}}));
  scene.add(rainPoints);
}}

// ---------------------------------------------------------------------
// Decision banner (fades, never blinks)
// ---------------------------------------------------------------------
const decisionEl = document.getElementById('decision-banner');
const reasonEl = document.getElementById('reason-banner');
if (STATE.decision) {{
  const isDispatch = STATE.decision.action === 'DISPATCH';
  decisionEl.textContent = STATE.decision.action;
  decisionEl.style.background = isDispatch ? 'rgba(46,204,113,0.9)' : 'rgba(90,100,110,0.85)';
  reasonEl.textContent = STATE.decision.reason || '';
  requestAnimationFrame(() => {{ decisionEl.style.opacity = 1; reasonEl.style.opacity = 1; }});
  setTimeout(() => {{ decisionEl.style.opacity = 0; reasonEl.style.opacity = 0; }}, 1600);
  if (isDispatch) CHIME_DISPATCH();
}}

// ---------------------------------------------------------------------
// Queue bars/figures — set to PREVIOUS values first, animate to current
// (this is what makes boarding/alighting look continuous, not a snap)
// ---------------------------------------------------------------------
function setQueueVisual(i, q) {{
  const bar = queueBars[i];
  const h = Math.min(1 + q/8, 6);
  bar.scale.y = h; bar.position.y = h/2;
  const shown = Math.min(Math.round(q), MAX_FIG);
  bar.userData.figs.forEach((p, idx) => p.visible = idx < shown);
  refreshLabel(queueLabels[i], 'Q:'+Math.round(q));
  platformMeshes[i].material.color.setHex(congestionColor(q));  // congestion heatmap
}}
const prevQ = STATE.prev_queues || STATE.queues;
STATE.queues.forEach((q, i) => setQueueVisual(i, prevQ[i]));

function animateQueueTo(i, fromQ, toQ, durMs, delayMs) {{
  setTimeout(() => {{
    const t0 = performance.now();
    function step() {{
      const p = Math.min((performance.now()-t0)/durMs, 1);
      setQueueVisual(i, fromQ + (toQ-fromQ)*p);
      if (p<1) requestAnimationFrame(step);
    }}
    step();
  }}, delayMs);
}}

// ---------------------------------------------------------------------
// Digital displays — decorative countdown based on elapsed/queue pressure
// ---------------------------------------------------------------------
displaySprites.forEach((d, i) => refreshLabel(d, i===0 ? 'BOARDING' : 'STANDBY', 'rgba(10,30,10,0.85)'));

// ---------------------------------------------------------------------
// Tram placement + dwell state machine
// ---------------------------------------------------------------------
const tramMeshes = {{}};

function animateDoor(door, targetZ, durMs) {{
  const startZ = door.position.z, t0 = performance.now();
  function step() {{
    const p = Math.min((performance.now()-t0)/durMs, 1);
    door.position.z = startZ + (targetZ-startZ)*p;
    if (p<1) requestAnimationFrame(step);
  }}
  step();
}}

function runDwellSequence(mesh, startX, endX, destIndex, prevOcc, currOcc) {{
  const travelMs = 900;
  const t0 = performance.now();
  function easeInOutCubic(t) {{ return t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2; }}

  function travelStep() {{
    const elapsed = performance.now()-t0;
    const p = Math.min(elapsed/travelMs, 1);
    const eased = easeInOutCubic(p);
    mesh.position.x = startX + (endX-startX)*eased;
    mesh.userData.wheels.forEach(w => w.rotation.x -= 0.35*(1-p*0.5));
    mesh.userData.brakeLight.material.emissiveIntensity = p>0.75 ? (p-0.75)/0.25 : 0;
    if (p<1) {{ requestAnimationFrame(travelStep); }}
    else {{
      mesh.position.x = endX;
      mesh.userData.brakeLight.material.emissiveIntensity = 1;
      setTimeout(dock, 100);
    }}
  }}
  function dock() {{
    mesh.userData.brakeLight.material.emissiveIntensity = 0;
    CHIME_ARRIVAL();
    setTimeout(openDoors, 120);
  }}
  function openDoors() {{
    animateDoor(mesh.userData.doorL, -0.9, 280);
    animateDoor(mesh.userData.doorR, 0.9, 280);
    CHIME_DOOR();
    setTimeout(boardAlight, 320);
  }}
  function boardAlight() {{
    // interior passenger dots reflect new occupancy
    const shown = Math.min(Math.round(currOcc/8), 10);
    mesh.userData.interior.forEach((p, idx) => p.visible = idx < shown);
    // queue bar at destination shrinks smoothly during this phase
    if (destIndex >= 0 && destIndex < 4) {{
      animateQueueTo(destIndex, prevQ[destIndex], STATE.queues[destIndex], 500, 0);
    }}
    setTimeout(closeDoors, 550);
  }}
  function closeDoors() {{
    animateDoor(mesh.userData.doorL, -0.4, 220);
    animateDoor(mesh.userData.doorR, 0.4, 220);
  }}
  travelStep();
}}

(STATE.trams || []).forEach(t => {{
  let mesh = tramMeshes[t.id];
  const isNew = !mesh;
  if (isNew) {{
    mesh = buildTram();
    scene.add(mesh);
    tramMeshes[t.id] = mesh;
  }}
  // A tram id that never appeared in prev_trams was dispatched THIS step.
  // tram_env.py's own step() advances a freshly-dispatched tram from
  // position 0 to 1 within the same call (dispatch -> board -> move all
  // happen in one env.step()), so by the time app.py reads env.trams the
  // "new" tram already reports position 1, not 0. Left as t.position, this
  // makes brand-new trams pop into existence already at Station 1 instead
  // of visibly leaving the Depot. Presentation-only fix: for genuinely new
  // trams, always animate the departure from the Depot (station index 0),
  // regardless of which station they've already reached this step. No env/
  // reward/state logic is touched — this only changes where the mesh is
  // drawn starting from.
  const hasPrev = STATE.prev_trams && STATE.prev_trams[t.id];
  const prev = hasPrev ? STATE.prev_trams[t.id].position : (isNew ? 0 : t.position);
  const startX = prev * SPACING;
  const endX = t.position * SPACING;
  if (isNew) mesh.position.x = startX;

  if (startX === endX) {{
    // no movement this frame — just reflect occupancy, no animation needed
    const shown = Math.min(Math.round(t.occupancy/8), 10);
    mesh.userData.interior.forEach((p, idx) => p.visible = idx < shown);
  }} else {{
    runDwellSequence(mesh, startX, endX, t.position, t.occupancy, t.occupancy);
  }}
}});

Object.keys(tramMeshes).forEach(id => {{
  if (!(STATE.trams||[]).some(t => String(t.id) === id)) {{
    scene.remove(tramMeshes[id]);
    delete tramMeshes[id];
  }}
}});

// non-boarding stations (not visited by a tram this frame) still settle to current values
STATE.queues.forEach((q, i) => {{
  const visited = (STATE.trams||[]).some(t => t.position === i);
  if (!visited) animateQueueTo(i, prevQ[i], q, 400, 0);
}});

// ---------------------------------------------------------------------
// Cameras: Top View, Tracking, Driver View, Station View, Free Orbit
// ---------------------------------------------------------------------
let orbitYaw = Math.PI*0.15, orbitPitch = 0.5, orbitDist = 26;
const sceneCenterX = SPACING*1.5;

if (CAMERA_MODE === 'Free Orbit') {{
  let dragging=false,lastX=0,lastY=0;
  renderer.domElement.addEventListener('mousedown', e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;}});
  window.addEventListener('mouseup', ()=>dragging=false);
  window.addEventListener('mousemove', e=>{{
    if(!dragging) return;
    orbitYaw += (e.clientX-lastX)*0.005;
    orbitPitch = Math.max(0.1, Math.min(1.4, orbitPitch+(e.clientY-lastY)*0.005));
    lastX=e.clientX; lastY=e.clientY;
  }});
  renderer.domElement.addEventListener('wheel', e=>{{
    orbitDist = Math.max(8, Math.min(60, orbitDist+e.deltaY*0.02));
  }});
}}

function leadTramX() {{
  return (STATE.trams && STATE.trams[0]) ? STATE.trams[0].position*SPACING : 0;
}}
function nearestStationIndex(x) {{
  return Math.max(0, Math.min(4, Math.round(x/SPACING)));
}}

function updateCamera() {{
  if (CAMERA_MODE === 'Tracking') {{
    const lead = leadTramX();
    camera.position.lerp(new THREE.Vector3(lead-8, 6, 10), 0.08);
    camera.lookAt(lead, 1, 0);
  }} else if (CAMERA_MODE === 'Driver View') {{
    const lead = leadTramX();
    camera.position.lerp(new THREE.Vector3(lead+1.3, 1.6, 0), 0.15);
    camera.lookAt(lead+10, 1.2, 0);
  }} else if (CAMERA_MODE === 'Station View') {{
    const si = nearestStationIndex(leadTramX());
    const sx = stationAnchors[si];
    camera.position.set(sx-3, 2.2, 7.5);
    camera.lookAt(sx, 1.2, 2.8);
  }} else if (CAMERA_MODE === 'Free Orbit') {{
    camera.position.set(
      sceneCenterX + orbitDist*Math.sin(orbitYaw)*Math.cos(orbitPitch),
      orbitDist*Math.sin(orbitPitch)+3,
      orbitDist*Math.cos(orbitYaw)*Math.cos(orbitPitch)
    );
    camera.lookAt(sceneCenterX, 1, 0);
  }} else {{ // Top View
    camera.position.set(sceneCenterX, 40, 0.01);
    camera.lookAt(sceneCenterX, 0, 0);
  }}
}}

// ---------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------
function loop() {{
  updateCamera();
  if (rainPoints) {{
    const pos = rainPoints.geometry.attributes.position;
    for (let i=0;i<pos.count;i++) {{
      let y = pos.getY(i)-0.5;
      if (y<0) y=20;
      pos.setY(i,y);
    }}
    pos.needsUpdate = true;
  }}
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}}
loop();
requestAnimationFrame(() => {{ wrap.style.opacity = 1; }});
</script>
</body>
</html>
"""