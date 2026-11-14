// Client-side replay player for Mission Control (Streamlit components v2).
//
// Why this exists: the replay used to advance by a fixed step on every Streamlit
// rerun (every 0.5 s) and rebuilt the whole Plotly figure each time. That gave
// ~2 frames per second, jumps of 2 s of replay at 4x, and a WebGL scene whose
// tree mesh changed shape every frame. Here the browser owns the clock while
// playing: replay time = anchor + speed * elapsed wall time, and every display
// frame is drawn on a canvas (3D orchard or 2D map, ~1 ms per frame) with the
// robot interpolated between the 10 Hz samples. Plotly was measured at 28 to 50 ms
// per animated frame in 3D and ~34 ms in 2D, too slow for smooth motion. Every
// ~0.25 s the current time is reported to Python so the Streamlit side panel
// (status, tiles, checklist, alerts, progress) follows the same clock.
//
// The pure helpers below are exported for unit tests (tests/test_replay.py runs
// them with Node). The default export is the Streamlit entry point.

const REPORT_MS = 250;      // send the clock to Python at most 4 times per second
const TRAIL_MAX = 3000;     // trail history points kept per redraw (robot itself is never decimated)

// ------------------------------------------------------------ pure helpers --

export function simTime(anchorT, anchorWallMs, nowMs, speed, end) {
  const t = anchorT + (speed * (nowMs - anchorWallMs)) / 1000;
  return Math.min(t, end);
}

export function lastIndex(ts, t) {
  // largest i with ts[i] <= t (0 if t is before the first sample)
  let lo = 0, hi = ts.length - 1;
  if (hi < 0 || t <= ts[0]) return 0;
  if (t >= ts[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid;
  }
  return lo;
}

export function interp(arr, ts, t) {
  const n = ts.length, i = lastIndex(ts, t);
  if (i >= n - 1 || t <= ts[0]) return arr[i];
  const f = (t - ts[i]) / (ts[i + 1] - ts[i]);
  return arr[i] + (arr[i + 1] - arr[i]) * f;
}

export function interpAngle(arr, ts, t) {
  const n = ts.length, i = lastIndex(ts, t);
  if (i >= n - 1 || t <= ts[0]) return arr[i];
  const f = (t - ts[i]) / (ts[i + 1] - ts[i]);
  let d = arr[i + 1] - arr[i];
  d = Math.atan2(Math.sin(d), Math.cos(d));      // shortest way round
  return arr[i] + d * f;
}

export function poseAt(S, t) {
  const i = lastIndex(S.t, t);
  return {
    i,
    x: interp(S.x, S.t, t), y: interp(S.y, S.t, t), th: interpAngle(S.th, S.t, t),
    lvl: S.lvl[i], ua: interp(S.ua, S.t, t), ub: interp(S.ub, S.t, t), ang: S.ang[i],
    ringR: S.ring ? interp(S.ring.r, S.t, t) : null, ringLvl: S.ring ? S.ring.lvl[i] : null,
  };
}

export function treeStatus(changes, nTrees, t) {
  // changes: [[treeIndex, time, code], ...] sorted by time; later changes win
  const out = new Array(nTrees).fill(0);
  for (const [k, tc, code] of changes) {
    if (tc > t) break;
    out[k] = code;
  }
  return out;
}

export function visibleWorkers(workers, t) {
  return workers.filter((w) => (w.t0 === null || t >= w.t0) && (w.t1 === null || t <= w.t1));
}

// ---------------------------------------------------------------- geometry --

function ring(cx, cy, r, z, n = 64) {
  const x = [], y = [], zz = [];
  for (let k = 0; k <= n; k++) {
    const a = (2 * Math.PI * k) / n;
    x.push(cx + r * Math.cos(a)); y.push(cy + r * Math.sin(a)); zz.push(z);
  }
  return { x, y, z: zz };
}

function ellipse(cx, cy, a, b, ang, n = 64) {
  const x = [], y = [], c = Math.cos(ang), s = Math.sin(ang);
  for (let k = 0; k <= n; k++) {
    const t = (2 * Math.PI * k) / n, ex = a * Math.cos(t), ey = b * Math.sin(t);
    x.push(cx + c * ex - s * ey); y.push(cy + s * ex + c * ey);
  }
  return { x, y };
}

// ------------------------------------------------------- canvas 3D view --
// Plotly's WebGL 3D scene re-processes every trace on any data change (measured ~28 ms of
// fixed overhead even with a single trace, ~50 ms with the orchard), which caps animation at
// a few frames per second. This small perspective renderer draws the same scene on a 2D
// canvas: moving the robot costs a redraw of a few hundred shapes (well under 5 ms).

function hexToRgb(h) {
  const v = h.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}
function shade(h, f, a = 1) {
  const [r, g, b] = hexToRgb(h);
  return `rgba(${Math.round(Math.min(255, r * f))},${Math.round(Math.min(255, g * f))},${Math.round(Math.min(255, b * f))},${a})`;
}

class Canvas3D {
  constructor(div, S, follow, colors) {
    this.div = div; this.S = S; this.follow = follow; this.col = colors;
    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%"; this.canvas.style.height = "100%"; this.canvas.style.display = "block";
    this.canvas.style.cursor = "grab"; this.canvas.style.borderRadius = "10px";
    div.innerHTML = ""; div.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    const [x0, x1, y0, y1] = S.bounds;
    this.center = { x: (x0 + x1) / 2, y: (y0 + y1) / 2 };
    this.span = Math.max(x1 - x0, y1 - y0);
    // camera: orbit around a target; the user can drag (orbit) and scroll (zoom) at any time
    this.yaw = -2.0; this.pitch = follow ? 0.62 : 0.85; this.dist = follow ? 34 : this.span * 1.05;
    this.target = { x: follow ? S.x[0] : this.center.x, y: follow ? S.y[0] : this.center.y, z: 0 };
    this.fov = (45 * Math.PI) / 180;
    this.bindInput();
    this.resize();
    this.ro = new ResizeObserver(() => { this.resize(); if (this.last) this.render(this.last); });
    this.ro.observe(div);
  }

  bindInput() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; this.canvas.setPointerCapture(e.pointerId); this.canvas.style.cursor = "grabbing"; });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.yaw -= (e.clientX - drag.x) * 0.006;
      this.pitch = Math.min(1.45, Math.max(0.12, this.pitch + (e.clientY - drag.y) * 0.004));
      drag = { x: e.clientX, y: e.clientY };
      if (this.last) this.render(this.last);
    });
    const up = () => { drag = null; this.canvas.style.cursor = "grab"; };
    this.canvas.addEventListener("pointerup", up); this.canvas.addEventListener("pointercancel", up);
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.dist = Math.min(this.span * 3, Math.max(8, this.dist * Math.exp(e.deltaY * 0.0012)));
      if (this.last) this.render(this.last);
    }, { passive: false });
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, w = this.div.clientWidth || 600, h = this.div.clientHeight || 470;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.W = w; this.H = h; this.dpr = dpr;
  }

  destroy() { if (this.ro) this.ro.disconnect(); }

  setupCamera() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch), t = this.target;
    const eye = { x: t.x + this.dist * cp * Math.cos(this.yaw), y: t.y + this.dist * cp * Math.sin(this.yaw), z: t.z + this.dist * sp };
    let f = { x: t.x - eye.x, y: t.y - eye.y, z: t.z - eye.z };
    const fl = Math.hypot(f.x, f.y, f.z); f = { x: f.x / fl, y: f.y / fl, z: f.z / fl };
    let r = { x: f.y, y: -f.x, z: 0 };                               // forward x up(z)
    const rl = Math.hypot(r.x, r.y) || 1; r = { x: r.x / rl, y: r.y / rl, z: 0 };
    const u = { x: r.y * f.z - r.z * f.y, y: r.z * f.x - r.x * f.z, z: r.x * f.y - r.y * f.x };
    this.cam = { eye, f, r, u, F: this.H / 2 / Math.tan(this.fov / 2) };
  }

  view(x, y, z) {   // camera space: [right, up, depth]
    const c = this.cam, px = x - c.eye.x, py = y - c.eye.y, pz = z - c.eye.z;
    return [px * c.r.x + py * c.r.y + pz * c.r.z, px * c.u.x + py * c.u.y + pz * c.u.z, px * c.f.x + py * c.f.y + pz * c.f.z];
  }

  screen(v) { const c = this.cam; return [this.W / 2 + (v[0] / v[2]) * c.F, this.H / 2 - (v[1] / v[2]) * c.F]; }

  // polygon / polyline in world space, clipped against the near plane, then projected
  path(pts, closed) {
    const NEAR = 0.3, vs = pts.map((p) => this.view(p[0], p[1], p[2])), out = [];
    const n = vs.length, lim = closed ? n : n - 1;
    const push = (v) => out.push(this.screen(v));
    if (!closed) {
      // polyline: split into visible runs
      const runs = []; let cur = [];
      for (let k = 0; k < n; k++) {
        const a = vs[k];
        if (a[2] > NEAR) cur.push(this.screen(a));
        else if (cur.length) { runs.push(cur); cur = []; }
      }
      if (cur.length) runs.push(cur);
      return runs;
    }
    for (let k = 0; k < lim; k++) {
      const a = vs[k], b = vs[(k + 1) % n], ina = a[2] > NEAR, inb = b[2] > NEAR;
      if (ina) push(a);
      if (ina !== inb) {
        const s = (NEAR - a[2]) / (b[2] - a[2]);
        push([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, NEAR]);
      }
    }
    return out;
  }

  poly(pts, fill, stroke, lw = 1, dash = null) {
    const P = this.path(pts, true), ctx = this.ctx;
    if (P.length < 3) return;
    ctx.beginPath(); ctx.moveTo(P[0][0], P[0][1]);
    for (let k = 1; k < P.length; k++) ctx.lineTo(P[k][0], P[k][1]);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.setLineDash(dash || []); ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]); }
  }

  line(pts, stroke, lw, dash = null) {
    const ctx = this.ctx; ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.setLineDash(dash || []);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const run of this.path(pts, false)) {
      if (run.length < 2) continue;
      ctx.beginPath(); ctx.moveTo(run[0][0], run[0][1]);
      for (let k = 1; k < run.length; k++) ctx.lineTo(run[k][0], run[k][1]);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  label(x, y, z, text, color, size = 12) {
    const v = this.view(x, y, z);
    if (v[2] <= 0.3) return;
    const [sx, sy] = this.screen(v), ctx = this.ctx;
    ctx.font = `600 ${size}px Inter, system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillStyle = "rgba(11,17,24,0.75)"; const w = ctx.measureText(text).width;
    ctx.fillRect(sx - w / 2 - 4, sy - size - 6, w + 8, size + 5);
    ctx.fillStyle = color; ctx.fillText(text, sx, sy - 3);
  }

  render(F) {
    this.last = F;
    const ctx = this.ctx, S = this.S, col = this.col;
    if (this.follow) {                 // ease the target towards the robot: smooth, never a jump-cut
      const now = performance.now(), dt = this.lastRender ? Math.min(0.2, (now - this.lastRender) / 1000) : 1;
      this.lastRender = now;
      const k = F.snap ? 1 : 1 - Math.exp(-dt / 0.18);
      this.target.x += (F.p.x - this.target.x) * k; this.target.y += (F.p.y - this.target.y) * k;
    }
    this.setupCamera();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = "#0e141b"; ctx.fillRect(0, 0, this.W, this.H);
    // ground and 10 m grid
    const [x0, x1, y0, y1] = S.bounds;
    this.poly([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], "#1b2420", null);
    for (let v = Math.ceil(x0 / 10) * 10; v < x1; v += 10) this.line([[v, y0, 0.01], [v, y1, 0.01]], "#26332d", 1);
    for (let v = Math.ceil(y0 / 10) * 10; v < y1; v += 10) this.line([[x0, v, 0.01], [x1, v, 0.01]], "#26332d", 1);
    // ground-level layers
    if (S.ref) this.line(S.ref.x.map((x, k) => [x, S.ref.y[k], 0.05]), col.ref, 1.6, [4, 4]);
    for (const seg of F.trail) this.line(seg.pts.map((q) => [q[0], q[1], 0.12]), seg.color, 4);
    for (const w of F.workers) {
      const r = ring(w.x, w.y, w.r, 0.05, 40);
      this.poly(r.x.map((x, k) => [x, r.y[k], 0.05]), "rgba(238,242,239,0.10)", col.ink, 1.5, [3, 3]);
    }
    const p = F.p;
    if (F.safety) this.line(F.safety.x.map((x, k) => [x, F.safety.y[k], 0.07]), F.safety.color, 2.4, [7, 5]);
    this.poly(F.bubble.x.map((x, k) => [x, F.bubble.y[k], 0.09]), shade(F.stateColor, 1, 0.18), F.stateColor, 2);
    // tall objects, painter's order (far to near)
    const items = [];
    const T = S.trees;
    for (let k = 0; k < T.x.length; k++) {
      const v = this.view(T.x[k], T.y[k], 2.3);
      if (v[2] > 0.5 && v[2] < this.dist * 4) items.push({ d: v[2], kind: "tree", k });
    }
    items.push({ d: this.view(p.x, p.y, 0.5)[2], kind: "robot" });
    for (const w of F.workers) items.push({ d: this.view(w.x, w.y, 1)[2], kind: "worker", w });
    for (const m of F.markers) items.push({ d: this.view(m.x, m.y, 1.5)[2], kind: "marker", m });
    items.sort((a, b) => b.d - a.d);
    for (const it of items) {
      if (it.kind === "tree") this.tree(T.x[it.k], T.y[it.k], F.treeColors[it.k]);
      else if (it.kind === "robot") this.robot(p, F.stateColor);
      else if (it.kind === "worker") {
        this.line([[it.w.x, it.w.y, 0], [it.w.x, it.w.y, 1.6]], col.ink, 2);
        this.dot(it.w.x, it.w.y, 1.7, 0.18, col.ink);
        this.label(it.w.x, it.w.y, 2.0, it.w.label, col.ink);
      } else {
        this.line([[it.m.x, it.m.y, 0], [it.m.x, it.m.y, 3.2]], col.warn, 2.5);
        this.label(it.m.x, it.m.y, 3.3, it.m.label, col.warn);
      }
    }
  }

  dot(x, y, z, r, color) {
    const v = this.view(x, y, z);
    if (v[2] <= 0.3) return;
    const [sx, sy] = this.screen(v), rad = Math.max(2, (r * this.cam.F) / v[2]), ctx = this.ctx;
    ctx.beginPath(); ctx.arc(sx, sy, rad, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#0b0f14"; ctx.stroke();
  }

  tree(x, y, color) {
    const v = this.view(x, y, 2.3);
    const [sx, sy] = this.screen(v), rad = (1.4 * this.cam.F) / v[2], ctx = this.ctx;
    if (sx < -rad || sx > this.W + rad || sy < -rad * 2 || sy > this.H + rad * 2) return;
    this.line([[x, y, 0], [x, y, 1.2]], "#5b4636", Math.max(1, (0.25 * this.cam.F) / v[2]));
    const g = ctx.createRadialGradient(sx - rad * 0.35, sy - rad * 0.4, rad * 0.1, sx, sy, rad);
    g.addColorStop(0, shade(color, 1.35)); g.addColorStop(1, shade(color, 0.75));
    ctx.beginPath(); ctx.ellipse(sx, sy, rad, rad * 1.05, 0, 0, 2 * Math.PI);
    ctx.fillStyle = g; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = shade(color, 0.55); ctx.stroke();
  }

  robot(p, stateColor) {
    const c = Math.cos(p.th), s = Math.sin(p.th), L = 0.75, Wd = 0.625;
    const corner = (dx, dy, z) => [p.x + c * dx - s * dy, p.y + s * dx + c * dy, z];
    const b = [[L, Wd], [-L, Wd], [-L, -Wd], [L, -Wd]];
    const lo = b.map(([dx, dy]) => corner(dx, dy, 0.12)), hi = b.map(([dx, dy]) => corner(dx, dy, 0.92));
    const faces = [];
    for (let k = 0; k < 4; k++) {
      const q = [lo[k], lo[(k + 1) % 4], hi[(k + 1) % 4], hi[k]];
      const mx = (q[0][0] + q[1][0]) / 2, my = (q[0][1] + q[1][1]) / 2;
      faces.push({ q, d: this.view(mx, my, 0.5)[2], light: 0.65 + 0.25 * ((k % 2) ? 1 : 0) });
    }
    faces.sort((a, b2) => b2.d - a.d);
    for (const f of faces) this.poly(f.q, shade("#d9dee3", f.light), "#7d868f", 1);
    this.poly(hi, shade("#d9dee3", 1.05), "#7d868f", 1);
    this.line([[p.x, p.y, 0.95], [p.x + 1.8 * c, p.y + 1.8 * s, 0.95]], this.col.ink, 3);
    this.dot(p.x, p.y, 1.3, 0.28, stateColor);
  }
}

// ------------------------------------------------------- canvas 2D map --
// Top-down map in metres (gt.csv frame, equal axes). Plotly's SVG scatter took ~34 ms per
// frame here (long tasks of 50-80 ms), so the map is drawn on a canvas as well.

class Canvas2D {
  constructor(div, S, follow, colors) {
    this.div = div; this.S = S; this.follow = follow; this.col = colors;
    this.canvas = document.createElement("canvas");
    Object.assign(this.canvas.style, { width: "100%", height: "100%", display: "block", cursor: "grab", borderRadius: "10px" });
    div.innerHTML = ""; div.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    const [x0, x1, y0, y1] = S.bounds;
    this.center = follow ? { x: S.x[0], y: S.y[0] } : { x: (x0 + x1) / 2, y: (y0 + y1) / 2 };
    this.pan = { x: 0, y: 0 };          // user offset from the followed robot (metres)
    this.halfSpan = follow ? 18 : null;  // metres from centre to the shorter edge
    this.bindInput();
    this.resize();
    this.ro = new ResizeObserver(() => { this.resize(); if (this.last) this.render(this.last); });
    this.ro.observe(div);
  }

  bindInput() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; this.canvas.setPointerCapture(e.pointerId); this.canvas.style.cursor = "grabbing"; });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = (e.clientX - drag.x) / this.scale, dy = (e.clientY - drag.y) / this.scale;
      if (this.follow) { this.pan.x -= dx; this.pan.y += dy; } else { this.center.x -= dx; this.center.y += dy; }
      drag = { x: e.clientX, y: e.clientY };
      if (this.last) this.render(this.last);
    });
    const up = () => { drag = null; this.canvas.style.cursor = "grab"; };
    this.canvas.addEventListener("pointerup", up); this.canvas.addEventListener("pointercancel", up);
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.halfSpan = Math.min(2000, Math.max(4, this.currentHalf() * Math.exp(e.deltaY * 0.0012)));
      if (this.last) this.render(this.last);
    }, { passive: false });
  }

  currentHalf() {
    if (this.halfSpan) return this.halfSpan;
    const [x0, x1, y0, y1] = this.S.bounds;           // whole route: fit the bounds
    return Math.max((x1 - x0) / 2 * (this.H / this.W), (y1 - y0) / 2);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, w = this.div.clientWidth || 600, h = this.div.clientHeight || 470;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.W = w; this.H = h; this.dpr = dpr;
  }

  destroy() { if (this.ro) this.ro.disconnect(); }

  px(x, y) { return [this.W / 2 + (x - this.c.x) * this.scale, this.H / 2 - (y - this.c.y) * this.scale]; }

  path(xs, ys, close) {
    const ctx = this.ctx; ctx.beginPath();
    let pen = false;
    for (let k = 0; k < xs.length; k++) {
      if (xs[k] === null) { pen = false; continue; }
      const [a, b] = this.px(xs[k], ys[k]);
      if (!pen) { ctx.moveTo(a, b); pen = true; } else ctx.lineTo(a, b);
    }
    if (close) ctx.closePath();
  }

  render(F) {
    this.last = F;
    const ctx = this.ctx, S = this.S, col = this.col, p = F.p;
    if (this.follow) {
      const now = performance.now(), dt = this.lastRender ? Math.min(0.2, (now - this.lastRender) / 1000) : 1;
      this.lastRender = now;
      const k = F.snap ? 1 : 1 - Math.exp(-dt / 0.18);
      this.center.x += (p.x - this.center.x) * k; this.center.y += (p.y - this.center.y) * k;
      this.c = { x: this.center.x + this.pan.x, y: this.center.y + this.pan.y };
    } else this.c = this.center;
    this.scale = Math.min(this.W, this.H) / 2 / this.currentHalf();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = "#0e141b"; ctx.fillRect(0, 0, this.W, this.H);
    // grid with metre labels
    const half = Math.max(this.W, this.H) / 2 / this.scale, raw = half / 4;
    const step = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((s) => s >= raw) || 1000;
    ctx.strokeStyle = "#1f2a35"; ctx.lineWidth = 1; ctx.fillStyle = "#8b96a3";
    ctx.font = "11px Inter, system-ui, sans-serif";
    const xa = this.c.x - this.W / 2 / this.scale, xb = this.c.x + this.W / 2 / this.scale;
    const ya = this.c.y - this.H / 2 / this.scale, yb = this.c.y + this.H / 2 / this.scale;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    for (let v = Math.ceil(xa / step) * step; v <= xb; v += step) {
      const [a] = this.px(v, 0); ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, this.H); ctx.stroke();
      ctx.fillText(`${v}`, a, this.H - 2);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(ya / step) * step; v <= yb; v += step) {
      const [, b] = this.px(0, v); ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(this.W, b); ctx.stroke();
      ctx.fillText(`${v}`, 3, b - 7);
    }
    ctx.fillText("x, y in metres (gt.csv frame)", 3, 10);
    // reference (Engineer), GPS fixes (backend EKF), trees
    if (S.ref) { ctx.setLineDash([4, 4]); ctx.strokeStyle = col.ref; ctx.lineWidth = 1.4; this.path(S.ref.x, S.ref.y); ctx.stroke(); ctx.setLineDash([]); }
    if (S.fixes && F.fixesN) {
      ctx.fillStyle = col.gps; ctx.globalAlpha = 0.55;
      for (let k = 0; k < F.fixesN; k++) { const [a, b] = this.px(S.fixes.x[k], S.fixes.y[k]); ctx.fillRect(a - 1.5, b - 1.5, 3, 3); }
      ctx.globalAlpha = 1;
    }
    const tr = Math.max(3, 0.9 * this.scale);
    for (let k = 0; k < S.trees.x.length; k++) {
      const [a, b] = this.px(S.trees.x[k], S.trees.y[k]);
      if (a < -tr || a > this.W + tr || b < -tr || b > this.H + tr) continue;
      ctx.beginPath(); ctx.arc(a, b, tr, 0, 2 * Math.PI); ctx.fillStyle = F.treeColors[k]; ctx.fill();
    }
    // trail coloured by confidence
    ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const seg of F.trail) {
      ctx.strokeStyle = seg.color; this.path(seg.pts.map((q) => q[0]), seg.pts.map((q) => q[1])); ctx.stroke();
    }
    // simulated workers (2 m zone)
    for (const w of F.workers) {
      const r = ring(w.x, w.y, w.r, 0, 48);
      this.path(r.x, r.y, true); ctx.fillStyle = "rgba(238,242,239,0.12)"; ctx.fill();
      ctx.setLineDash([3, 3]); ctx.strokeStyle = col.ink; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      const [a, b] = this.px(w.x, w.y); this.text(w.label, a, b - w.r * this.scale - 4, col.ink);
    }
    // GPS lost / restored markers
    for (const m of F.markers) {
      const [a, b] = this.px(m.x, m.y);
      ctx.strokeStyle = col.warn; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(a - 5, b - 5); ctx.lineTo(a + 5, b + 5); ctx.moveTo(a + 5, b - 5); ctx.lineTo(a - 5, b + 5); ctx.stroke();
      this.text(m.label, a, b - 8, col.warn);
    }
    // safety ring, uncertainty, robot
    if (F.safety) { ctx.setLineDash([7, 5]); ctx.strokeStyle = F.safety.color; ctx.lineWidth = 2; this.path(F.safety.x, F.safety.y, true); ctx.stroke(); ctx.setLineDash([]); }
    this.path(F.bubble.x, F.bubble.y, true); ctx.fillStyle = shade(F.stateColor, 1, 0.16); ctx.fill();
    ctx.strokeStyle = F.stateColor; ctx.lineWidth = 2; ctx.stroke();
    const [rx, ry] = this.px(p.x, p.y), hd = -p.th, sz = 11;
    ctx.save(); ctx.translate(rx, ry); ctx.rotate(hd);
    ctx.beginPath(); ctx.moveTo(sz, 0); ctx.lineTo(-sz * 0.7, sz * 0.65); ctx.lineTo(-sz * 0.35, 0); ctx.lineTo(-sz * 0.7, -sz * 0.65); ctx.closePath();
    ctx.fillStyle = F.stateColor; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#0b0f14"; ctx.stroke();
    ctx.restore();
  }

  text(t, a, b, color) {
    const ctx = this.ctx; ctx.font = "600 11px Inter, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    const w = ctx.measureText(t).width; ctx.fillStyle = "rgba(11,17,24,0.75)"; ctx.fillRect(a - w / 2 - 3, b - 14, w + 6, 14);
    ctx.fillStyle = color; ctx.fillText(t, a, b - 1);
  }
}

// ------------------------------------------------------------------ player --

class Player {
  constructor(parent) {
    this.div = document.createElement("div");
    this.div.className = "orchard-replay";
    this.div.style.position = "relative";
    parent.appendChild(this.div);
    this.sceneId = null; this.viewKey = null; this.built = false;
    this.t = 0; this.playing = false; this.seekId = null; this.speed = 1; this.end = 0;
    this.rev = 0; this.lastDraw = 0; this.lastReport = 0;
    this.cache = {};
  }

  update(data, setStateValue) {
    this.setStateValue = setStateValue;
    const S = data.scene, c = data.ctl;
    this.S = S; this.c = c;
    this.div.style.height = (c.height || 470) + "px";
    const viewKey = `${c.view}|${c.follow}`;
    if (S.id !== this.sceneId || viewKey !== this.viewKey) {
      this.sceneId = S.id; this.viewKey = viewKey; this.built = false; this.cache = {}; this.snap = true;
      this.build();
    }
    if (c.seek_id !== this.seekId) {        // the server moved the time (slider, jump, reset, demo)
      this.seekId = c.seek_id;
      this.t = c.seek_t;
      this.snap = true;
      this.anchor();
    }
    if (c.speed !== this.speed || c.end !== this.end) { this.speed = c.speed; this.end = c.end; this.anchor(); }
    if (c.playing && !this.playing) this.start();
    else if (!c.playing && this.playing) this.stop(true);
    if (!this.playing) this.draw(true);
  }

  anchor() { this.anchorT = this.t; this.anchorWall = performance.now(); }

  start() {
    this.playing = true;
    if (this.t >= this.end - 1e-6) this.t = this.c.seek_t;
    this.anchor();
    const tick = (now) => {
      if (!this.playing) return;
      this.t = simTime(this.anchorT, this.anchorWall, now, this.speed, this.end);
      const ended = this.t >= this.end;
      this.draw(false);                         // canvas: redraw on every display frame
      if (ended) { this.playing = false; this.report(false); return; }
      if (now - this.lastReport >= REPORT_MS) { this.lastReport = now; this.report(true); }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop(report) { this.playing = false; this.draw(true); if (report) this.report(false); }

  destroy() { this.destroyed = true; this.playing = false; this.setStateValue = null; if (this.view) this.view.destroy(); }

  report(playing) {
    if (this.destroyed) return;
    // monotonic across page reloads, so Python can ignore a report that arrives out of order
    this.rev = Math.max(this.rev + 1, Date.now());
    if (this.setStateValue) this.setStateValue("clock", { t: this.t, rev: this.rev, playing });
  }

  build() {
    if (this.view) this.view.destroy();
    const View = this.c.view === "3d" ? Canvas3D : Canvas2D;
    this.view = new View(this.div, this.S, this.c.follow, this.S.colors);
    this.built = true;
    this.draw(true);
  }

  // ---------------------------------------------------- shared frame state --
  frame(force) {
    const S = this.S, t = this.t, p = poseAt(S, t), col = S.colors, C = this.cache;
    // trail: rebuilt only when a new 10 Hz sample is reached; the robot itself moves every frame
    if (force || p.i !== C.trailIdx) {
      C.trailIdx = p.i;
      const step = Math.max(1, Math.ceil((p.i + 1) / TRAIL_MAX)), segs = [];
      let cur = null;
      for (let k = 0; k <= p.i; k += step) {
        const l = S.lvl[k];
        if (!cur || cur.lvl !== l) {
          if (cur) cur.pts.push([S.x[k], S.y[k]]);          // join the colour runs
          cur = { lvl: l, color: col.state[l], pts: [] }; segs.push(cur);
        }
        cur.pts.push([S.x[k], S.y[k]]);
      }
      C.trail = segs;
    }
    const trail = C.trail.map((s, k) => (k === C.trail.length - 1 ? { ...s, pts: [...s.pts, [p.x, p.y]] } : s));
    const status = treeStatus(S.trees.changes, S.trees.x.length, t), tkey = status.join("");
    if (force || tkey !== C.treeKey) {
      C.treeKey = tkey; C.status = status;
      C.treeColors = status.map((s) => S.trees.colors[String(s)]);
    }
    const ub = ellipse(p.x, p.y, p.ua, p.ub, p.ang);
    let safety = null;
    if (p.ringR !== null) { const r = ring(p.x, p.y, p.ringR, 0.07); safety = { x: r.x, y: r.y, color: col.state[p.ringLvl] }; }
    const snap = this.snap; this.snap = false;
    return { p, trail, treeColors: C.treeColors, status: C.status, bubble: ub, safety, stateColor: col.state[p.lvl],
             workers: visibleWorkers(S.workers, t), markers: S.markers.filter((m) => m.t <= t), snap,
             fixesN: S.fixes && S.fixes.t.length && t >= S.fixes.t[0] ? lastIndex(S.fixes.t, t) + 1 : 0 };
  }

  draw(force) {
    if (!this.built) return;
    const t0 = performance.now();
    const F = this.frame(force);
    this.view.render(F);
    // lightweight frame statistics (read in the browser console: window.__orchardReplayStats)
    const st = (window.__orchardReplayStats = window.__orchardReplayStats || { frames: 0, drawMs: 0, maxDrawMs: 0 });
    const ms = performance.now() - t0;
    st.frames += 1; st.drawMs += ms; st.maxDrawMs = Math.max(st.maxDrawMs, ms); st.t = this.t; st.playing = this.playing;
  }
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!data || !data.scene) return;
  let player = parentElement.__orchardReplay;
  if (!player) player = parentElement.__orchardReplay = new Player(parentElement);
  player.update(data, setStateValue);
  // Streamlit calls the latest returned cleanup when the component is removed (source or page
  // switch): stop the loop so an orphaned player never reports its clock again.
  return () => { player.destroy(); delete parentElement.__orchardReplay; };
}
