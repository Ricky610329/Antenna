/* Pattern Browser SPA — vanilla JS,無 build step。規格=../SPEC.md(v1+v2 增補)
 * 渲染方向鐵則:pixel (i,j) → rect(x=j*s, y=i*s) ⇒ i 朝下、j 朝右=饋線邊在圖下緣。
 * 菱形站點:server 給 HFSS mm 座標 [cx,cy,w],轉圖座標 x=cy/0.2*s, y=(5-cx)/0.2*s。
 * 圖語言(全站一致):序列色=已驗證色票 slot1-4(藍/橘/aqua/黃);紅虛線=規格門檻;
 * 金色底/扇=目標頻帶或 ±45° 波束窗;橘虛線=窗界。 */
"use strict";

const GRID = 25;
const COLORS = {
  metal: "#33322f", empty: "#f3f2ee",
  both: "#b8b6ae", aOnly: "#2a78d6", bOnly: "#e34948",
  site: "rgba(235,104,52,0.9)", siteEdge: "#a54312",
  bar: "#2a78d6", gridline: "#e1e0d9", axis: "#c3c2b7", muted: "#898781",
  ink: "#0b0b0b", ink2: "#52514e",
  target: "#d03b3b",                       // 紅虛線=規格門檻/G0−3dB
  bandFill: "rgba(237,161,0,0.10)",        // 金色底=目標頻帶/波束窗
  windowFill: "rgba(237,161,0,0.16)",
  windowEdge: "#eb6834",                   // 橘虛線=窗界
  ptBase: "rgba(42,120,214,0.55)", ptDim: "rgba(184,182,174,0.55)",
  sel: "#eb6834",
};
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; // 已驗證 slot1-4(疊圖 2-4 線)
const AXES = {
  wm: "wm(最差餘裕,dB)", lo: "lo(左帶外上限,dB)", rad: "rad(窗地板餘裕,dB)",
  ndiag: "ndiag(對角接點)", n8: "n8(組塊數)", total: "total(金屬格數)",
};

const S = {
  view: null,
  detailId: null,
  tray: loadLS("pb_tray_v1", []),
  groups: loadLS("pb_groups_v1", { groups: [] }),
  activeGroup: loadLS("pb_active_group", null),
  list: { offset: 0, limit: 50, sort: "wm", dir: "desc", total: 0 },
  patternCache: new Map(),   // id → /api/pattern 回應
  respCache: new Map(),      // id → /api/resp 曲線|null
  radcCache: new Map(),      // id → /api/radc 曲線|null
  targets: null,
  statsLoaded: false,
  ovMode: loadLS("pb_ov_mode", "table"),
  cmpTab: "pattern",
  sc: { pts: [], sel: new Set() },   // 散點狀態
  pareto: { pts: [] },
  mfg: { axis: "wm", rows: [], selId: null },
};

/* ---------------- 小工具 ---------------- */
function $(sel) { return document.querySelector(sel); }
function loadLS(key, fallback) {
  try { const v = JSON.parse(localStorage.getItem(key)); return v == null ? fallback : v; }
  catch (e) { return fallback; }
}
function saveLS(key, v) { localStorage.setItem(key, JSON.stringify(v)); }
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kd of kids) if (kd != null) n.append(kd);
  return n;
}
let msgTimer = null;
function toast(msg, isErr = false) {
  const m = $("#msg");
  m.textContent = msg;
  m.className = isErr ? "err" : "";
  m.hidden = false;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => { m.hidden = true; }, 3200);
}
async function api(path) {
  const r = await fetch(path);
  let body = null;
  try { body = await r.json(); } catch (e) { /* 非 JSON */ }
  if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
  return body;
}
function fmt(v, digits = 2) {
  return (v === null || v === undefined) ? "—" : Number(v).toFixed(digits);
}
function b64ToBits(b64) {
  const bin = atob(b64);
  const bits = new Uint8Array(GRID * GRID);
  for (let k = 0; k < bits.length; k++) {
    bits[k] = (bin.charCodeAt(k >> 3) >> (7 - (k & 7))) & 1;   // np.packbits=MSB first
  }
  return bits;
}
function hamming(a, b) {
  let d = 0;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) d++;
  return d;
}
async function getTargets() {
  if (!S.targets) S.targets = await api("/api/targets");
  return S.targets;
}
async function getCurves(kind, ids) {
  // kind='resp'|'radc';逐 id 快取,未快取的一次批抓;回 Map(id → 曲線|null)
  const cache = kind === "resp" ? S.respCache : S.radcCache;
  const need = ids.filter((id) => !cache.has(id));
  if (need.length) {
    const data = await api(`/api/${kind}?ids=${need.map(encodeURIComponent).join(",")}`);
    for (const id of need) cache.set(id, data[id] === undefined ? null : data[id]);
  }
  return new Map(ids.map((id) => [id, cache.get(id)]));
}

/* ---------------- tooltip(自製,hover 浮層) ---------------- */
function showTip(pageX, pageY, text) {
  const tip = $("#tooltip");
  tip.textContent = text;
  tip.hidden = false;
  const maxLeft = window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8;
  tip.style.left = `${Math.max(4, Math.min(pageX, maxLeft))}px`;
  tip.style.top = `${pageY}px`;
}
function hideTip() { $("#tooltip").hidden = true; }
function initTips() {
  // 事件委派:任何帶 data-tip 的元素 hover 即浮出;canvas 自帶座標型 tooltip,不在此蓋掉
  document.addEventListener("mouseover", (ev) => {
    const tgt = ev.target;
    if (!(tgt instanceof Element)) return;
    if (tgt.tagName === "CANVAS") return;
    const t = tgt.closest("[data-tip]");
    if (!t) { hideTip(); return; }
    const r = t.getBoundingClientRect();
    showTip(window.scrollX + r.left, window.scrollY + r.bottom + 6, t.dataset.tip);
  });
  document.addEventListener("mouseleave", hideTip);
}

/* ---------------- pattern 繪圖 ---------------- */
function hexToRgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}
const RGB_METAL = hexToRgb(COLORS.metal), RGB_EMPTY = hexToRgb(COLORS.empty);
function makeThumb(bits, cls = "thumb") {
  // 25x25 ImageData + CSS 放大(image-rendering: pixelated);i 朝下=直接照索引寫入
  const cv = el("canvas", { class: cls, width: GRID, height: GRID });
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(GRID, GRID);
  for (let k = 0; k < GRID * GRID; k++) {
    const c = bits[k] ? RGB_METAL : RGB_EMPTY;
    img.data.set([c[0], c[1], c[2], 255], k * 4);
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}
function drawBig(cv, bits, opts = {}) {
  const s = Math.floor(cv.width / GRID);
  const ctx = cv.getContext("2d");
  ctx.fillStyle = COLORS.empty;
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = COLORS.metal;
  for (let k = 0; k < GRID * GRID; k++) {
    if (bits[k]) ctx.fillRect((k % GRID) * s, Math.floor(k / GRID) * s, s, s);  // 鐵則:x=j*s, y=i*s
  }
  ctx.strokeStyle = "rgba(11,11,11,0.07)";
  ctx.lineWidth = 1;
  for (let t = 0; t <= GRID; t++) {
    ctx.beginPath(); ctx.moveTo(t * s + 0.5, 0); ctx.lineTo(t * s + 0.5, GRID * s); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, t * s + 0.5); ctx.lineTo(GRID * s, t * s + 0.5); ctx.stroke();
  }
  if (opts.sites && opts.showSites) drawSites(ctx, opts.sites, s);
}
function drawSites(ctx, sites, s) {
  // [cx,cy,w] HFSS mm → 圖座標 x=cy/0.2*s, y=(5-cx)/0.2*s;45° 菱形半對角=w*√2/2
  for (const [cx, cy, w] of sites) {
    const x = cy / 0.2 * s, y = (5 - cx) / 0.2 * s, r = (w * Math.SQRT2 / 2) / 0.2 * s;
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
    ctx.closePath();
    ctx.fillStyle = COLORS.site;
    ctx.fill();
    ctx.strokeStyle = COLORS.siteEdge;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}
function drawXor(cv, bitsA, bitsB) {
  const s = Math.floor(cv.width / GRID);
  const ctx = cv.getContext("2d");
  ctx.fillStyle = COLORS.empty;
  ctx.fillRect(0, 0, cv.width, cv.height);
  for (let k = 0; k < GRID * GRID; k++) {
    const a = bitsA[k], b = bitsB[k];
    if (!a && !b) continue;
    ctx.fillStyle = (a && b) ? COLORS.both : (a ? COLORS.aOnly : COLORS.bOnly);
    ctx.fillRect((k % GRID) * s, Math.floor(k / GRID) * s, s, s);
  }
}

/* ---------------- 長條圖(單序列,hover tooltip) ---------------- */
function drawBars(cv, items) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pad = { l: 34, r: 6, t: 8, b: 18 };
  ctx.clearRect(0, 0, W, H);
  cv._bars = [];
  if (!items.length) {
    ctx.fillStyle = COLORS.muted; ctx.font = "12px system-ui"; ctx.fillText("無資料", W / 2 - 18, H / 2);
    return;
  }
  const maxC = Math.max(...items.map((d) => d.count), 1);
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const step = plotW / items.length;
  const barW = Math.max(2, Math.min(step - 2, step * 0.8));
  ctx.font = "10px system-ui";
  for (const frac of [0.5, 1]) {
    const yv = Math.round(maxC * frac);
    const y = pad.t + plotH - (yv / maxC) * plotH;
    ctx.strokeStyle = COLORS.gridline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y + 0.5); ctx.lineTo(W - pad.r, y + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "right";
    ctx.fillText(String(yv), pad.l - 4, y + 3);
  }
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + plotH + 0.5); ctx.lineTo(W - pad.r, pad.t + plotH + 0.5); ctx.stroke();
  const labelEvery = Math.ceil(items.length / 8);
  items.forEach((d, i) => {
    const h = (d.count / maxC) * plotH;
    const x = pad.l + i * step + (step - barW) / 2;
    const y = pad.t + plotH - h;
    ctx.fillStyle = COLORS.bar;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, barW, h, [2, 2, 0, 0]); else ctx.rect(x, y, barW, h);
    ctx.fill();
    cv._bars.push({ x, y: pad.t, w: barW, h: plotH, label: d.label, count: d.count });
    if (i % labelEvery === 0) {
      ctx.fillStyle = COLORS.muted; ctx.textAlign = "center";
      ctx.fillText(String(d.label), x + barW / 2, H - 5);
    }
  });
  if (!cv._hoverBound) {
    cv._hoverBound = true;
    cv.addEventListener("mousemove", (ev) => {
      const rect = cv.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const hit = (cv._bars || []).find((b) => mx >= b.x - 1 && mx <= b.x + b.w + 1);
      if (hit) showTip(ev.pageX + 12, ev.pageY - 24, `${hit.label}:${hit.count} 筆`);
      else hideTip();
    });
    cv.addEventListener("mouseleave", hideTip);
  }
}

/* ---------------- 曲線圖(多序列折線,軸+格線+目標線+內帶底色+crosshair) ---------------- */
function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}
function drawEmptyChart(cv, text) {
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = COLORS.muted; ctx.font = "13px system-ui"; ctx.textAlign = "center";
  ctx.fillText(text, cv.width / 2, cv.height / 2);
  ctx.textAlign = "left";
}
function drawLineChart(cv, o, hoverIdx = -1) {
  // o={series:[{label,color,ys}], xs, band:[x1,x2]|null, hline, hlineLabel, yLabel, xLabel}
  cv._opts = o;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pad = { l: 46, r: 14, t: 14, b: 28 };
  ctx.clearRect(0, 0, W, H);
  if (!o || !o.series.length) { drawEmptyChart(cv, "無曲線資料"); return; }
  const xs = o.xs;
  const xmin = xs[0], xmax = xs[xs.length - 1];
  let ymin = Infinity, ymax = -Infinity;
  for (const s of o.series) for (const v of s.ys) if (v != null && isFinite(v)) {
    ymin = Math.min(ymin, v); ymax = Math.max(ymax, v);
  }
  if (!isFinite(ymin)) { drawEmptyChart(cv, "無曲線資料"); return; }
  if (o.hline != null) { ymin = Math.min(ymin, o.hline); ymax = Math.max(ymax, o.hline); }
  const padY = Math.max((ymax - ymin) * 0.08, 0.5);
  ymin -= padY; ymax += padY;
  const X = (x) => pad.l + (x - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const Y = (y) => pad.t + (ymax - y) / (ymax - ymin) * (H - pad.t - pad.b);
  // 內帶底色(金)=目標頻帶
  if (o.band) {
    ctx.fillStyle = COLORS.bandFill;
    ctx.fillRect(X(o.band[0]), pad.t, X(o.band[1]) - X(o.band[0]), H - pad.t - pad.b);
  }
  // y 格線+刻度
  ctx.font = "10px system-ui";
  for (const tv of niceTicks(ymin, ymax, 5)) {
    const y = Y(tv);
    ctx.strokeStyle = COLORS.gridline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y + 0.5); ctx.lineTo(W - pad.r, y + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "right";
    ctx.fillText(String(tv), pad.l - 5, y + 3);
  }
  // x 刻度(每 1GHz)
  ctx.textAlign = "center";
  for (let f = Math.ceil(xmin); f <= xmax + 1e-9; f += 1) {
    const x = X(f);
    ctx.strokeStyle = COLORS.gridline;
    ctx.beginPath(); ctx.moveTo(x + 0.5, pad.t); ctx.lineTo(x + 0.5, H - pad.b); ctx.stroke();
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(String(f), x, H - pad.b + 13);
  }
  // 基線
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b + 0.5); ctx.lineTo(W - pad.r, H - pad.b + 0.5); ctx.stroke();
  // 窗界(橘虛線)
  if (o.band) {
    ctx.strokeStyle = COLORS.windowEdge; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    for (const bx of o.band) {
      ctx.beginPath(); ctx.moveTo(X(bx) + 0.5, pad.t); ctx.lineTo(X(bx) + 0.5, H - pad.b); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  // 目標線(紅虛線)
  if (o.hline != null) {
    const y = Y(o.hline);
    ctx.strokeStyle = COLORS.target; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.setLineDash([]);
    if (o.hlineLabel) {
      ctx.fillStyle = COLORS.target; ctx.textAlign = "right";
      ctx.fillText(o.hlineLabel, W - pad.r - 2, y - 4);
    }
  }
  // 序列折線(null=斷線)
  for (const s of o.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    let pen = false;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const v = s.ys[i];
      if (v == null || !isFinite(v)) { pen = false; continue; }
      if (pen) ctx.lineTo(X(xs[i]), Y(v)); else { ctx.moveTo(X(xs[i]), Y(v)); pen = true; }
    }
    ctx.stroke();
  }
  // hover crosshair+節點
  if (hoverIdx >= 0 && hoverIdx < xs.length) {
    const x = X(xs[hoverIdx]);
    ctx.strokeStyle = COLORS.axis; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    for (const s of o.series) {
      const v = s.ys[hoverIdx];
      if (v == null || !isFinite(v)) continue;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(x, Y(v), 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  // 軸標
  ctx.fillStyle = COLORS.ink2; ctx.textAlign = "left";
  if (o.yLabel) ctx.fillText(o.yLabel, 4, 11);
  if (o.xLabel) { ctx.textAlign = "right"; ctx.fillText(o.xLabel, W - 4, H - 4); }
  ctx.textAlign = "left";
  cv._plot = { X, pad, W, H };
  if (!cv._lineBound) {
    cv._lineBound = true;
    cv.addEventListener("mousemove", (ev) => {
      const o2 = cv._opts;
      if (!o2 || !o2.series.length) return;
      const rect = cv.getBoundingClientRect();
      const mx = (ev.clientX - rect.left) * (cv.width / rect.width);
      let best = -1, bestD = 1e9;
      for (let i = 0; i < o2.xs.length; i++) {
        const d = Math.abs(cv._plot.X(o2.xs[i]) - mx);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (bestD > 24) { drawLineChart(cv, o2); hideTip(); return; }
      drawLineChart(cv, o2, best);
      const lines = [`${o2.xs[best]} GHz`];
      for (const s of o2.series) {
        const v = s.ys[best];
        lines.push(`${s.label}:${v == null ? "—" : v.toFixed(2)} dB`);
      }
      showTip(ev.pageX + 14, ev.pageY - 10, lines.join("\n"));
    });
    cv.addEventListener("mouseleave", () => { if (cv._opts) drawLineChart(cv, cv._opts); hideTip(); });
  }
}

/* ---------------- rad 極座標(φ 切面;主波束朝上) ---------------- */
function drawPolar(cv, o, hoverIdx = -1) {
  // o={theta:[181], series:[{label,color,vals}], windowDeg}
  cv._popts = o;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!o || !o.series.length) { drawEmptyChart(cv, "無曲線資料"); return; }
  let vmax = -Infinity;
  for (const s of o.series) for (const v of s.vals) if (v != null && isFinite(v)) vmax = Math.max(vmax, v);
  if (!isFinite(vmax)) { drawEmptyChart(cv, "無曲線資料"); return; }
  const cx = W / 2, cy = H - 22;
  const R = Math.min(W / 2 - 34, H - 52);
  const rmax = Math.ceil(vmax / 5) * 5;
  const span = 30;                          // 6 環 × 5dB
  const rOf = (v) => R * Math.max(0, Math.min(1, (v - (rmax - span)) / span));
  const P = (deg, r) => [cx + r * Math.sin(deg * Math.PI / 180), cy - r * Math.cos(deg * Math.PI / 180)];
  const wdeg = o.windowDeg || 45;
  // 金色 ±window 扇形
  ctx.fillStyle = COLORS.windowFill;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, R, -Math.PI / 2 - wdeg * Math.PI / 180, -Math.PI / 2 + wdeg * Math.PI / 180);
  ctx.closePath();
  ctx.fill();
  // 5dB 環(上半圓)+環標
  ctx.font = "9px system-ui";
  for (let k = 0; k <= span / 5; k++) {
    const rr = R * (1 - k * 5 / span);
    if (rr <= 0) continue;
    ctx.strokeStyle = COLORS.gridline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, rr, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "left";
    ctx.fillText(String(rmax - k * 5), cx + 3, cy - rr - 2);
  }
  // 放射線:±90 底線+0 軸(灰)、±45 窗界(橘虛線)
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(cx - R, cy + 0.5); ctx.lineTo(cx + R, cy + 0.5); ctx.stroke();
  ctx.strokeStyle = COLORS.gridline;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R); ctx.stroke();
  ctx.strokeStyle = COLORS.windowEdge; ctx.setLineDash([4, 3]);
  for (const a of [-wdeg, wdeg]) {
    const [x, y] = P(a, R);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
  }
  ctx.setLineDash([]);
  // θ 刻度標
  ctx.fillStyle = COLORS.muted; ctx.textAlign = "center";
  for (const a of [-90, -45, 0, 45, 90]) {
    const [x, y] = P(a, R + 13);
    ctx.fillText(`${a}°`, x, y + 3);
  }
  // 紅虛圈=G0−3dB(G0=各曲線 θ=0 的最大值)
  const i0 = o.theta.reduce((bi, t, i) => Math.abs(t) < Math.abs(o.theta[bi]) ? i : bi, 0);
  let g0 = -Infinity;
  for (const s of o.series) {
    const v = s.vals[i0];
    if (v != null && isFinite(v)) g0 = Math.max(g0, v);
  }
  if (isFinite(g0)) {
    const rr = rOf(g0 - 3);
    if (rr > 0) {
      ctx.strokeStyle = COLORS.target; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, rr, Math.PI, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.target; ctx.textAlign = "left";
      ctx.fillText("G0−3dB", cx - R + 2, cy - rr - 3);
    }
  }
  // 序列曲線
  for (const s of o.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    let pen = false;
    ctx.beginPath();
    for (let i = 0; i < o.theta.length; i++) {
      const v = s.vals[i];
      if (v == null || !isFinite(v)) { pen = false; continue; }
      const [x, y] = P(o.theta[i], rOf(v));
      if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
    }
    ctx.stroke();
  }
  // hover 節點
  if (hoverIdx >= 0 && hoverIdx < o.theta.length) {
    for (const s of o.series) {
      const v = s.vals[hoverIdx];
      if (v == null || !isFinite(v)) continue;
      const [x, y] = P(o.theta[hoverIdx], rOf(v));
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.strokeStyle = COLORS.axis; ctx.setLineDash([2, 3]);
    const [x, y] = P(o.theta[hoverIdx], R);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.setLineDash([]);
  }
  cv._pgeom = { cx, cy, R };
  if (!cv._polarBound) {
    cv._polarBound = true;
    cv.addEventListener("mousemove", (ev) => {
      const o2 = cv._popts;
      if (!o2 || !o2.series.length || !cv._pgeom) return;
      const rect = cv.getBoundingClientRect();
      const mx = (ev.clientX - rect.left) * (cv.width / rect.width);
      const my = (ev.clientY - rect.top) * (cv.height / rect.height);
      const dx = mx - cv._pgeom.cx, dy = my - cv._pgeom.cy;
      const ang = Math.atan2(dx, -dy) * 180 / Math.PI;
      if (dy > 4 || Math.hypot(dx, dy) > cv._pgeom.R + 14 || Math.abs(ang) > 92) {
        drawPolar(cv, o2); hideTip(); return;
      }
      let best = 0, bestD = 1e9;
      for (let i = 0; i < o2.theta.length; i++) {
        const d = Math.abs(o2.theta[i] - ang);
        if (d < bestD) { bestD = d; best = i; }
      }
      drawPolar(cv, o2, best);
      const lines = [`θ=${o2.theta[best]}°`];
      for (const s of o2.series) {
        const v = s.vals[best];
        lines.push(`${s.label}:${v == null ? "—" : v.toFixed(2)} dB`);
      }
      showTip(ev.pageX + 14, ev.pageY - 10, lines.join("\n"));
    });
    cv.addEventListener("mouseleave", () => { if (cv._popts) drawPolar(cv, cv._popts); hideTip(); });
  }
}
function legendInto(box, series, onClick = null) {
  box.textContent = "";
  for (const s of series) {
    box.append(el("span", { class: "leg-item" + (onClick ? " clickable" : ""),
      ...(onClick ? { onclick: () => onClick(s) } : {}) },
      el("span", { class: "sw", style: `background:${s.color}` }),
      el("span", { text: s.label })));
  }
}

/* ---------------- 路由 ---------------- */
const VIEWS = ["overview", "detail", "compare", "groups", "mfg", "research", "help"];
function nav(view, arg) {
  location.hash = view + (arg ? "/" + encodeURIComponent(arg) : "");
}
function route() {
  const h = location.hash.slice(1) || "overview";
  const slash = h.indexOf("/");
  const view = slash < 0 ? h : h.slice(0, slash);
  const arg = slash < 0 ? null : decodeURIComponent(h.slice(slash + 1));
  const v = VIEWS.includes(view) ? view : "overview";
  S.view = v;
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== v;
  document.querySelectorAll("#nav button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === v);
  });
  if (v === "overview") initOverview();
  else if (v === "detail") { if (arg) loadDetail(arg); else renderDetailShell(); }
  else if (v === "compare") renderCompare();
  else if (v === "groups") renderGroups();
  else if (v === "mfg") renderMfg();
  else if (v === "research") renderResearch();
  else if (v === "help") renderHelp(arg);
}
window.addEventListener("hashchange", route);

/* ---------------- 總攬 ---------------- */
async function initOverview() {
  if (!S.statsLoaded) {
    S.statsLoaded = true;
    try {
      const st = await api("/api/stats");
      const cards = [
        ["總筆數", st.total], ["合格(wm≥0.15∧rad≥0)", st.qualified], ["有 wm 值", st.wm_nonnull],
        ["db100 消融", st.db100_count], ["sl100 消融", st.sl100_count],
        ["有 S11/Gain 曲線", st.resp_count == null ? "—" : st.resp_count],
        ["有 rad 曲線", st.rad_count == null ? "—" : st.rad_count],
      ];
      const box = $("#stat-cards");
      box.textContent = "";
      for (const [k, v] of cards) {
        box.append(el("div", { class: "card" },
          el("div", { class: "v", text: String(v) }), el("div", { class: "k", text: k })));
      }
      drawBars($("#chart-n8"), st.n8_hist.map(([v, c]) => ({ label: v, count: c })));
      drawBars($("#chart-ndiag"), st.ndiag_hist.map(([v, c]) => ({ label: v, count: c })));
      const wmItems = (st.wm_hist.counts || []).map((c, i) => ({
        label: st.wm_hist.edges[i].toFixed(1), count: c,
      }));
      drawBars($("#chart-wm"), wmItems);
    } catch (e) { S.statsLoaded = false; toast(`載入統計失敗:${e.message}`, true); }
  }
  setOvMode(S.ovMode);
}
function setOvMode(m) {
  S.ovMode = m;
  saveLS("pb_ov_mode", m);
  document.querySelectorAll("#ov-mode button").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === m);
  });
  $("#ov-table").hidden = m !== "table";
  $("#ov-wall").hidden = m !== "wall";
  $("#ov-scatter").hidden = m !== "scatter";
  $("#pager").hidden = m === "scatter";
  if (S.view === "overview") {
    if (m === "scatter") renderScatter(); else fetchList();
  }
}
function listQuery() {
  const p = new URLSearchParams();
  p.set("offset", S.list.offset);
  p.set("limit", S.list.limit);
  p.set("sort", S.list.sort);
  p.set("dir", S.list.dir);
  if ($("#f-qual").checked) p.set("f_qual", "1");
  if ($("#f-has-db").checked) p.set("f_has_db100", "1");
  if ($("#f-has-sl").checked) p.set("f_has_sl100", "1");
  for (const [inp, key] of [["#f-diag-min", "f_diag_min"], ["#f-diag-max", "f_diag_max"],
    ["#f-n8-min", "f_n8_min"], ["#f-n8-max", "f_n8_max"],
    ["#f-wm-min", "f_wm_min"], ["#f-lo-max", "f_lo_max"]]) {
    const v = $(inp).value.trim();
    if (v !== "") p.set(key, v);
  }
  const q = $("#f-q").value.trim();
  if (q) p.set("q", q);
  return p.toString();
}
async function fetchList() {
  let data;
  try { data = await api(`/api/list?${listQuery()}`); }
  catch (e) { toast(`載入列表失敗:${e.message}`, true); return; }
  S.list.total = data.total;
  $("#list-info").textContent =
    `符合 ${data.total} 筆;顯示 ${data.total ? data.offset + 1 : 0}–${data.offset + data.rows.length}`;
  document.querySelectorAll("th.sortable").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.sort === S.list.sort);
    th.textContent = th.dataset.sort + (th.dataset.sort === S.list.sort ? (S.list.dir === "asc" ? " ↑" : " ↓") : "");
  });
  if (S.ovMode === "wall") renderWall(data); else renderListTable(data);
  const last = data.offset + data.rows.length;
  $("#pg-info").textContent = S.list.total ? `${data.offset + 1}–${last} / ${S.list.total}` : "0 筆";
  $("#pg-prev").disabled = data.offset <= 0;
  $("#pg-next").disabled = last >= S.list.total;
}
function renderListTable(data) {
  const tb = $("#list-table tbody");
  tb.textContent = "";
  for (const row of data.rows) {
    const wmCell = el("td", { class: "num" + (row.wm !== null && row.wm >= 0 ? " wm-pos" : ""), text: fmt(row.wm) });
    const abl = el("td", {});
    if (row.has_db100) abl.append(el("span", { class: "abl-badge", text: `db100 ${fmt(row.db100_wm)}` }));
    if (row.has_sl100) abl.append(el("span", { class: "abl-badge", text: `sl100 ${fmt(row.sl100_wm)}` }));
    if (!row.has_db100 && !row.has_sl100) abl.textContent = "—";
    const idCell = el("td", {}, el("span", { text: row.id }));
    if (row.has_resp || row.has_rad) {
      idCell.append(el("span", { class: "curve-flags",
        "data-tip": `有曲線資料:${row.has_resp ? " S11/Gain" : ""}${row.has_rad ? " rad" : ""}`,
        text: ` ${row.has_resp ? "∿" : ""}${row.has_rad ? "◠" : ""}` }));
    }
    const tr = el("tr", { onclick: () => nav("detail", row.id) },
      el("td", {}, makeThumb(b64ToBits(row.bits_b64))),
      idCell,
      wmCell,
      el("td", { class: "num", text: fmt(row.rad) }),
      el("td", { class: "num", text: fmt(row.lo) }),
      el("td", { class: "num", text: String(row.ndiag) }),
      el("td", { class: "num", text: String(row.n8) }),
      el("td", { class: "num", text: String(row.total) }),
      abl,
      el("td", { text: row.kind || "—" }),
      el("td", { text: row.store || "—" }),
      el("td", {},
        el("button", { text: "比對＋", onclick: (ev) => { ev.stopPropagation(); addTray(row.id); } }),
        " ",
        el("button", { text: "群組＋", onclick: (ev) => { ev.stopPropagation(); addToActiveGroup(row.id); } })),
    );
    tb.append(tr);
  }
}
function renderWall(data) {
  const grid = $("#ov-wall");
  grid.textContent = "";
  if (!data.rows.length) { grid.append(el("div", { class: "muted", text: "沒有符合的 pattern" })); return; }
  for (const row of data.rows) {
    const tip = `${row.id}\nwm ${fmt(row.wm)} · rad ${fmt(row.rad)} · lo ${fmt(row.lo)}`
      + `\nndiag ${row.ndiag} · n8 ${row.n8} · total ${row.total}`
      + (row.has_db100 ? `\ndb100_wm ${fmt(row.db100_wm)}` : "")
      + (row.kind ? `\nkind ${row.kind}` : "") + (row.store ? ` · ${row.store}` : "");
    grid.append(el("div", { class: "wall-card", "data-tip": tip, onclick: () => nav("detail", row.id) },
      makeThumb(b64ToBits(row.bits_b64), "thumb wall"),
      el("div", { class: "tid", text: row.id })));
  }
}

/* ---- 總攬:散點模式(點=進詳情;拖框=送比對/建群組) ---- */
function scatterQuery() {
  const p = new URLSearchParams(listQuery());
  p.set("lite", "1"); p.set("limit", "50000"); p.set("offset", "0");
  return p.toString();
}
async function renderScatter() {
  const xk = $("#sc-x").value, yk = $("#sc-y").value;
  let data;
  try { data = await api(`/api/list?${scatterQuery()}`); }
  catch (e) { toast(`載入散點資料失敗:${e.message}`, true); return; }
  const pts = [];
  for (const r of data.rows) {
    if (r[xk] == null || r[yk] == null) continue;
    pts.push({ id: r.id, x: +r[xk], y: +r[yk], wm: r.wm, rad: r.rad, lo: r.lo });
  }
  S.sc.pts = pts;
  S.sc.xk = xk; S.sc.yk = yk;
  S.sc.sel = new Set([...S.sc.sel].filter((id) => pts.some((p) => p.id === id)));
  $("#scatter-info").textContent = `畫 ${pts.length} 點(兩軸皆有值)/篩選符合 ${data.total} 筆`;
  $("#list-info").textContent = `符合 ${data.total} 筆`;
  scatterRedraw();
  updateScatterActions();
}
function scatterRedraw(hoverI = -1, rect = null) {
  const cv = $("#scatter-canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pad = { l: 54, r: 16, t: 12, b: 36 };
  ctx.clearRect(0, 0, W, H);
  const pts = S.sc.pts;
  if (!pts.length) { drawEmptyChart(cv, "沒有可畫的點(檢查篩選/軸是否有值)"); return; }
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of pts) {
    xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x);
    ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y);
  }
  const px = Math.max((xmax - xmin) * 0.05, 0.5), py = Math.max((ymax - ymin) * 0.05, 0.5);
  xmin -= px; xmax += px; ymin -= py; ymax += py;
  const X = (x) => pad.l + (x - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const Y = (y) => pad.t + (ymax - y) / (ymax - ymin) * (H - pad.t - pad.b);
  ctx.font = "10px system-ui";
  for (const tv of niceTicks(ymin, ymax, 6)) {
    const y = Y(tv);
    ctx.strokeStyle = COLORS.gridline;
    ctx.beginPath(); ctx.moveTo(pad.l, y + 0.5); ctx.lineTo(W - pad.r, y + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "right"; ctx.fillText(String(tv), pad.l - 5, y + 3);
  }
  for (const tv of niceTicks(xmin, xmax, 8)) {
    const x = X(tv);
    ctx.strokeStyle = COLORS.gridline;
    ctx.beginPath(); ctx.moveTo(x + 0.5, pad.t); ctx.lineTo(x + 0.5, H - pad.b); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "center"; ctx.fillText(String(tv), x, H - pad.b + 13);
  }
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b + 0.5); ctx.lineTo(W - pad.r, H - pad.b + 0.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l + 0.5, pad.t); ctx.lineTo(pad.l + 0.5, H - pad.b); ctx.stroke();
  ctx.fillStyle = COLORS.ink2; ctx.textAlign = "right";
  ctx.fillText(AXES[S.sc.xk] || S.sc.xk, W - 4, H - 4);
  ctx.textAlign = "left";
  ctx.fillText(AXES[S.sc.yk] || S.sc.yk, 4, 11);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    p.px = X(p.x); p.py = Y(p.y);
    const selected = S.sc.sel.has(p.id);
    ctx.fillStyle = selected ? COLORS.sel : COLORS.ptBase;
    ctx.beginPath(); ctx.arc(p.px, p.py, selected ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    if (selected) { ctx.strokeStyle = COLORS.siteEdge; ctx.lineWidth = 1; ctx.stroke(); }
  }
  if (hoverI >= 0) {
    const p = pts[hoverI];
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p.px, p.py, 5.5, 0, Math.PI * 2); ctx.stroke();
  }
  if (rect) {
    ctx.strokeStyle = COLORS.sel; ctx.setLineDash([5, 3]); ctx.lineWidth = 1;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(235,104,52,0.08)";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  }
}
function scatterHit(mx, my) {
  let best = -1, bestD = 100;
  const pts = S.sc.pts;
  for (let i = 0; i < pts.length; i++) {
    const d = (pts[i].px - mx) ** 2 + (pts[i].py - my) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
function updateScatterActions() {
  const n = S.sc.sel.size;
  $("#scatter-actions").hidden = n === 0;
  $("#scatter-selinfo").textContent = `已框選 ${n} 筆`;
}
function initScatterEvents() {
  const cv = $("#scatter-canvas");
  let drag = null;
  const mpos = (ev) => {
    const r = cv.getBoundingClientRect();
    return [(ev.clientX - r.left) * (cv.width / r.width), (ev.clientY - r.top) * (cv.height / r.height)];
  };
  cv.addEventListener("mousedown", (ev) => {
    const [mx, my] = mpos(ev);
    drag = { x0: mx, y0: my, moved: false };
  });
  cv.addEventListener("mousemove", (ev) => {
    const [mx, my] = mpos(ev);
    if (drag) {
      if (Math.abs(mx - drag.x0) + Math.abs(my - drag.y0) > 6) drag.moved = true;
      if (drag.moved) {
        hideTip();
        scatterRedraw(-1, { x: Math.min(drag.x0, mx), y: Math.min(drag.y0, my),
          w: Math.abs(mx - drag.x0), h: Math.abs(my - drag.y0) });
      }
      return;
    }
    const i = scatterHit(mx, my);
    scatterRedraw(i);
    if (i >= 0) {
      const p = S.sc.pts[i];
      showTip(ev.pageX + 14, ev.pageY - 10,
        `${p.id}\n${S.sc.xk}=${p.x} · ${S.sc.yk}=${p.y}\nwm ${fmt(p.wm)} · rad ${fmt(p.rad)} · lo ${fmt(p.lo)}\n點一下進詳情`);
    } else hideTip();
  });
  cv.addEventListener("mouseup", (ev) => {
    const [mx, my] = mpos(ev);
    if (drag && drag.moved) {
      const x1 = Math.min(drag.x0, mx), x2 = Math.max(drag.x0, mx);
      const y1 = Math.min(drag.y0, my), y2 = Math.max(drag.y0, my);
      const hit = S.sc.pts.filter((p) => p.px >= x1 && p.px <= x2 && p.py >= y1 && p.py <= y2);
      if (ev.shiftKey) for (const p of hit) S.sc.sel.add(p.id);
      else S.sc.sel = new Set(hit.map((p) => p.id));
      drag = null;
      scatterRedraw();
      updateScatterActions();
      if (hit.length) toast(`框選 ${hit.length} 筆(Shift+拖=累加)`);
      return;
    }
    drag = null;
    const i = scatterHit(mx, my);
    if (i >= 0) nav("detail", S.sc.pts[i].id);
  });
  cv.addEventListener("mouseleave", () => { drag = null; scatterRedraw(); hideTip(); });
  $("#sc-to-compare").addEventListener("click", () => {
    const ids = [...S.sc.sel];
    if (ids.length < 2 || ids.length > 4) { toast("比對需 2–4 筆(重新框選或先清除)", true); return; }
    S.tray = ids;
    saveLS("pb_tray_v1", S.tray);
    updateTrayBadge();
    nav("compare");
  });
  $("#sc-to-group").addEventListener("click", () => {
    const ids = [...S.sc.sel];
    if (!ids.length) { toast("尚未框選任何點", true); return; }
    addIdsToGroup(ids);
  });
  $("#sc-clear").addEventListener("click", () => {
    S.sc.sel = new Set();
    scatterRedraw();
    updateScatterActions();
  });
}

/* ---------------- 詳情 ---------------- */
async function getPattern(id) {
  if (S.patternCache.has(id)) return S.patternCache.get(id);
  const p = await api(`/api/pattern/${encodeURIComponent(id)}`);
  S.patternCache.set(id, p);
  return p;
}
function renderDetailShell() {
  $("#detail-empty").hidden = !!S.detailId;
  $("#detail-body").hidden = !S.detailId;
}
async function loadDetail(id) {
  let p;
  try { p = await getPattern(id); }
  catch (e) { toast(`載入 ${id} 失敗:${e.message}`, true); return; }
  S.detailId = id;
  renderDetailShell();
  const bits = p.bits;
  const redraw = () => drawBig($("#detail-canvas"), bits,
    { sites: p.diag_sites, showSites: $("#ov-sites").checked });
  $("#ov-sites").onchange = redraw;
  redraw();
  $("#site-count").textContent = String(p.diag_sites.length);
  $("#detail-id").textContent = p.id + (p.variant ? `(變體 ${p.variant},bits 同親本 ${p.variant_of})` : "");
  const dl = $("#detail-meta");
  dl.textContent = "";
  const fields = [
    ["kind", p.kind || "—"], ["store", p.store || "—"], ["folder", p.folder || "—"],
    ["wm(worst margin)", fmt(p.wm)], ["rad", fmt(p.rad)], ["lo(oob_gain_max_lo)", fmt(p.lo)],
    ["sel", fmt(p.sel)], ["total(金屬格數)", String(p.total)],
    ["n4 / n8(組塊)", `${p.n4} / ${p.n8}`], ["largest8_frac", fmt(p.largest8_frac, 4)],
    ["ndiag(對角接點)", String(p.ndiag)],
    ["曲線資料", `${p.has_resp ? "S11/Gain ✓" : "S11/Gain —"} · ${p.has_rad ? "rad ✓" : "rad —"}`],
  ];
  for (const [k, v] of fields) dl.append(el("dt", { text: k }), el("dd", { text: v }));
  // 消融變體對照(wm 數字卡;曲線疊圖在下方曲線區)
  const vp = $("#variant-panel");
  vp.textContent = "";
  if (p.has_db100 || p.has_sl100) {
    vp.append(el("h3", { text: "消融變體對照(wm)" }));
    const cards = el("div", { class: "variant-cards" });
    cards.append(el("div", { class: "card" },
      el("div", { class: "v", text: fmt(p.wm) }), el("div", { class: "k", text: "本體" })));
    for (const [flag, key, name] of [["has_db100", "db100_wm", "db100(菱形橋)"],
      ["has_sl100", "sl100_wm", "sl100(挖空槽)"]]) {
      if (!p[flag]) continue;
      const v = p[key];
      const delta = (v !== null && p.wm !== null) ? `Δ ${(v - p.wm) >= 0 ? "+" : ""}${(v - p.wm).toFixed(2)}` : "";
      cards.append(el("div", { class: "card" },
        el("div", { class: "v", text: fmt(v) }),
        el("div", { class: "k", text: name }),
        el("div", { class: "d muted", text: delta })));
    }
    vp.append(cards);
  }
  $("#detail-to-compare").onclick = () => addTray(p.variant_of || p.id);
  $("#detail-to-group").onclick = () => addToActiveGroup(p.variant_of || p.id);
  $("#neighbors-grid").textContent = "";
  $("#nb-go").onclick = () => findNeighbors(p.variant_of || p.id);
  renderDetailCurves(p).catch((e) => {
    $("#det-curve-note").textContent = `曲線載入失敗:${e.message}`;
  });
}
async function renderDetailCurves(p) {
  // 原始/菱形(~db100)/挖空(~sl100)曲線疊圖(有才畫);目標線與內帶=/api/targets
  const t = await getTargets();
  const base = p.variant_of || p.id;
  const wanted = [{ id: base, label: "原始" }];
  if (p.has_db100) wanted.push({ id: `${base}~db100`, label: "菱形 db100" });
  if (p.has_sl100) wanted.push({ id: `${base}~sl100`, label: "挖空 sl100" });
  const ids = wanted.map((w) => w.id);
  const [respMap, radMap] = await Promise.all([getCurves("resp", ids), getCurves("radc", ids)]);
  const rs = wanted.map((w, i) => ({ ...w, color: SERIES[i % SERIES.length],
    resp: respMap.get(w.id), rad: radMap.get(w.id) }));
  // S11 / Gain
  const respSeries = rs.filter((s) => s.resp);
  drawLineChart($("#det-s11"), {
    xs: t.freqs,
    series: respSeries.map((s) => ({ label: s.label, color: s.color, ys: s.resp.s11 })),
    band: t.band, hline: t.s11_max, hlineLabel: `目標 ≤ ${t.s11_max}dB`,
    yLabel: "S11(dB)", xLabel: "GHz",
  });
  drawLineChart($("#det-gain"), {
    xs: t.freqs,
    series: respSeries.map((s) => ({ label: s.label, color: s.color, ys: s.resp.gain })),
    band: t.band, hline: t.gain_min, hlineLabel: `目標 ≥ ${t.gain_min}dB`,
    yLabel: "Gain(dB)", xLabel: "GHz",
  });
  legendInto($("#det-curve-legend"), respSeries);
  const noResp = rs.filter((s) => !s.resp).map((s) => s.label);
  $("#det-curve-note").textContent = respSeries.length
    ? (noResp.length ? `無曲線資料:${noResp.join("、")}` : "")
    : "此 pattern 尚無 S11/Gain 曲線資料(build_index 增量刷新後出現)。";
  // rad 極座標
  const radSeries = rs.filter((s) => s.rad);
  const theta = radSeries.length ? radSeries[0].rad.theta : null;
  const mk = (key) => radSeries.filter((s) => s.rad[key])
    .map((s) => ({ label: s.label, color: s.color, vals: s.rad[key] }));
  drawPolar($("#det-rad0"), theta ? { theta, series: mk("phi0"), windowDeg: t.rad_window } : null);
  drawPolar($("#det-rad90"), theta ? { theta, series: mk("phi90"), windowDeg: t.rad_window } : null);
  legendInto($("#det-rad-legend"), radSeries);
  const noRad = rs.filter((s) => !s.rad).map((s) => s.label);
  $("#det-rad-note").textContent = radSeries.length
    ? (noRad.length ? `無 rad 曲線:${noRad.join("、")}` : "")
    : "此 pattern 尚無 rad 場型資料。";
}
async function findNeighbors(id) {
  const maxd = parseInt($("#nb-maxd").value, 10) || 100;
  const limit = parseInt($("#nb-limit").value, 10) || 24;
  let rows;
  try { rows = await api(`/api/hamming?id=${encodeURIComponent(id)}&maxd=${maxd}&limit=${limit}`); }
  catch (e) { toast(`找鄰居失敗:${e.message}`, true); return; }
  const grid = $("#neighbors-grid");
  grid.textContent = "";
  if (!rows.length) { grid.append(el("div", { class: "muted", text: `maxd=${maxd} 內沒有鄰居` })); return; }
  for (const r of rows) {
    grid.append(el("div", { class: "thumb-card", onclick: () => nav("detail", r.id) },
      makeThumb(b64ToBits(r.bits_b64), "thumb md"),
      el("div", { class: "tid", text: r.id }),
      el("div", { class: "muted", text: `d=${r.d} · wm ${fmt(r.wm)}` })));
  }
}

/* ---------------- 比對 ---------------- */
function addTray(id) {
  if (S.tray.includes(id)) { toast(`${id} 已在比對清單`); return; }
  if (S.tray.length >= 4) { toast("比對最多 4 筆,先移除一筆", true); return; }
  S.tray.push(id);
  saveLS("pb_tray_v1", S.tray);
  updateTrayBadge();
  toast(`已加入比對(${S.tray.length}/4):${id}`);
  if (S.view === "compare") renderCompare();
}
function removeTray(id) {
  S.tray = S.tray.filter((x) => x !== id);
  saveLS("pb_tray_v1", S.tray);
  updateTrayBadge();
  renderCompare();
}
function updateTrayBadge() { $("#tray-badge").textContent = String(S.tray.length); }
function setCmpTab(tab) {
  S.cmpTab = tab;
  document.querySelectorAll("#cmp-tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  $("#cmp-pattern").hidden = tab !== "pattern";
  $("#cmp-s11").hidden = tab !== "s11";
  $("#cmp-gain").hidden = tab !== "gain";
  $("#cmp-rad").hidden = tab !== "rad";
  if (S.view === "compare") renderCompare();
}
async function renderCompare() {
  const ed = $("#tray-editor");
  ed.textContent = "";
  for (const id of S.tray) {
    ed.append(el("span", { class: "chip" }, el("b", { text: id }),
      el("button", { text: "✕", "data-tip": "從比對清單移除", onclick: () => removeTray(id) })));
  }
  const input = el("input", { type: "text", placeholder: "輸入 id 加入…" });
  const addBtn = el("button", {
    text: "加入", "data-tip": "輸入完整 id(含變體如 xxx~db100 也可)加入比對",
    onclick: async () => {
      const id = input.value.trim();
      if (!id) return;
      try { await getPattern(id); addTray(id); input.value = ""; }
      catch (e) { toast(`加入失敗:${e.message}`, true); }
    },
  });
  input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); addBtn.click(); } });
  ed.append(input, addBtn);
  if (S.tray.length) ed.append(el("button", { text: "清空", "data-tip": "清空比對清單",
    onclick: () => { S.tray = []; saveLS("pb_tray_v1", S.tray); updateTrayBadge(); renderCompare(); } }));

  const hint = $("#compare-hint");
  if (S.tray.length < 2) {
    hint.textContent = "至少選 2 筆(總攬/詳情按「比對＋」,或上方輸入 id)。";
    $("#compare-grid").textContent = ""; $("#xor-panel").textContent = "";
    for (const id of ["#cmp-s11-cv", "#cmp-gain-cv", "#cmp-rad0", "#cmp-rad90"]) {
      const cv = $(id); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      cv._opts = null; cv._popts = null;
    }
    for (const id of ["#cmp-s11-legend", "#cmp-gain-legend", "#cmp-rad-legend",
      "#cmp-s11-note", "#cmp-gain-note", "#cmp-rad-note"]) $(id).textContent = "";
    return;
  }
  hint.textContent = "";
  try {
    if (S.cmpTab === "pattern") await renderComparePattern();
    else if (S.cmpTab === "rad") await renderCompareRad();
    else await renderCompareResp(S.cmpTab);
  } catch (e) { toast(`比對載入失敗:${e.message}`, true); }
}
async function renderComparePattern() {
  const grid = $("#compare-grid"), xor = $("#xor-panel");
  grid.textContent = ""; xor.textContent = "";
  const items = await api(`/api/compare?ids=${S.tray.map(encodeURIComponent).join(",")}`);
  const bitsMap = new Map(items.map((it) => [it.id, Uint8Array.from(it.bits)]));
  for (const it of items) {
    const cv = el("canvas", { width: 250, height: 250 });
    drawBig(cv, bitsMap.get(it.id));
    const tbl = el("table", {});
    for (const [k, v] of [["wm", fmt(it.wm)], ["rad", fmt(it.rad)], ["lo", fmt(it.lo)],
      ["ndiag", it.ndiag], ["n8", it.n8], ["total", it.total]]) {
      tbl.append(el("tr", {}, el("td", { class: "muted", text: k }), el("td", { text: String(v) })));
    }
    grid.append(el("div", { class: "cmp-col" }, cv,
      el("div", { class: "cid", text: it.id, onclick: () => nav("detail", it.id) }), tbl));
  }
  const ids = items.map((it) => it.id);
  let baseId = xor._baseId && ids.includes(xor._baseId) ? xor._baseId : ids[0];
  const sel = el("select", {
    "data-tip": "XOR 的基準:其他每筆都跟它比差異",
    onchange: () => { xor._baseId = sel.value; renderCompare(); },
  });
  for (const id of ids) sel.append(el("option", { value: id, text: id, ...(id === baseId ? { selected: "" } : {}) }));
  xor._baseId = baseId;
  xor.append(el("h3", { text: "XOR 差異" }),
    el("div", { class: "xor-legend" }, "基準 A:", sel,
      el("span", { class: "sw", style: `background:${COLORS.both}` }), "共同",
      el("span", { class: "sw", style: `background:${COLORS.aOnly}` }), "A 獨有",
      el("span", { class: "sw", style: `background:${COLORS.bOnly}` }), "B 獨有"));
  const xg = el("div", { class: "xor-grid" });
  const aBits = bitsMap.get(baseId);
  for (const id of ids) {
    if (id === baseId) continue;
    const cv = el("canvas", { width: 250, height: 250 });
    drawXor(cv, aBits, bitsMap.get(id));
    xg.append(el("div", { class: "xor-card" }, cv,
      el("div", { text: `B=${id} · d=${hamming(aBits, bitsMap.get(id))}` })));
  }
  xor.append(xg);
  if (ids.length > 2) {
    const mt = el("table", { id: "pair-matrix" });
    mt.append(el("tr", {}, el("th", { text: "d" }), ...ids.map((id) => el("th", { text: id }))));
    for (const a of ids) {
      mt.append(el("tr", {}, el("th", { text: a }),
        ...ids.map((b) => el("td", { class: "num", text: a === b ? "·" : String(hamming(bitsMap.get(a), bitsMap.get(b))) }))));
    }
    xor.append(el("h3", { text: "兩兩 Hamming 距離" }), el("div", { class: "table-wrap" }, mt));
  }
}
async function renderCompareResp(tab) {
  // tab='s11'|'gain':疊圖+目標線+內帶底色
  const t = await getTargets();
  const map = await getCurves("resp", S.tray);
  const series = [], missing = [];
  S.tray.forEach((id, i) => {
    const c = map.get(id);
    if (c) series.push({ label: id, color: SERIES[i % SERIES.length], ys: tab === "s11" ? c.s11 : c.gain });
    else missing.push(id);
  });
  const cv = $(tab === "s11" ? "#cmp-s11-cv" : "#cmp-gain-cv");
  drawLineChart(cv, {
    xs: t.freqs, series,
    band: t.band,
    hline: tab === "s11" ? t.s11_max : t.gain_min,
    hlineLabel: tab === "s11" ? `目標 ≤ ${t.s11_max}dB` : `目標 ≥ ${t.gain_min}dB`,
    yLabel: tab === "s11" ? "S11(dB)" : "Gain(dB)", xLabel: "GHz",
  });
  legendInto($(tab === "s11" ? "#cmp-s11-legend" : "#cmp-gain-legend"), series,
    (s) => nav("detail", s.label));
  $(tab === "s11" ? "#cmp-s11-note" : "#cmp-gain-note").textContent =
    missing.length ? `無曲線資料:${missing.join("、")}` : "";
}
async function renderCompareRad() {
  const t = await getTargets();
  const map = await getCurves("radc", S.tray);
  const have = [], missing = [];
  S.tray.forEach((id, i) => {
    const c = map.get(id);
    if (c) have.push({ id, color: SERIES[i % SERIES.length], rad: c });
    else missing.push(id);
  });
  const theta = have.length ? have[0].rad.theta : null;
  const mk = (key) => have.filter((s) => s.rad[key])
    .map((s) => ({ label: s.id, color: s.color, vals: s.rad[key] }));
  drawPolar($("#cmp-rad0"), theta ? { theta, series: mk("phi0"), windowDeg: t.rad_window } : null);
  drawPolar($("#cmp-rad90"), theta ? { theta, series: mk("phi90"), windowDeg: t.rad_window } : null);
  legendInto($("#cmp-rad-legend"), have.map((s) => ({ label: s.id, color: s.color })),
    (s) => nav("detail", s.label));
  $("#cmp-rad-note").textContent = missing.length ? `無 rad 曲線:${missing.join("、")}` : "";
}

/* ---------------- 群組 ---------------- */
function findGroup(name) { return S.groups.groups.find((g) => g.name === name); }
function saveGroups() { saveLS("pb_groups_v1", S.groups); saveLS("pb_active_group", S.activeGroup); }
function addToActiveGroup(id) { addIdsToGroup([id]); }
function addIdsToGroup(ids) {
  let g = S.activeGroup && findGroup(S.activeGroup);
  if (!g) {
    const name = prompt("尚無使用中群組,輸入新群組名稱:", "我的群組");
    if (!name) return;
    g = findGroup(name) || { name, ids: [] };
    if (!findGroup(name)) S.groups.groups.push(g);
    S.activeGroup = name;
  }
  let added = 0;
  for (const id of ids) if (!g.ids.includes(id)) { g.ids.push(id); added++; }
  saveGroups();
  toast(added ? `已加入群組「${g.name}」+${added} 筆(共 ${g.ids.length})` : `都已在群組「${g.name}」`);
  if (S.view === "groups") renderGroups();
}
function renderGroups() {
  const ul = $("#group-list");
  ul.textContent = "";
  if (!S.groups.groups.length) ul.append(el("li", { class: "muted", text: "尚無群組" }));
  for (const g of S.groups.groups) {
    ul.append(el("li", { class: g.name === S.activeGroup ? "active" : "", onclick: () => { S.activeGroup = g.name; saveGroups(); renderGroups(); } },
      el("span", { class: "gname", text: `${g.name}(${g.ids.length})` }),
      el("button", {
        text: "✎", "data-tip": "改名", onclick: (ev) => {
          ev.stopPropagation();
          const name = prompt("新名稱:", g.name);
          if (!name || name === g.name) return;
          if (findGroup(name)) { toast("名稱已存在", true); return; }
          if (S.activeGroup === g.name) S.activeGroup = name;
          g.name = name;
          saveGroups(); renderGroups();
        },
      }),
      el("button", {
        text: "✕", "data-tip": "刪除群組(不影響資料本體)", onclick: (ev) => {
          ev.stopPropagation();
          if (!confirm(`刪除群組「${g.name}」?(不影響資料本體)`)) return;
          S.groups.groups = S.groups.groups.filter((x) => x !== g);
          if (S.activeGroup === g.name) S.activeGroup = null;
          saveGroups(); renderGroups();
        },
      })));
  }
  $("#grp-new").onclick = () => {
    const name = $("#grp-new-name").value.trim();
    if (!name) { toast("先輸入群組名稱", true); return; }
    if (findGroup(name)) { toast("名稱已存在", true); return; }
    S.groups.groups.push({ name, ids: [] });
    S.activeGroup = name;
    $("#grp-new-name").value = "";
    saveGroups(); renderGroups();
  };
  $("#grp-export").onclick = () => {
    const blob = new Blob([JSON.stringify(S.groups, null, 2)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "pattern_groups.json" });
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $("#grp-import").onclick = () => $("#grp-import-file").click();
  $("#grp-import-file").onchange = async (ev) => {
    const f = ev.target.files[0];
    ev.target.value = "";
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text());
      if (!parsed || !Array.isArray(parsed.groups)) throw new Error("格式需為 {groups:[{name,ids}]}");
      let added = 0;
      for (const g of parsed.groups) {
        if (typeof g.name !== "string" || !Array.isArray(g.ids)) continue;
        const mine = findGroup(g.name);
        if (mine) { for (const id of g.ids) if (!mine.ids.includes(id)) { mine.ids.push(id); added++; } }
        else { S.groups.groups.push({ name: g.name, ids: g.ids.filter((x) => typeof x === "string") }); added += g.ids.length; }
      }
      saveGroups(); renderGroups();
      toast(`匯入完成,合併 ${added} 筆成員`);
    } catch (e) { toast(`匯入失敗:${e.message}`, true); }
  };
  renderGroupDetail();
}
async function renderGroupDetail() {
  const box = $("#group-detail");
  const g = S.activeGroup && findGroup(S.activeGroup);
  if (!g) { box.textContent = ""; box.append(el("div", { class: "muted", text: "左側選一個群組,或在總攬/詳情按「群組＋」。" })); return; }
  box.textContent = "";
  box.append(el("h3", { text: `群組「${g.name}」— ${g.ids.length} 筆` }));
  const btns = el("div", { class: "btn-row" },
    el("button", { text: "勾選→送比對", "data-tip": "把勾選的 2–4 筆成員送進比對頁(沒勾=取前 4 筆)", onclick: () => {
      const checked = [...box.querySelectorAll(".member-card input:checked")].map((c) => c.dataset.id);
      const ids = checked.length ? checked : g.ids.slice(0, 4);
      if (ids.length < 2 || ids.length > 4) { toast("比對需 2–4 筆(勾選 2–4 筆成員)", true); return; }
      S.tray = ids;
      saveLS("pb_tray_v1", S.tray);
      updateTrayBadge();
      nav("compare");
    } }),
    el("button", { text: "移除勾選", "data-tip": "把勾選的成員從這個群組移除(不動資料本體)", onclick: () => {
      const checked = new Set([...box.querySelectorAll(".member-card input:checked")].map((c) => c.dataset.id));
      if (!checked.size) { toast("沒有勾選任何成員", true); return; }
      g.ids = g.ids.filter((id) => !checked.has(id));
      saveGroups(); renderGroups();
    } }));
  box.append(btns);
  const wall = el("div", { class: "thumb-grid" });
  box.append(wall);
  const members = [];
  for (const id of g.ids) {
    try {
      const p = await getPattern(id);
      members.push(p);
      wall.append(el("div", { class: "thumb-card member-card" },
        el("input", { type: "checkbox", "data-id": id, onclick: (ev) => ev.stopPropagation() }),
        el("span", { onclick: () => nav("detail", id) },
          makeThumb(b64ToBits(p.bits_b64), "thumb md"),
          el("div", { class: "tid", text: id }),
          el("div", { class: "muted", text: `wm ${fmt(p.wm)}` }))));
    } catch (e) {
      wall.append(el("div", { class: "thumb-card member-card" },
        el("input", { type: "checkbox", "data-id": id, onclick: (ev) => ev.stopPropagation() }),
        el("div", { class: "tid", text: id }),
        el("div", { class: "muted", text: "載入失敗" })));
    }
  }
  if (members.length) {
    const tbl = el("table", {});
    tbl.append(el("tr", {}, ...["id", "wm", "rad", "lo", "ndiag", "n8", "total", "kind", "store"]
      .map((h) => el("th", { text: h }))));
    for (const p of members) {
      tbl.append(el("tr", { onclick: () => nav("detail", p.id) },
        el("td", { text: p.id }),
        el("td", { class: "num" + (p.wm !== null && p.wm >= 0 ? " wm-pos" : ""), text: fmt(p.wm) }),
        el("td", { class: "num", text: fmt(p.rad) }),
        el("td", { class: "num", text: fmt(p.lo) }),
        el("td", { class: "num", text: String(p.ndiag) }),
        el("td", { class: "num", text: String(p.n8) }),
        el("td", { class: "num", text: String(p.total) }),
        el("td", { text: p.kind || "—" }),
        el("td", { text: p.store || "—" })));
    }
    box.append(el("div", { class: "table-wrap grp-table-wrap" }, tbl));
  }
}

/* ---------------- 製造視角(排行榜+規格達成卡+匯出) ---------------- */
async function renderMfg() {
  await getTargets();
  const axis = S.mfg.axis;
  const dir = (axis === "lo" || axis === "sel") ? "asc" : "desc";
  const p = new URLSearchParams({ sort: axis, dir, limit: "200", offset: "0" });
  if ($("#mfg-gates").checked) p.set("f_qual", "1");
  let data;
  try { data = await api(`/api/list?${p.toString()}`); }
  catch (e) { toast(`載入排行榜失敗:${e.message}`, true); return; }
  S.mfg.rows = data.rows;
  $("#mfg-info").textContent =
    `閘門${$("#mfg-gates").checked ? "開(wm≥0.15∧rad≥0)" : "關"} · 依 ${axis} 排序 · 符合 ${data.total} 筆(取前 ${data.rows.length})`;
  document.querySelectorAll("#mfg-table th.mfg-sort").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.axis === axis);
    th.textContent = th.dataset.axis + (th.dataset.axis === axis ? " ↓" : "");
  });
  const sel = S.mfg.rows.find((r) => r.id === S.mfg.selId) || S.mfg.rows[0] || null;
  S.mfg.selId = sel ? sel.id : null;
  renderSpecCard(sel);
  const tb = $("#mfg-table tbody");
  tb.textContent = "";
  S.mfg.rows.forEach((r, i) => {
    const tr = el("tr", { class: r.id === S.mfg.selId ? "selrow" : "",
      onclick: () => { S.mfg.selId = r.id; renderSpecCard(r);
        tb.querySelectorAll("tr").forEach((x) => x.classList.remove("selrow")); tr.classList.add("selrow"); } },
      el("td", { class: "num", text: String(i + 1) }),
      el("td", {}, makeThumb(b64ToBits(r.bits_b64))),
      el("td", {}, el("span", { class: "cid", text: r.id,
        onclick: (ev) => { ev.stopPropagation(); nav("detail", r.id); } })),
      el("td", { class: "num" + (r.wm !== null && r.wm >= 0.15 ? " wm-pos" : ""), text: fmt(r.wm) }),
      el("td", { class: "num", text: fmt(r.rad) }),
      el("td", { class: "num", text: fmt(r.lo) }),
      el("td", { class: "num", text: fmt(r.sel) }),
      el("td", { class: "num" + (r.db100_wm !== null && r.db100_wm >= 0.15 ? " wm-pos" : ""), text: fmt(r.db100_wm) }),
      el("td", { class: "num", text: fmt(r.sl100_wm) }),
      el("td", { class: "num", text: String(r.ndiag) }),
      el("td", { text: r.kind || "—" }),
      el("td", { text: r.store || "—" }));
    tb.append(tr);
  });
}
function renderSpecCard(r) {
  const box = $("#mfg-speccard");
  box.textContent = "";
  if (!r) { box.append(el("div", { class: "muted", text: "排行榜無資料(檢查閘門/資料)" })); return; }
  const t = S.targets;
  box.append(el("h3", { text: `規格達成 — ${r.id}(點排行榜任一列切換)` }));
  const cards = el("div", { class: "cards" });
  const items = [
    { k: `wm 對合格線 ${t.wm_buffer}`, v: r.wm, d: r.wm == null ? null : r.wm - t.wm_buffer,
      tip: "雙帶最差餘裕與合格 buffer 的差:正=過門檻" },
    { k: "rad 對窗地板 0", v: r.rad, d: r.rad == null ? null : r.rad,
      tip: "±45° 窗地板餘裕:正=窗內都在 G0−3dB 之上" },
    { k: `db100_wm 對 ${t.wm_buffer}(可製造)`, v: r.db100_wm, d: r.db100_wm == null ? null : r.db100_wm - t.wm_buffer,
      tip: "菱形化後重模擬的 wm=實際出貨值;正=做出來也合格" },
    { k: "lo(左帶外上限,越負越好)", v: r.lo, d: null, tip: "左側帶外增益上限;沒有硬門檻,拿來排序比較" },
  ];
  for (const it of items) {
    const dtxt = it.d == null ? "—" : `${it.d >= 0 ? "達標 +" : "差 "}${Math.abs(it.d).toFixed(2)} dB`;
    cards.append(el("div", { class: "card", "data-tip": it.tip },
      el("div", { class: "v", text: fmt(it.v) }),
      el("div", { class: "k", text: it.k }),
      el("div", { class: "d " + (it.d == null ? "muted" : (it.d >= 0 ? "delta-good" : "delta-bad")), text: dtxt })));
  }
  box.append(cards);
}
function csvEsc(v) {
  const s = String(v == null ? "" : v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function mfgTopN() {
  const n = parseInt($("#mfg-topn").value, 10) || 20;
  return S.mfg.rows.slice(0, Math.max(1, Math.min(n, S.mfg.rows.length)));
}
function downloadBlob(blob, name) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  a.click();
  URL.revokeObjectURL(a.href);
}
function mfgExportCsv() {
  const rows = mfgTopN();
  if (!rows.length) { toast("排行榜無資料", true); return; }
  const cols = ["rank", "id", "wm", "rad", "lo", "sel", "db100_wm", "sl100_wm",
    "ndiag", "n8", "total", "largest8_frac", "kind", "store", "folder"];
  const lines = [cols.join(",")];
  rows.forEach((r, i) => {
    lines.push([i + 1, csvEsc(r.id), r.wm, r.rad, r.lo, r.sel, r.db100_wm, r.sl100_wm,
      r.ndiag, r.n8, r.total, r.largest8_frac, csvEsc(r.kind), csvEsc(r.store), csvEsc(r.folder)]
      .map((v) => v == null ? "" : v).join(","));
  });
  downloadBlob(new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" }),
    `pattern_top${rows.length}_${S.mfg.axis}.csv`);
  toast(`已匯出 top-${rows.length} CSV(依 ${S.mfg.axis})`);
}
function mfgExportPng() {
  const rows = mfgTopN();
  if (!rows.length) { toast("排行榜無資料", true); return; }
  const cell = 96, capH = 26, pad = 12, cols = Math.min(5, rows.length);
  const nrow = Math.ceil(rows.length / cols);
  const cv = el("canvas", { width: cols * (cell + pad) + pad, height: nrow * (cell + capH + pad) + pad + 20 });
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#fcfcfb";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = COLORS.ink2; ctx.font = "11px system-ui";
  ctx.fillText(`Pattern top-${rows.length}(依 ${S.mfg.axis};圖下緣=饋線邊)`, pad, 14);
  ctx.imageSmoothingEnabled = false;
  rows.forEach((r, i) => {
    const cx = pad + (i % cols) * (cell + pad);
    const cy = 20 + pad + Math.floor(i / cols) * (cell + capH + pad);
    ctx.drawImage(makeThumb(b64ToBits(r.bits_b64)), cx, cy, cell, cell);
    ctx.strokeStyle = COLORS.gridline;
    ctx.strokeRect(cx + 0.5, cy + 0.5, cell, cell);
    ctx.fillStyle = COLORS.ink;
    ctx.fillText(`#${i + 1} ${r.id}`.slice(0, 18), cx, cy + cell + 11);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`wm ${fmt(r.wm)} db100 ${fmt(r.db100_wm)}`, cx, cy + cell + 22);
  });
  cv.toBlob((b) => downloadBlob(b, `pattern_top${rows.length}_${S.mfg.axis}.png`));
  toast(`已匯出 top-${rows.length} PNG`);
}

/* ---------------- 研究視角(帕累托+家族+覆蓋) ---------------- */
function isQual(r) { return r.wm != null && r.rad != null && r.wm >= 0.15 && r.rad >= 0; }
async function renderResearch() {
  let stats, rows;
  try {
    await getTargets();
    [stats, rows] = await Promise.all([
      api("/api/stats"),
      api("/api/list?lite=1&limit=50000&offset=0").then((d) => d.rows),
    ]);
  } catch (e) { toast(`載入研究視角失敗:${e.message}`, true); return; }
  const pts = rows.filter((r) => r.wm != null && r.lo != null)
    .map((r) => ({ id: r.id, x: +r.lo, y: +r.wm, rad: r.rad, qual: isQual(r) }));
  // 帕累托前緣:lo 越小越好、wm 越大越好
  const sorted = [...pts].sort((a, b) => a.x - b.x || b.y - a.y);
  const front = [];
  let best = -Infinity;
  for (const p of sorted) if (p.y > best) { front.push(p); best = p.y; }
  S.pareto.pts = pts;
  S.pareto.front = front;
  $("#pareto-info").textContent =
    `${pts.length} 點(wm、lo 皆有值)· 合格 ${pts.filter((p) => p.qual).length} · 前緣 ${front.length} 點`;
  paretoRedraw();
  renderFamilies(rows);
  renderCoverage(stats, rows);
}
function paretoRedraw(hoverI = -1) {
  const cv = $("#pareto-canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pad = { l: 54, r: 16, t: 14, b: 36 };
  ctx.clearRect(0, 0, W, H);
  const pts = S.pareto.pts || [];
  if (!pts.length) { drawEmptyChart(cv, "沒有 wm、lo 皆有值的點"); return; }
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of pts) {
    xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x);
    ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y);
  }
  const px = Math.max((xmax - xmin) * 0.05, 0.3), py = Math.max((ymax - ymin) * 0.05, 0.3);
  xmin -= px; xmax += px; ymin -= py; ymax += py;
  const X = (x) => pad.l + (x - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const Y = (y) => pad.t + (ymax - y) / (ymax - ymin) * (H - pad.t - pad.b);
  ctx.font = "10px system-ui";
  for (const tv of niceTicks(ymin, ymax, 6)) {
    ctx.strokeStyle = COLORS.gridline;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(tv) + 0.5); ctx.lineTo(W - pad.r, Y(tv) + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "right"; ctx.fillText(String(tv), pad.l - 5, Y(tv) + 3);
  }
  for (const tv of niceTicks(xmin, xmax, 8)) {
    ctx.strokeStyle = COLORS.gridline;
    ctx.beginPath(); ctx.moveTo(X(tv) + 0.5, pad.t); ctx.lineTo(X(tv) + 0.5, H - pad.b); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "center"; ctx.fillText(String(tv), X(tv), H - pad.b + 13);
  }
  // 合格線 wm=0.15(紅虛線)
  if (S.targets) {
    const yq = Y(S.targets.wm_buffer);
    if (yq > pad.t && yq < H - pad.b) {
      ctx.strokeStyle = COLORS.target; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, yq); ctx.lineTo(W - pad.r, yq); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.target; ctx.textAlign = "left";
      ctx.fillText(`wm=${S.targets.wm_buffer}(合格線)`, pad.l + 4, yq - 4);
    }
  }
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b + 0.5); ctx.lineTo(W - pad.r, H - pad.b + 0.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l + 0.5, pad.t); ctx.lineTo(pad.l + 0.5, H - pad.b); ctx.stroke();
  ctx.fillStyle = COLORS.ink2;
  ctx.textAlign = "right"; ctx.fillText("lo(dB,越左越好)", W - 4, H - 4);
  ctx.textAlign = "left"; ctx.fillText("wm(dB,越上越好)", 4, 11);
  for (const p of pts) {
    p.px = X(p.x); p.py = Y(p.y);
    ctx.fillStyle = p.qual ? COLORS.ptBase : COLORS.ptDim;
    ctx.beginPath(); ctx.arc(p.px, p.py, p.qual ? 3.5 : 2.5, 0, Math.PI * 2); ctx.fill();
  }
  // 前緣(橘,階梯線+點)
  const front = S.pareto.front || [];
  if (front.length) {
    ctx.strokeStyle = COLORS.sel; ctx.lineWidth = 2;
    ctx.beginPath();
    front.forEach((p, i) => { if (i) ctx.lineTo(p.px, p.py); else ctx.moveTo(p.px, p.py); });
    ctx.stroke();
    for (const p of front) {
      ctx.fillStyle = COLORS.sel;
      ctx.beginPath(); ctx.arc(p.px, p.py, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  if (hoverI >= 0) {
    const p = pts[hoverI];
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p.px, p.py, 6, 0, Math.PI * 2); ctx.stroke();
  }
}
function initParetoEvents() {
  const cv = $("#pareto-canvas");
  const hit = (ev) => {
    const r = cv.getBoundingClientRect();
    const mx = (ev.clientX - r.left) * (cv.width / r.width);
    const my = (ev.clientY - r.top) * (cv.height / r.height);
    let best = -1, bestD = 100;
    (S.pareto.pts || []).forEach((p, i) => {
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  };
  cv.addEventListener("mousemove", (ev) => {
    const i = hit(ev);
    paretoRedraw(i);
    if (i >= 0) {
      const p = S.pareto.pts[i];
      const onFront = (S.pareto.front || []).includes(p);
      showTip(ev.pageX + 14, ev.pageY - 10,
        `${p.id}${onFront ? "(前緣)" : ""}${p.qual ? "(合格)" : ""}\nlo ${p.x} · wm ${p.y} · rad ${fmt(p.rad)}\n點一下進詳情`);
    } else hideTip();
  });
  cv.addEventListener("mouseleave", () => { paretoRedraw(); hideTip(); });
  cv.addEventListener("click", (ev) => {
    const i = hit(ev);
    if (i >= 0) nav("detail", S.pareto.pts[i].id);
  });
}
function familyKey(id) {
  const cut = id.indexOf("_");
  return cut > 0 ? id.slice(0, cut) : id;
}
function renderFamilies(rows) {
  const fam = new Map();
  for (const r of rows) {
    const k = familyKey(r.id);
    let f = fam.get(k);
    if (!f) { f = { key: k, n: 0, qual: 0, bestWm: null, bestLo: null, db: 0 }; fam.set(k, f); }
    f.n++;
    if (isQual(r)) f.qual++;
    if (r.wm != null && (f.bestWm == null || r.wm > f.bestWm)) f.bestWm = r.wm;
    if (r.lo != null && (f.bestLo == null || r.lo < f.bestLo)) f.bestLo = r.lo;
    if (r.has_db100) f.db++;
  }
  const top = [...fam.values()].sort((a, b) => b.n - a.n).slice(0, 20);
  const tb = $("#family-table tbody");
  tb.textContent = "";
  for (const f of top) {
    tb.append(el("tr", { onclick: () => {
      $("#f-q").value = f.key;
      S.list.offset = 0;
      nav("overview");
      if (S.view === "overview") setOvMode(S.ovMode);
    }, "data-tip": `點一下 → 總攬用「${f.key}」篩選` },
      el("td", { text: f.key }),
      el("td", { class: "num", text: String(f.n) }),
      el("td", { class: "num" + (f.qual ? " wm-pos" : ""), text: String(f.qual) }),
      el("td", { class: "num", text: fmt(f.bestWm) }),
      el("td", { class: "num", text: fmt(f.bestLo) }),
      el("td", { class: "num", text: String(f.db) })));
  }
}
function renderCoverage(stats, rows) {
  const box = $("#abl-cards");
  box.textContent = "";
  const qualRows = rows.filter(isQual);
  const qualDb = qualRows.filter((r) => r.has_db100).length;
  const pct = (a, b) => b ? `${(100 * a / b).toFixed(1)}%` : "—";
  const cards = [
    ["總筆數", stats.total, ""],
    ["db100 消融", stats.db100_count, pct(stats.db100_count, stats.total)],
    ["sl100 消融", stats.sl100_count, pct(stats.sl100_count, stats.total)],
    ["合格且有 db100", qualDb, pct(qualDb, qualRows.length) + " 的合格解"],
    ["S11/Gain 曲線", stats.resp_count == null ? "—" : stats.resp_count, pct(stats.resp_count || 0, stats.total)],
    ["rad 曲線", stats.rad_count == null ? "—" : stats.rad_count, pct(stats.rad_count || 0, stats.total)],
  ];
  for (const [k, v, d] of cards) {
    box.append(el("div", { class: "card" },
      el("div", { class: "v", text: String(v) }),
      el("div", { class: "k", text: k }),
      el("div", { class: "d muted", text: d })));
  }
}

/* ---------------- 說明頁(文件站:TOC+scroll-spy;內容在 help.js) ---------------- */
function renderHelp(anchor) {
  const box = $("#help-content");
  if (!box.dataset.ready) {
    box.dataset.ready = "1";
    box.innerHTML = window.HELP_HTML || "<p class='muted'>help.js 未載入(重新整理試試)</p>";
    buildHelpToc();
  }
  if (anchor) {
    const t = document.getElementById(anchor);
    if (t) setTimeout(() => { t.scrollIntoView({ behavior: "smooth", block: "start" }); setTocActive(anchor); }, 30);
  }
}
function buildHelpToc() {
  const toc = $("#help-toc");
  toc.textContent = "";
  toc.append(el("div", { class: "toc-title", text: "目錄" }));
  for (const h of $("#help-content").querySelectorAll("h2[id], h3[id]")) {
    // href 只為可書籤;click 用 replaceState 更新網址(不觸發 hashchange=不打擾視圖路由)
    toc.append(el("a", {
      class: "toc-" + h.tagName.toLowerCase(), href: `#help/${h.id}`, "data-target": h.id,
      text: h.textContent,
      onclick: (ev) => {
        ev.preventDefault();
        history.replaceState(null, "", `#help/${h.id}`);
        h.scrollIntoView({ behavior: "smooth", block: "start" });
        setTocActive(h.id);
      },
    }));
  }
  window.addEventListener("scroll", helpScrollSpy, { passive: true });
}
function setTocActive(id) {
  document.querySelectorAll("#help-toc a").forEach((a) => {
    a.classList.toggle("active", a.dataset.target === id);
  });
}
let spyPending = false;
function helpScrollSpy() {
  if (S.view !== "help" || spyPending) return;
  spyPending = true;
  requestAnimationFrame(() => {
    spyPending = false;
    let cur = null;
    for (const h of $("#help-content").querySelectorAll("h2[id], h3[id]")) {
      if (h.getBoundingClientRect().top <= 96) cur = h; else break;
    }
    if (cur) setTocActive(cur.id);
  });
}

/* ---------------- 初始化 ---------------- */
function init() {
  document.querySelectorAll("#nav button").forEach((b) => {
    b.addEventListener("click", () => nav(b.dataset.view));
  });
  $("#filters").addEventListener("submit", (ev) => {
    ev.preventDefault();
    S.list.offset = 0;
    S.list.limit = parseInt($("#limit").value, 10) || 50;
    if (S.ovMode === "scatter") renderScatter(); else fetchList();
  });
  $("#filters-reset").addEventListener("click", () => {
    $("#filters").reset();
    S.list = { offset: 0, limit: 50, sort: "wm", dir: "desc", total: 0 };
    if (S.ovMode === "scatter") renderScatter(); else fetchList();
  });
  document.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (S.list.sort === col) S.list.dir = S.list.dir === "asc" ? "desc" : "asc";
      else { S.list.sort = col; S.list.dir = "desc"; }
      S.list.offset = 0;
      fetchList();
    });
  });
  $("#pg-prev").addEventListener("click", () => {
    S.list.offset = Math.max(0, S.list.offset - S.list.limit);
    fetchList();
  });
  $("#pg-next").addEventListener("click", () => {
    S.list.offset += S.list.limit;
    fetchList();
  });
  // v2:總攬渲染模式
  document.querySelectorAll("#ov-mode button").forEach((b) => {
    b.addEventListener("click", () => setOvMode(b.dataset.mode));
  });
  for (const sid of ["#sc-x", "#sc-y"]) {
    const sel = $(sid);
    for (const [k, label] of Object.entries(AXES)) sel.append(el("option", { value: k, text: label }));
  }
  $("#sc-x").value = "lo";
  $("#sc-y").value = "wm";
  $("#sc-x").addEventListener("change", renderScatter);
  $("#sc-y").addEventListener("change", renderScatter);
  initScatterEvents();
  // v2:比對分頁籤
  document.querySelectorAll("#cmp-tabs button").forEach((b) => {
    b.addEventListener("click", () => setCmpTab(b.dataset.tab));
  });
  setCmpTab(S.cmpTab);
  // v2:製造視角
  $("#mfg-axis").addEventListener("change", () => { S.mfg.axis = $("#mfg-axis").value; renderMfg(); });
  $("#mfg-gates").addEventListener("change", renderMfg);
  document.querySelectorAll("#mfg-table th.mfg-sort").forEach((th) => {
    th.addEventListener("click", () => {
      S.mfg.axis = th.dataset.axis;
      $("#mfg-axis").value = th.dataset.axis;
      renderMfg();
    });
  });
  $("#mfg-csv").addEventListener("click", mfgExportCsv);
  $("#mfg-png").addEventListener("click", mfgExportPng);
  // v2:研究視角
  initParetoEvents();
  $("#res-to-scatter").addEventListener("click", () => {
    setOvMode("scatter");
    nav("overview");
  });
  initTips();
  updateTrayBadge();
  route();
}
init();
