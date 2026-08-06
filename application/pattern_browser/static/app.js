/* Pattern Browser SPA — vanilla JS,無 build step。規格=../SPEC.md
 * 渲染方向鐵則:pixel (i,j) → rect(x=j*s, y=i*s) ⇒ i 朝下、j 朝右=饋線邊在圖下緣。
 * 菱形站點:server 給 HFSS mm 座標 [cx,cy,w],轉圖座標 x=cy/0.2*s, y=(5-cx)/0.2*s。 */
"use strict";

const GRID = 25;
const COLORS = {
  metal: "#33322f", empty: "#f3f2ee",
  both: "#b8b6ae", aOnly: "#2a78d6", bOnly: "#e34948",
  site: "rgba(235,104,52,0.9)", siteEdge: "#a54312",
  bar: "#2a78d6", gridline: "#e1e0d9", axis: "#c3c2b7", muted: "#898781",
};

const S = {
  view: null,
  detailId: null,
  tray: loadLS("pb_tray_v1", []),
  groups: loadLS("pb_groups_v1", { groups: [] }),
  activeGroup: loadLS("pb_active_group", null),
  list: { offset: 0, limit: 50, sort: "wm", dir: "desc", total: 0 },
  patternCache: new Map(),   // id → /api/pattern 回應
  statsLoaded: false,
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
  // items = [{label, count}];文字用 ink tokens、序列色=已驗證 slot-1 藍
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
  // 水平格線+刻度(遞減灰階,不搶資料)
  for (const frac of [0.5, 1]) {
    const yv = Math.round(maxC * frac);
    const y = pad.t + plotH - (yv / maxC) * plotH;
    ctx.strokeStyle = COLORS.gridline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y + 0.5); ctx.lineTo(W - pad.r, y + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.textAlign = "right";
    ctx.fillText(String(yv), pad.l - 4, y + 3);
  }
  // 基線
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + plotH + 0.5); ctx.lineTo(W - pad.r, pad.t + plotH + 0.5); ctx.stroke();
  // 長條
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
      const tip = $("#tooltip");
      if (hit) {
        tip.textContent = `${hit.label}:${hit.count} 筆`;
        tip.hidden = false;
        tip.style.left = `${ev.pageX + 12}px`;
        tip.style.top = `${ev.pageY - 24}px`;
      } else tip.hidden = true;
    });
    cv.addEventListener("mouseleave", () => { $("#tooltip").hidden = true; });
  }
}

/* ---------------- 路由 ---------------- */
const VIEWS = ["overview", "detail", "compare", "groups"];
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
  fetchList();
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
  const tb = $("#list-table tbody");
  tb.textContent = "";
  for (const row of data.rows) {
    const wmCell = el("td", { class: "num" + (row.wm !== null && row.wm >= 0 ? " wm-pos" : ""), text: fmt(row.wm) });
    const abl = el("td", {});
    if (row.has_db100) abl.append(el("span", { class: "abl-badge", text: `db100 ${fmt(row.db100_wm)}` }));
    if (row.has_sl100) abl.append(el("span", { class: "abl-badge", text: `sl100 ${fmt(row.sl100_wm)}` }));
    if (!row.has_db100 && !row.has_sl100) abl.textContent = "—";
    const tr = el("tr", { onclick: () => nav("detail", row.id) },
      el("td", {}, makeThumb(b64ToBits(row.bits_b64))),
      el("td", { text: row.id }),
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
  const last = data.offset + data.rows.length;
  $("#pg-info").textContent = S.list.total ? `${data.offset + 1}–${last} / ${S.list.total}` : "0 筆";
  $("#pg-prev").disabled = data.offset <= 0;
  $("#pg-next").disabled = last >= S.list.total;
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
  ];
  for (const [k, v] of fields) dl.append(el("dt", { text: k }), el("dd", { text: v }));
  // 消融變體對照
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
async function renderCompare() {
  const ed = $("#tray-editor");
  ed.textContent = "";
  for (const id of S.tray) {
    ed.append(el("span", { class: "chip" }, el("b", { text: id }),
      el("button", { text: "✕", onclick: () => removeTray(id) })));
  }
  const input = el("input", { type: "text", placeholder: "輸入 id 加入…" });
  const addBtn = el("button", {
    text: "加入", onclick: async () => {
      const id = input.value.trim();
      if (!id) return;
      try { await getPattern(id); addTray(id); input.value = ""; }
      catch (e) { toast(`加入失敗:${e.message}`, true); }
    },
  });
  input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); addBtn.click(); } });
  ed.append(input, addBtn);
  if (S.tray.length) ed.append(el("button", { text: "清空", onclick: () => { S.tray = []; saveLS("pb_tray_v1", S.tray); updateTrayBadge(); renderCompare(); } }));

  const hint = $("#compare-hint"), grid = $("#compare-grid"), xor = $("#xor-panel");
  grid.textContent = ""; xor.textContent = "";
  if (S.tray.length < 2) { hint.textContent = "至少選 2 筆(總攬/詳情按「比對＋」,或上方輸入 id)。"; return; }
  hint.textContent = "";
  let items;
  try { items = await api(`/api/compare?ids=${S.tray.map(encodeURIComponent).join(",")}`); }
  catch (e) { toast(`比對載入失敗:${e.message}`, true); return; }
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
  // XOR 差異(基準 A vs 其他) + 兩兩距離
  const ids = items.map((it) => it.id);
  let baseId = xor._baseId && ids.includes(xor._baseId) ? xor._baseId : ids[0];
  const sel = el("select", {
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

/* ---------------- 群組 ---------------- */
function findGroup(name) { return S.groups.groups.find((g) => g.name === name); }
function saveGroups() { saveLS("pb_groups_v1", S.groups); saveLS("pb_active_group", S.activeGroup); }
function addToActiveGroup(id) {
  let g = S.activeGroup && findGroup(S.activeGroup);
  if (!g) {
    const name = prompt("尚無使用中群組,輸入新群組名稱:", "我的群組");
    if (!name) return;
    g = findGroup(name) || { name, ids: [] };
    if (!findGroup(name)) S.groups.groups.push(g);
    S.activeGroup = name;
  }
  if (g.ids.includes(id)) { toast(`${id} 已在群組「${g.name}」`); return; }
  g.ids.push(id);
  saveGroups();
  toast(`已加入群組「${g.name}」(${g.ids.length} 筆)`);
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
        text: "✎", title: "改名", onclick: (ev) => {
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
        text: "✕", title: "刪除", onclick: (ev) => {
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
    el("button", { text: "勾選→送比對", onclick: () => {
      const checked = [...box.querySelectorAll(".member-card input:checked")].map((c) => c.dataset.id);
      const ids = checked.length ? checked : g.ids.slice(0, 4);
      if (ids.length < 2 || ids.length > 4) { toast("比對需 2–4 筆(勾選 2–4 筆成員)", true); return; }
      S.tray = ids;
      saveLS("pb_tray_v1", S.tray);
      updateTrayBadge();
      nav("compare");
    } }),
    el("button", { text: "移除勾選", onclick: () => {
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

/* ---------------- 初始化 ---------------- */
function init() {
  document.querySelectorAll("#nav button").forEach((b) => {
    b.addEventListener("click", () => nav(b.dataset.view));
  });
  $("#filters").addEventListener("submit", (ev) => {
    ev.preventDefault();
    S.list.offset = 0;
    S.list.limit = parseInt($("#limit").value, 10) || 50;
    fetchList();
  });
  $("#filters-reset").addEventListener("click", () => {
    $("#filters").reset();
    S.list = { offset: 0, limit: 50, sort: "wm", dir: "desc", total: 0 };
    fetchList();
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
  updateTrayBadge();
  route();
}
init();
