"use strict";

// Tier shown on a tile is the server's persisted AlertManager level (the one
// being logged/alerted on). The forecast ribbon shows the raw per-horizon
// probabilities that feed it.
const TIER = { GREEN: 0, AMBER: 1, RED: 2 };
const COLORS = { GREEN: "#2fbf71", AMBER: "#f5a524", RED: "#f04444" };
const ZONE_RGB = "56, 189, 248";
const STALE_MS = 3000;          // no packet for this long -> "No signal"
const RELEASE_HOLD_MS = 8000;   // keep the focused feed this long after it returns to GREEN
const MAX_LOG_ROWS = 300;

let cfg = null;
const cams = new Map();
const focus = { cam: null, mode: "auto", since: 0, greenSince: null, dismissed: new Map() };
const seenAlerts = new Set();

const $ = (id) => document.getElementById(id);
const fmtTime = (t) => new Date(t).toLocaleTimeString([], { hour12: false });

// ---------------------------------------------------------------- setup

async function init() {
  cfg = await (await fetch("/config")).json();
  if (cfg.mode === "simulated") {
    $("mode").hidden = false;
    $("mode").textContent = "SIMULATED";
    $("mode").title = "Scripted risk curves, no detector or model running";
  }
  const n = cfg.cameras.length;
  $("tiles").style.setProperty("--cols", n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4);
  $("wall").classList.toggle("single", n === 1);

  const tpl = $("tileTpl");
  cfg.cameras.forEach((meta, i) => {
    const el = tpl.content.firstElementChild.cloneNode(true);
    el.querySelector(".tile-name").textContent = meta.name;
    el.title = `${meta.name}\n${meta.source}\nClick to focus (key ${i + 1})`;
    el.addEventListener("click", () => pin(meta.id));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") pin(meta.id); });
    $("tiles").appendChild(el);
    cams.set(meta.id, {
      meta, el,
      canvas: el.querySelector("canvas"),
      badge: el.querySelector(".badge b"),
      status: el.querySelector(".status-overlay"),
      packet: null, bitmap: null, drawnSeq: -1,
      lastSeen: 0, level: "GREEN", shown: null, redAt: 0,
    });
  });

  $("closeBtn").addEventListener("click", dismissFocus);
  $("pinBtn").addEventListener("click", () => {
    if (!focus.cam) return;
    if (focus.mode === "pinned") resumeAuto();
    else { focus.mode = "pinned"; focus.since = Date.now(); renderFocusChrome(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") dismissFocus();
    const idx = Number(e.key) - 1;
    if (Number.isInteger(idx) && idx >= 0 && idx < cfg.cameras.length) pin(cfg.cameras[idx].id);
  });

  new ResizeObserver(() => {
    for (const c of cams.values()) drawTile(c);
    if (focus.cam) drawFocus(cams.get(focus.cam));
  }).observe($("wall"));

  connect();
  setInterval(tick, 500);
  tick();
}

function connect() {
  const es = new EventSource("/events");
  es.onopen = () => $("conn").classList.add("ok");
  es.onerror = () => $("conn").classList.remove("ok");
  es.addEventListener("frame", (e) => onFrame(JSON.parse(e.data)));
  es.addEventListener("alert", (e) => onAlert(JSON.parse(e.data)));
}

// ---------------------------------------------------------------- data in

async function onFrame(p) {
  const c = cams.get(p.cam);
  if (!c) return;
  c.packet = p;
  c.lastSeen = Date.now();
  if (p.level !== c.level) {
    if (TIER[p.level] === 2) c.redAt = Date.now();
    c.level = p.level;
    refreshTile(c);
    renderSummary();
  }
  const bytes = Uint8Array.from(atob(p.img), (ch) => ch.charCodeAt(0));
  const bmp = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
  if (p.seq < c.drawnSeq) { bmp.close(); return; } // an older decode finished late
  c.drawnSeq = p.seq;
  if (c.bitmap) c.bitmap.close();
  c.bitmap = bmp;
  c.frame = p; // packet the bitmap belongs to: overlays are drawn from this, never from a newer one
  refreshTile(c);
  drawTile(c);
  evaluateFocus();
  if (focus.cam === c.meta.id) drawFocus(c);
}

function onAlert(a) {
  const key = `${a.cam}|${a.t}|${a.new}`;
  if (seenAlerts.has(key)) return; // replayed history after an SSE reconnect
  seenAlerts.add(key);
  const c = cams.get(a.cam);
  const p = a.probs[String(cfg.alert_horizon)];
  const li = document.createElement("li");
  li.innerHTML =
    `<span class="t">${fmtTime(a.t * 1000)}</span>` +
    `<span class="cam"></span>` +
    `<span class="chg"><span class="lv ${a.old}">${a.old}</span>→<span class="lv ${a.new}">${a.new}</span>` +
    (p != null ? `<span>· +${cfg.alert_horizon / 30}s ${p.toFixed(2)}</span>` : "") + `</span>`;
  li.querySelector(".cam").textContent = c ? c.meta.name : a.cam;
  li.title = `frame ${a.frame_id} — click to view camera`;
  li.addEventListener("click", () => pin(a.cam));
  const list = $("logList");
  list.prepend(li);
  while (list.children.length > MAX_LOG_ROWS) list.lastChild.remove();
  $("logEmpty").hidden = true;
}

// ---------------------------------------------------------------- state

const isStale = (c) => !c.packet || Date.now() - c.lastSeen > STALE_MS;
const displayTier = (c) => (isStale(c) ? "OFFLINE" : c.level);
const peakProb = (c) => (c.packet && c.packet.probs ? Math.max(...Object.values(c.packet.probs)) : 0);

function refreshTile(c) {
  const tier = displayTier(c);
  const key = c.packet ? tier : "CONNECTING"; // never received a frame != lost signal
  if (key !== c.shown) {
    const escalatedToRed = tier === "RED" && c.shown !== null && c.shown !== "OFFLINE" && c.shown !== "CONNECTING";
    c.el.classList.remove("tier-GREEN", "tier-AMBER", "tier-RED", "tier-OFFLINE");
    c.el.classList.add(`tier-${tier}`);
    c.badge.textContent = key === "CONNECTING" ? "" : tier === "OFFLINE" ? "NO SIGNAL" : tier;
    if (escalatedToRed) {
      c.el.classList.remove("just-escalated");
      void c.el.offsetWidth; // restart the animation
      c.el.classList.add("just-escalated");
    }
    c.shown = key;
  }
  const msg = !c.packet ? "Connecting…" : isStale(c) ? "No signal" : !c.packet.probs ? "Warming up" : "";
  c.status.hidden = !msg;
  c.status.textContent = msg;
}

function tick() {
  $("clock").textContent = fmtTime(Date.now());
  for (const c of cams.values()) refreshTile(c);
  evaluateFocus();
  if (focus.cam) renderFocusChrome();
  renderSummary();
}

function renderSummary() {
  let red = 0, amber = 0, offline = 0;
  for (const c of cams.values()) {
    const t = displayTier(c);
    if (t === "RED") red++; else if (t === "AMBER") amber++; else if (t === "OFFLINE" && c.packet) offline++;
  }
  const parts = [];
  if (red) parts.push(`<span class="chip RED">${red} RED</span>`);
  if (amber) parts.push(`<span class="chip AMBER">${amber} AMBER</span>`);
  if (!red && !amber) parts.push(`<span>All clear · ${cams.size} camera${cams.size === 1 ? "" : "s"}</span>`);
  if (offline) parts.push(`<span>${offline} offline</span>`);
  $("summary").innerHTML = parts.join("");
}

// ---------------------------------------------------------------- focus policy
//
// The system pulls the operator's eye; the operator can override.
//  - Auto: the highest-tier live camera (ties -> highest forecast peak) takes
//    focus once it reaches AMBER. A camera only displaces the current one by
//    being in a strictly higher tier, so two AMBER feeds don't ping-pong.
//  - When the focused feed returns to GREEN and nothing else is elevated,
//    focus is held RELEASE_HOLD_MS before going back to the grid.
//  - "Back to grid" on an elevated feed suppresses re-focusing it until it
//    escalates further or clears.
//  - Pinned (operator clicked a tile): held, except that a camera newly
//    escalating to RED after the pin still takes over unless the pinned feed
//    is itself RED. An operator looking elsewhere must not miss a new RED.

function isDismissed(c) {
  const d = focus.dismissed.get(c.meta.id);
  return d !== undefined && TIER[c.level] <= d;
}

function evaluateFocus() {
  const now = Date.now();
  for (const [id] of focus.dismissed) if (cams.get(id).level === "GREEN") focus.dismissed.delete(id);

  const ranked = [...cams.values()]
    .filter((c) => !isStale(c) && TIER[c.level] >= 1 && !isDismissed(c))
    .sort((a, b) => TIER[b.level] - TIER[a.level] || peakProb(b) - peakProb(a));
  const best = ranked[0];
  const cur = focus.cam ? cams.get(focus.cam) : null;

  if (focus.mode === "pinned") {
    if (best && best !== cur && best.level === "RED" && best.redAt > focus.since && cur.level !== "RED") {
      setFocus(best.meta.id, "auto");
    }
    return;
  }

  if (cur && TIER[cur.level] > 0 && !isStale(cur)) focus.greenSince = null;
  if (!best) {
    if (cur && TIER[cur.level] === 0) {
      focus.greenSince ??= now;
      if (now - focus.greenSince > RELEASE_HOLD_MS) setFocus(null);
    } else if (cur && isStale(cur)) {
      setFocus(null);
    }
    return;
  }
  if (!cur || (best !== cur && TIER[best.level] > TIER[cur.level])) setFocus(best.meta.id, "auto");
}

function setFocus(id, mode = "auto") {
  const prev = focus.cam;
  focus.cam = id;
  focus.mode = id ? mode : "auto";
  focus.since = Date.now();
  focus.greenSince = null;
  for (const c of cams.values()) c.el.classList.toggle("is-focused", c.meta.id === id);
  $("wall").classList.toggle("has-focus", !!id);
  $("focus").hidden = !id;
  if (!id) { renderFocusState(); return; }

  const c = cams.get(id);
  focus.reason = mode === "auto"
    ? `Auto-focused · ${c.level} at ${fmtTime(Date.now())}`
    : `Selected by operator`;
  $("focusName").textContent = c.meta.name;
  if (mode === "auto" && prev !== id) {
    const f = $("focus");
    f.classList.remove("just-focused");
    void f.offsetWidth;
    f.classList.add("just-focused");
  }
  renderFocusChrome();
  requestAnimationFrame(() => {
    for (const other of cams.values()) drawTile(other);
    drawFocus(c);
  });
}

function pin(id) {
  focus.dismissed.delete(id);
  setFocus(id, "pinned");
}

function resumeAuto() {
  focus.mode = "auto";
  renderFocusChrome();
  evaluateFocus();
}

function dismissFocus() {
  if (!focus.cam) return;
  const c = cams.get(focus.cam);
  if (TIER[c.level] > 0) focus.dismissed.set(c.meta.id, TIER[c.level]);
  setFocus(null);
}

function renderFocusState() {
  const el = $("focusState");
  if (focus.mode === "pinned") {
    el.innerHTML = `Pinned by operator <button class="btn" type="button">Resume auto-focus</button>`;
    el.querySelector("button").onclick = resumeAuto;
  } else {
    el.textContent = "Auto-focus on";
  }
}

function renderFocusChrome() {
  const c = cams.get(focus.cam);
  const tier = displayTier(c);
  const f = $("focus");
  f.classList.remove("tier-GREEN", "tier-AMBER", "tier-RED", "tier-OFFLINE");
  f.classList.add(`tier-${tier}`);
  const pill = $("focusTier");
  pill.className = `tier-pill ${tier}`;
  pill.textContent = tier === "OFFLINE" ? "NO SIGNAL" : tier;
  $("focusReason").textContent = focus.mode === "pinned" ? "Pinned by operator" : focus.reason;
  const pinBtn = $("pinBtn");
  pinBtn.textContent = focus.mode === "pinned" ? "Unpin" : "Pin";
  pinBtn.classList.toggle("on", focus.mode === "pinned");
  const p = c.packet;
  $("focusMeta").textContent = p
    ? `${cfg.mode === "simulated" ? "simulated" : `${p.boxes.length} tracked`} · ${p.fps.toFixed(1)} fps`
    : "";
  const msg = !p ? "Connecting…" : isStale(c) ? "No signal" : !p.probs ? "Warming up · forecast needs ~5 s of history" : "";
  $("focusStatus").hidden = !msg;
  $("focusStatus").textContent = msg;
  renderFocusState();
}

// ---------------------------------------------------------------- drawing

function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { ctx: canvas.getContext("2d"), w, h, dpr };
}

function drawVideo(canvas, bitmap) {
  const { ctx, w, h, dpr } = fitCanvas(canvas);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  if (!bitmap) return null;
  const s = Math.min(w / bitmap.width, h / bitmap.height);
  const box = { x: (w - bitmap.width * s) / 2, y: (h - bitmap.height * s) / 2, w: bitmap.width * s, h: bitmap.height * s };
  ctx.drawImage(bitmap, box.x, box.y, box.w, box.h);
  return { ctx, box, dpr };
}

function drawTile(c) {
  if (c.el.offsetParent === null) return; // hidden (it's the focused feed)
  drawVideo(c.canvas, c.bitmap);
}

function drawFocus(c) {
  const v = drawVideo($("focusCanvas"), c.bitmap);
  const p = c.frame;
  if (v && p) drawOverlay(v, p);
  drawRibbon(p ? p.probs : null, c.level);
}

function drawOverlay({ ctx, box, dpr }, p) {
  const [rows, cols] = p.grid;
  const zw = box.w / cols, zh = box.h / rows;

  if (p.zone_flow) {
    let hot = -1, hotV = 0.5;
    p.zone_flow.forEach((v, i) => {
      if (v > hotV) { hot = i; hotV = v; }
      if (v < 0.15) return;
      ctx.fillStyle = `rgba(${ZONE_RGB}, ${Math.min(0.4, v * 0.45).toFixed(3)})`;
      ctx.fillRect(box.x + (i % cols) * zw, box.y + Math.floor(i / cols) * zh, zw, zh);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = dpr;
    ctx.beginPath();
    for (let cI = 1; cI < cols; cI++) { const x = box.x + cI * zw; ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); }
    for (let r = 1; r < rows; r++) { const y = box.y + r * zh; ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); }
    ctx.stroke();
    if (hot >= 0) {
      ctx.strokeStyle = `rgb(${ZONE_RGB})`;
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(box.x + (hot % cols) * zw + dpr, box.y + Math.floor(hot / cols) * zh + dpr, zw - 2 * dpr, zh - 2 * dpr);
    }
  }

  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1.25 * dpr;
  for (const [x1, y1, x2, y2] of p.boxes) {
    ctx.strokeRect(box.x + x1 * box.w, box.y + y1 * box.h, (x2 - x1) * box.w, (y2 - y1) * box.h);
  }
}

function tierOf(p) {
  return p >= cfg.red_threshold ? "RED" : p >= cfg.amber_threshold ? "AMBER" : "GREEN";
}

function forecastHeadline(probs, level) {
  if (!probs) return { tier: "", text: "Waiting for forecast", sub: "Model needs ~5 s of history" };
  const ps = cfg.horizons.map((k) => probs[String(k)]);
  const secs = cfg.horizon_seconds;
  const firstRed = ps.findIndex((p) => p >= cfg.red_threshold);
  const firstAmber = ps.findIndex((p) => p >= cfg.amber_threshold);
  let text, tier;
  if (firstRed === 0) { text = "RED now"; tier = "RED"; }
  else if (firstRed > 0) { text = `RED within ${secs[firstRed]} s`; tier = "RED"; }
  else if (firstAmber === 0) { text = "AMBER now"; tier = "AMBER"; }
  else if (firstAmber > 0) { text = `AMBER within ${secs[firstAmber]} s`; tier = "AMBER"; }
  else { text = "Below alert levels"; tier = "GREEN"; }
  const d = ps[ps.length - 1] - ps[0];
  let sub = d > 0.08 ? "Risk rising" : d < -0.08 ? "Risk easing" : "Risk steady";
  // The alert level clears only after sustained low risk (AlertManager fall
  // persistence), so it can legitimately outlast the forecast. Say so rather
  // than leave the pill and the headline looking contradictory.
  if (TIER[level] > TIER[tier]) sub = `${level} alert holding until risk stays low`;
  return { tier, text, sub };
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

// One continuous shape across now → +3 s: height is probability, colour is
// the tier at that horizon, so "getting worse to the right" reads at a glance
// without parsing numbers.
function drawRibbon(probs, level) {
  const hl = forecastHeadline(probs, level);
  const head = $("forecastHeadline");
  head.className = `forecast-headline ${hl.tier}`;
  head.innerHTML = `${hl.text}<small>${hl.sub}</small>`;

  const { ctx, w, h, dpr } = fitCanvas($("ribbon"));
  ctx.clearRect(0, 0, w, h);
  const padL = 52 * dpr, padR = 14 * dpr, padT = 8 * dpr, padB = 26 * dpr;
  const n = cfg.horizons.length;
  const X = (i) => padL + (i * (w - padL - padR)) / (n - 1);
  const Y = (p) => padT + (1 - p) * (h - padT - padB);
  const plotW = w - padL - padR;

  // threshold bands + lines
  ctx.fillStyle = hexA(COLORS.RED, 0.07);
  ctx.fillRect(padL, Y(1), plotW, Y(cfg.red_threshold) - Y(1));
  ctx.fillStyle = hexA(COLORS.AMBER, 0.06);
  ctx.fillRect(padL, Y(cfg.red_threshold), plotW, Y(cfg.amber_threshold) - Y(cfg.red_threshold));
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.lineWidth = dpr;
  ctx.font = `600 ${10 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const [name, t] of [["RED", cfg.red_threshold], ["AMBER", cfg.amber_threshold]]) {
    ctx.strokeStyle = hexA(COLORS[name], 0.55);
    ctx.beginPath(); ctx.moveTo(padL, Y(t)); ctx.lineTo(w - padR, Y(t)); ctx.stroke();
    ctx.fillStyle = hexA(COLORS[name], 0.85);
    ctx.fillText(name, padL - 8 * dpr, Y(t));
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = "#242b34";
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();

  // horizon labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`;
  ctx.fillStyle = "#7f8a96";
  cfg.horizon_seconds.forEach((s, i) => ctx.fillText(s === 0 ? "Now" : `+${s}s`, X(i), Y(0) + 7 * dpr));

  if (!probs) return;
  const ps = cfg.horizons.map((k) => Math.min(1, Math.max(0, probs[String(k)])));
  const fill = ctx.createLinearGradient(X(0), 0, X(n - 1), 0);
  const line = ctx.createLinearGradient(X(0), 0, X(n - 1), 0);
  ps.forEach((p, i) => {
    fill.addColorStop(i / (n - 1), hexA(COLORS[tierOf(p)], 0.32));
    line.addColorStop(i / (n - 1), COLORS[tierOf(p)]);
  });

  ctx.beginPath();
  ctx.moveTo(X(0), Y(0));
  ps.forEach((p, i) => ctx.lineTo(X(i), Y(p)));
  ctx.lineTo(X(n - 1), Y(0));
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.beginPath();
  ps.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p)) : ctx.moveTo(X(i), Y(p))));
  ctx.strokeStyle = line;
  ctx.lineWidth = 3 * dpr;
  ctx.lineJoin = "round";
  ctx.stroke();

  ps.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(X(i), Y(p), 5 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[tierOf(p)];
    ctx.fill();
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = "#11151a";
    ctx.stroke();
  });
}

init();
