function _finiteNums(arr) {
  const out = [];
  for (const v of arr || []) {
    const x = Number(v);
    if (Number.isFinite(x)) out.push(x);
  }
  return out;
}

function _minmax(arr) {
  const xs = _finiteNums(arr);
  if (xs.length === 0) return { min: 0, max: 1 };
  let mn = xs[0], mx = xs[0];
  for (const x of xs) { if (x < mn) mn = x; if (x > mx) mx = x; }
  if (mn === mx) { mn *= 0.99; mx *= 1.01; }
  return { min: mn, max: mx };
}

function _downsample(xs, maxN) {
  const n = xs.length;
  if (n <= maxN) return { idx: xs.map((_, i) => i) };
  const step = Math.ceil(n / maxN);
  const idx = [];
  for (let i = 0; i < n; i += step) idx.push(i);
  if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
  return { idx };
}

function _palette(theme) {
  const t = String(theme || "light");
  if (t === "dark") {
    return {
      bg: "#161a22",
      text: "#eaeaea",
      muted: "#a8b0bd",
      grid: "#2a2f3a",
      cellText: "#ffffff",
    };
  }
  return {
    bg: "#ffffff",
    text: "#111111",
    muted: "#666666",
    grid: "#eeeeee",
    cellText: "#111111",
  };
}

function drawLineChart(ctx, { width, height, x, series, yMode = "linear", yFixed = null, title = "", yLabelFmt = null, theme = "light" }) {
  const padL = 46, padR = 16, padT = 18, padB = 22;
  const W = width, H = height;
  const pal = _palette(theme);
  ctx.clearRect(0, 0, W, H);
  ctx.setFillStyle(pal.bg);
  ctx.fillRect(0, 0, W, H);

  if (title) {
    ctx.setFillStyle(pal.text);
    ctx.setFontSize(12);
    ctx.fillText(title, padL, 14);
  }

  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // gather y range
  let allY = [];
  for (const s of series) allY = allY.concat(s.y || []);
  let yr = yFixed ? { min: yFixed[0], max: yFixed[1] } : _minmax(allY);

  function yMap(v) {
    const x = Number(v);
    if (!Number.isFinite(x)) return null;
    let val = x;
    if (yMode === "log") {
      if (val <= 0) return null;
      val = Math.log(val);
    }
    let mn = yr.min, mx = yr.max;
    if (yMode === "log") { mn = Math.log(Math.max(yr.min, 1e-12)); mx = Math.log(Math.max(yr.max, 1e-12)); }
    const t = (val - mn) / (mx - mn);
    return padT + (1 - t) * plotH;
  }

  function xMap(i, n) {
    if (n <= 1) return padL;
    return padL + (i / (n - 1)) * plotW;
  }

  // grid
  ctx.setStrokeStyle(pal.grid);
  ctx.setLineWidth(1);
  for (let k = 0; k <= 4; k++) {
    const yv = padT + (k / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, yv);
    ctx.lineTo(W - padR, yv);
    ctx.stroke();
  }

  // axes labels (min/max)
  ctx.setFillStyle(pal.muted);
  ctx.setFontSize(10);
  const yMin = yr.min;
  const yMax = yr.max;
  const fmt = (typeof yLabelFmt === "function") ? yLabelFmt : ((v) => Number(v).toFixed(4));
  ctx.fillText(String(fmt(yMax)), 2, padT + 10);
  ctx.fillText(String(fmt(yMin)), 2, padT + plotH);

  const n = (x || []).length;
  const ds = _downsample(x || [], 220);
  for (const s of series) {
    ctx.setStrokeStyle(s.color || "#1677ff");
    ctx.setLineWidth(s.width || 1.5);
    if (s.dash) ctx.setLineDash([4, 3], 0);
    else ctx.setLineDash([], 0);
    ctx.beginPath();
    let started = false;
    for (const ii of ds.idx) {
      const yy = yMap((s.y || [])[ii]);
      if (yy == null) { started = false; continue; }
      const xx = xMap(ii, n);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  }
  ctx.setLineDash([], 0);

  // x labels (first/last)
  if (n > 0) {
    ctx.setFillStyle(pal.muted);
    ctx.setFontSize(10);
    const first = String(x[0]);
    const last = String(x[n - 1]);
    ctx.fillText(first, padL, H - 6);
    const wLast = last.length * 6;
    ctx.fillText(last, W - padR - wLast, H - 6);
  }

  ctx.draw();
}

function drawHeatmap4(ctx, { width, height, title, labels, matrix, theme = "light" }) {
  const pad = 16;
  const W = width, H = height;
  const pal = _palette(theme);
  ctx.clearRect(0, 0, W, H);
  ctx.setFillStyle(pal.bg);
  ctx.fillRect(0, 0, W, H);
  ctx.setFillStyle(pal.text);
  ctx.setFontSize(12);
  ctx.fillText(title || "相关性", pad, 14);

  const n = (labels || []).length;
  if (!n) { ctx.draw(); return; }
  const gridTop = 24;
  // estimate left label width and center the matrix area in remaining space
  const leftLabelW = 18; // labels are "1..4"; keep a small gutter
  const availW = Math.max(1, W - leftLabelW - pad);
  const availH = Math.max(1, H - gridTop - pad);
  const cell = Math.min(availW / n, availH / n);
  const gridW = cell * n;
  const gridLeft = leftLabelW + Math.max(0, (availW - gridW) / 2);

  function color(v) {
    const x = Math.max(-1, Math.min(1, Number(v)));
    // green(-1) -> white(0) -> red(+1) (avoid purple/blue tint)
    if (!Number.isFinite(x)) return (theme === "dark") ? "rgb(45,45,45)" : "rgb(240,240,240)";
    if (theme === "dark") {
      // dark background: use deeper colors with dark center
      if (x >= 0) {
        const t = x;
        const r = Math.round(60 + 170 * t);
        const g = Math.round(60 * (1 - t));
        const b = Math.round(60 * (1 - t));
        return `rgb(${r},${g},${b})`;
      }
      const t = -x;
      const r = Math.round(60 * (1 - t));
      const g = Math.round(80 + 160 * t);
      const b = Math.round(60 * (1 - t));
      return `rgb(${r},${g},${b})`;
    }
    if (x >= 0) {
      // 0..1 : white -> red
      const t = x;
      const r = 255;
      const g = Math.round(255 * (1 - t));
      const b = Math.round(255 * (1 - t));
      return `rgb(${r},${g},${b})`;
    }
    // -1..0 : green -> white
    const t = -x;
    const r = Math.round(255 * (1 - t));
    const g = 255;
    const b = Math.round(255 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }

  ctx.setFontSize(10);
  ctx.setFillStyle(pal.muted);
  for (let i = 0; i < n; i++) {
    const lab = String(labels[i]);
    ctx.fillText(lab, 2, gridTop + i * cell + cell * 0.65);
    ctx.fillText(lab, gridLeft + i * cell + 2, gridTop - 4);
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix && matrix[i] ? matrix[i][j] : 0;
      ctx.setFillStyle(color(v));
      ctx.fillRect(gridLeft + j * cell, gridTop + i * cell, cell - 1, cell - 1);
      ctx.setFillStyle(pal.cellText);
      ctx.setFontSize(9);
      const txt = Number(v).toFixed(2);
      ctx.fillText(txt, gridLeft + j * cell + 2, gridTop + i * cell + cell * 0.65);
    }
  }
  ctx.draw();
}

function lineIndexFromTouchX(xRel, width, n) {
  const padL = 46, padR = 16;
  if (!Number.isFinite(xRel) || !Number.isFinite(width) || !Number.isFinite(n) || n <= 0) return null;
  const W = width;
  const plotW = W - padL - padR;
  if (plotW <= 1) return 0;
  const x = Math.max(padL, Math.min(W - padR, Number(xRel)));
  const t = (x - padL) / plotW;
  const idx = Math.round(t * (n - 1));
  return Math.max(0, Math.min(n - 1, idx));
}

function drawBarChart(ctx, { width, height, title, labels, values, colors = [], valueFmt = (x) => String(x), vmin = null, vmax = null, theme = "light" }) {
  const padL = 70, padR = 16, padT = 18, padB = 18;
  const W = width, H = height;
  const pal = _palette(theme);
  ctx.clearRect(0, 0, W, H);
  ctx.setFillStyle(pal.bg);
  ctx.fillRect(0, 0, W, H);
  ctx.setFillStyle(pal.text);
  ctx.setFontSize(12);
  ctx.fillText(title || "", padL, 14);

  const n = (labels || []).length;
  if (!n) { ctx.draw(); return; }

  const xs = (values || []).map((v) => Number(v)).filter((x) => Number.isFinite(x));
  const mn = (vmin != null) ? Number(vmin) : Math.min(0, ...(xs.length ? xs : [0]));
  const mx = (vmax != null) ? Number(vmax) : Math.max(1e-9, ...(xs.length ? xs : [1]));

  const plotW = W - padL - padR;
  const rowH = (H - padT - padB) / n;

  ctx.setFontSize(10);
  ctx.setFillStyle(pal.muted);
  for (let i = 0; i < n; i++) {
    const lab = String(labels[i]);
    ctx.fillText(lab, 2, padT + i * rowH + rowH * 0.7);
  }

  for (let i = 0; i < n; i++) {
    const v = Number((values || [])[i]);
    const ok = Number.isFinite(v);
    const t = ok ? (v - mn) / (mx - mn) : 0;
    const w = Math.max(0, Math.min(1, t)) * plotW;
    const y = padT + i * rowH + rowH * 0.18;
    const h = rowH * 0.64;
    ctx.setFillStyle(colors[i] || "#1677ff");
    ctx.fillRect(padL, y, w, h);
    ctx.setFillStyle(pal.text);
    ctx.setFontSize(10);
    ctx.fillText(ok ? valueFmt(v) : "-", padL + w + 6, y + h * 0.72);
  }
  ctx.draw();
}

function _heatColor(v) {
  const x = Math.max(-0.2, Math.min(0.2, Number(v))); // clamp to +/-20% for contrast
  if (!Number.isFinite(x)) return "rgb(240,240,240)";
  // negative -> red, positive -> green
  const t = (x + 0.2) / 0.4; // 0..1
  const r = Math.round(220 * (1 - t) + 240 * t * 0.2);
  const g = Math.round(220 * t + 240 * (1 - t) * 0.2);
  const b = Math.round(230 * 0.9);
  return `rgb(${r},${g},${b})`;
}

function drawCalendarDailyHeatmap(ctx, { width, height, title, dates, values, maxDays = 252 }) {
  const W = width, H = height;
  ctx.clearRect(0, 0, W, H);
  ctx.setFillStyle("#ffffff");
  ctx.fillRect(0, 0, W, H);
  ctx.setFillStyle("#111");
  ctx.setFontSize(12);
  ctx.fillText(title || "日度收益热力图", 16, 14);

  const n0 = (dates || []).length;
  if (!n0) { ctx.draw(); return; }

  const n = Math.min(n0, maxDays);
  const d = dates.slice(n0 - n);
  const v = values.slice(n0 - n);

  const padL = 26, padT = 24, padR = 12, padB = 18;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // map to (week, weekday)
  const cols = Math.ceil(n / 7);
  const cell = Math.min(plotW / cols, plotH / 7);
  const gridW = cell * cols;
  const gridH = cell * 7;

  // weekday labels
  const wdl = ["一","二","三","四","五","六","日"];
  ctx.setFontSize(9);
  ctx.setFillStyle("#666");
  for (let i = 0; i < 7; i++) {
    ctx.fillText(wdl[i], 2, padT + i * cell + cell * 0.7);
  }

  for (let i = 0; i < n; i++) {
    const col = Math.floor(i / 7);
    const row = i % 7;
    const x = padL + col * cell;
    const y = padT + row * cell;
    ctx.setFillStyle(_heatColor(v[i]));
    ctx.fillRect(x, y, cell - 1, cell - 1);
  }

  // range label
  ctx.setFillStyle("#666");
  ctx.setFontSize(10);
  ctx.fillText(`${String(d[0])} ~ ${String(d[d.length - 1])}`, padL, H - 6);
  ctx.draw();
}

function drawCalendarMonthHeatmap(ctx, { width, height, title, dates, values }) {
  const W = width, H = height;
  ctx.clearRect(0, 0, W, H);
  ctx.setFillStyle("#ffffff");
  ctx.fillRect(0, 0, W, H);
  ctx.setFillStyle("#111");
  ctx.setFontSize(12);
  ctx.fillText(title || "月度收益热力图", 16, 14);

  const n = (dates || []).length;
  if (!n) { ctx.draw(); return; }

  // parse years and months from date strings (YYYY-MM-DD at month end)
  const ym = [];
  for (let i = 0; i < n; i++) {
    const s = String(dates[i]);
    const y = Number(s.slice(0, 4));
    const m = Number(s.slice(5, 7));
    if (Number.isFinite(y) && Number.isFinite(m)) ym.push({ y, m, v: Number(values[i]) });
  }
  const years = Array.from(new Set(ym.map((x) => x.y))).sort((a, b) => a - b);
  if (!years.length) { ctx.draw(); return; }

  const padL = 40, padT = 24, padR = 10, padB = 18;
  const cols = 12;
  const rows = years.length;
  const cell = Math.min((W - padL - padR) / cols, (H - padT - padB) / rows);

  ctx.setFontSize(9);
  ctx.setFillStyle("#666");
  for (let c = 0; c < 12; c++) ctx.fillText(String(c + 1), padL + c * cell + 2, padT - 4);
  for (let r = 0; r < rows; r++) ctx.fillText(String(years[r]), 2, padT + r * cell + cell * 0.7);

  const map = {};
  for (const x of ym) map[`${x.y}-${x.m}`] = x.v;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const key = `${years[r]}-${c + 1}`;
      const v = map[key];
      ctx.setFillStyle(_heatColor(v));
      ctx.fillRect(padL + c * cell, padT + r * cell, cell - 1, cell - 1);
    }
  }
  ctx.draw();
}

module.exports = { drawLineChart, drawHeatmap4, drawBarChart, drawCalendarDailyHeatmap, drawCalendarMonthHeatmap, lineIndexFromTouchX };

