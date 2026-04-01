(function () {
  "use strict";

  const STATE_ORDER = ["UP_NARROW", "UP_WIDE", "DOWN_NARROW", "DOWN_WIDE", "SIDE_NARROW", "SIDE_WIDE"];
  const STATE_LABEL = {
    UP_NARROW: "上升-窄幅",
    UP_WIDE: "上升-宽幅",
    DOWN_NARROW: "下降-窄幅",
    DOWN_WIDE: "下降-宽幅",
    SIDE_NARROW: "横向-窄幅",
    SIDE_WIDE: "横向-宽幅",
  };
  const STATE_COLOR = {
    UP_NARROW: "rgba(44,160,44,0.95)",
    UP_WIDE: "rgba(44,160,44,0.55)",
    DOWN_NARROW: "rgba(214,39,40,0.95)",
    DOWN_WIDE: "rgba(214,39,40,0.55)",
    SIDE_NARROW: "rgba(127,127,127,0.90)",
    SIDE_WIDE: "rgba(127,127,127,0.50)",
  };

  function _num(v) {
    const x = Number(v);
    return Number.isFinite(x) ? x : NaN;
  }

  function _quantile(xs, q) {
    const arr = (Array.isArray(xs) ? xs : []).map(_num).filter(Number.isFinite).sort((a, b) => a - b);
    if (!arr.length) return NaN;
    const qq = Math.max(0, Math.min(1, _num(q)));
    const pos = (arr.length - 1) * qq;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return arr[lo];
    const w = pos - lo;
    return arr[lo] * (1 - w) + arr[hi] * w;
  }

  function _rollingSlopeAnn(logClose, i, win) {
    const s = Math.max(0, i - win + 1);
    const ys = [];
    for (let j = s; j <= i; j++) {
      const y = _num(logClose[j]);
      if (Number.isFinite(y)) ys.push(y);
    }
    const minp = Math.max(5, Math.floor(win / 2));
    if (ys.length < minp) return NaN;
    const n = ys.length;
    const xMean = (n - 1) / 2;
    const yMean = ys.reduce((a, b) => a + b, 0) / n;
    let cov = 0;
    let varx = 0;
    for (let k = 0; k < n; k++) {
      const dx = k - xMean;
      cov += dx * (ys[k] - yMean);
      varx += dx * dx;
    }
    if (varx <= 0) return NaN;
    return (cov / varx) * 252.0;
  }

  function _buildTrPct(close, high, low) {
    const n = close.length;
    const hi = (Array.isArray(high) && high.length === n) ? high.map(_num) : close.map(_num);
    const lo = (Array.isArray(low) && low.length === n) ? low.map(_num) : close.map(_num);
    const tr = new Array(n).fill(NaN);
    for (let i = 1; i < n; i++) {
      const prev = _num(close[i - 1]);
      const h = Number.isFinite(hi[i]) ? hi[i] : _num(close[i]);
      const l = Number.isFinite(lo[i]) ? lo[i] : _num(close[i]);
      if (!Number.isFinite(prev) || prev <= 0 || !Number.isFinite(h) || !Number.isFinite(l)) continue;
      const tr1 = Math.abs(h - l);
      const tr2 = Math.abs(h - prev);
      const tr3 = Math.abs(l - prev);
      tr[i] = Math.max(tr1, tr2, tr3) / prev;
    }
    return tr;
  }

  function computeRows(input) {
    const dates = Array.isArray(input && input.dates) ? input.dates : [];
    const close = Array.isArray(input && input.close) ? input.close.map(_num) : [];
    const high = Array.isArray(input && input.high) ? input.high : null;
    const low = Array.isArray(input && input.low) ? input.low : null;
    if (!dates.length || dates.length !== close.length) return [];
    const slopeWindow = Math.max(5, Math.round(_num((input && input.slope_window) || 20) || 20));
    const volWindow = Math.max(5, Math.round(_num((input && input.vol_window) || 20) || 20));
    const directionThresholdAnn = Math.max(0, _num((input && input.direction_threshold_ann) || 0.03) || 0.03);
    const volQuantile = Math.max(0.01, Math.min(0.99, _num((input && input.vol_quantile) || 0.5) || 0.5));

    const logClose = close.map((v) => (Number.isFinite(v) && v > 0 ? Math.log(v) : NaN));
    const trPct = _buildTrPct(close, high, low);
    const volMetric = new Array(close.length).fill(NaN);
    const volThreshold = new Array(close.length).fill(NaN);

    const minVol = Math.max(5, Math.floor(volWindow / 2));
    const minExpQ = Math.max(volWindow, 30);
    for (let i = 0; i < close.length; i++) {
      const s = Math.max(1, i - volWindow + 1);
      const xs = [];
      for (let j = s; j <= i; j++) {
        const v = _num(trPct[j]);
        if (Number.isFinite(v)) xs.push(v);
      }
      if (xs.length >= minVol) volMetric[i] = xs.reduce((a, b) => a + b, 0) / xs.length;
      const hist = [];
      for (let j = 0; j < i; j++) {
        const v = _num(volMetric[j]);
        if (Number.isFinite(v)) hist.push(v);
      }
      if (hist.length >= minExpQ) volThreshold[i] = _quantile(hist, volQuantile);
    }

    const rows = [];
    for (let i = 0; i < close.length; i++) {
      const sl = _rollingSlopeAnn(logClose, i, slopeWindow);
      let dir = "SIDE";
      if (Number.isFinite(sl) && sl > directionThresholdAnn) dir = "UP";
      else if (Number.isFinite(sl) && sl < -directionThresholdAnn) dir = "DOWN";
      const vm = _num(volMetric[i]);
      const vt = _num(volThreshold[i]);
      const amp = Number.isFinite(vm) && Number.isFinite(vt) && vm >= vt ? "WIDE" : "NARROW";
      rows.push({
        date: String(dates[i] || ""),
        close: _num(close[i]),
        slope_ann: sl,
        vol_metric: vm,
        vol_threshold: vt,
        regime: `${dir}_${amp}`,
      });
    }
    return rows;
  }

  function summarizeStateStats(rows) {
    const out = {};
    for (const s of STATE_ORDER) out[s] = { n: 0, rets: [] };
    for (let i = 1; i < rows.length; i++) {
      const st = String((rows[i] || {}).regime || "");
      if (!out[st]) continue;
      const a = _num(rows[i - 1].close);
      const b = _num(rows[i].close);
      if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0) continue;
      out[st].n += 1;
      out[st].rets.push(b / a - 1.0);
    }
    return out;
  }

  function summarizeForwardByState(rows, nAhead) {
    const n = Math.max(1, Math.round(_num(nAhead) || 1));
    const out = {};
    for (const s of STATE_ORDER) out[s] = [];
    for (let i = 0; i + n < rows.length; i++) {
      const st = String((rows[i] || {}).regime || "");
      if (!out[st]) continue;
      const a = _num(rows[i].close);
      const b = _num(rows[i + n].close);
      if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0) continue;
      out[st].push(b / a - 1.0);
    }
    return out;
  }

  function renderLegend(container, selected, onToggle, onIsolate) {
    if (!container) return;
    const sel = selected || new Set(STATE_ORDER);
    container.innerHTML = STATE_ORDER.map((st) => {
      const on = sel.has(st);
      const style = `display:inline-flex;align-items:center;gap:6px;margin:2px 8px 2px 0;padding:4px 8px;border-radius:12px;border:1px solid #999;cursor:pointer;opacity:${on ? "1" : "0.35"};`;
      return `<span data-regime-chip="${st}" style="${style}"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${STATE_COLOR[st]};"></span>${STATE_LABEL[st]}</span>`;
    }).join("") + `<span class="muted" style="margin-left:8px;">点击筛选；双击仅保留。</span>`;
    const chips = Array.from(container.querySelectorAll("[data-regime-chip]"));
    chips.forEach((el) => {
      const st = String(el.getAttribute("data-regime-chip") || "");
      el.onclick = () => onToggle && onToggle(st);
      el.ondblclick = () => onIsolate && onIsolate(st);
    });
  }

  function renderTimeline(plotId, rows, selected, title, codeName) {
    const el = document.getElementById(plotId);
    if (!el) return;
    if (typeof Plotly === "undefined") {
      el.innerHTML = "<div class='muted'>Plotly 未加载，无法绘图。</div>";
      return;
    }
    const sel = selected || new Set(STATE_ORDER);
    const dates = rows.map((r) => r.date);
    const close = rows.map((r) => r.close);
    const traces = [
      {
        x: dates, y: close, type: "scatter", mode: "lines",
        name: `${codeName || "标的"}（全样本）`,
        line: { color: "rgba(31,119,180,0.30)", width: 1.2 },
        hovertemplate: "%{x}<br>close=%{y:.4f}<extra></extra>",
      },
    ];
    for (const st of STATE_ORDER) {
      traces.push({
        x: dates,
        y: rows.map((r) => (r.regime === st ? r.close : null)),
        type: "scatter",
        mode: "lines",
        connectgaps: false,
        name: STATE_LABEL[st],
        line: { color: STATE_COLOR[st], width: 2.8 },
        visible: sel.has(st) ? true : "legendonly",
        hovertemplate: `%{x}<br>state=${STATE_LABEL[st]}<br>close=%{y:.4f}<extra></extra>`,
      });
    }
    Plotly.newPlot(plotId, traces, {
      margin: { t: 30, b: 40, l: 55, r: 15 },
      title: { text: title || "市场状态区间（图例可点击筛选）", font: { size: 13 } },
      yaxis: { title: "价格" },
      xaxis: { title: "日期" },
      legend: { orientation: "h", y: 1.18 },
    }, { responsive: true, displayModeBar: false });
  }

  function renderForwardBox(plotId, forwardByState, nAhead, selected) {
    const el = document.getElementById(plotId);
    if (!el) return;
    if (typeof Plotly === "undefined") {
      el.innerHTML = "<div class='muted'>Plotly 未加载，无法绘图。</div>";
      return;
    }
    const sel = selected || new Set(STATE_ORDER);
    const traces = STATE_ORDER.map((st) => ({
      x: (forwardByState[st] || []).map(() => STATE_LABEL[st]),
      y: forwardByState[st] || [],
      type: "box",
      name: STATE_LABEL[st],
      marker: { color: STATE_COLOR[st] },
      boxmean: true,
      visible: sel.has(st) ? true : "legendonly",
      hovertemplate: `状态=${STATE_LABEL[st]}<br>未来${nAhead}日收益=%{y:.2%}<extra></extra>`,
    }));
    Plotly.newPlot(plotId, traces, {
      margin: { t: 30, b: 55, l: 55, r: 15 },
      title: { text: `按状态分组的未来 ${nAhead} 日收益分布`, font: { size: 13 } },
      yaxis: { title: "未来收益", tickformat: ".1%" },
      xaxis: { title: "市场状态" },
      showlegend: false,
    }, { responsive: true, displayModeBar: false });
  }

  window.MarketRegimeUI = {
    STATE_ORDER,
    STATE_LABEL,
    STATE_COLOR,
    computeRows,
    summarizeStateStats,
    summarizeForwardByState,
    renderLegend,
    renderTimeline,
    renderForwardBox,
    quantile: _quantile,
  };
})();
