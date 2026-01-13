const { request } = require("../utils/api");
const { drawLineChart, drawHeatmap4, drawBarChart } = require("../utils/charts");

function ymd(d) {
  // d: "YYYY-MM-DD"
  return String(d || "").split("-").join("");
}

function pct(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "-";
  return (v * 100).toFixed(2) + "%";
}

function num(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "-";
  return v.toFixed(4);
}

const RANGE_MAP = {
  "1m": { label: "近一月", days: 21 },
  "3m": { label: "近三月", days: 63 },
  "6m": { label: "近半年", days: 126 },
  "1y": { label: "近一年", days: 252 },
  "3y": { label: "近三年", days: 756 },
  "5y": { label: "近五年", days: 1260 },
  "10y": { label: "近十年", days: 2520 },
  "all": { label: "全区间", days: null },
};

function _ymFromIso(d) {
  const s = String(d || "");
  return s.length >= 7 ? s.slice(0, 7) : "";
}

function _pctText(v) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "";
  return (x * 100).toFixed(2) + "%";
}

function _buildDailyMonthState(calDaily) {
  const dates = (calDaily && calDaily.dates) ? calDaily.dates : [];
  const values = (calDaily && calDaily.values) ? calDaily.values : [];
  const map = {};
  const months = [];
  for (let i = 0; i < dates.length; i++) {
    const d = String(dates[i]);
    const v = Number(values[i]);
    map[d] = v;
    const ym = _ymFromIso(d);
    if (ym) months.push(ym);
  }
  const uniqMonths = Array.from(new Set(months)).sort();
  const month = uniqMonths.length ? uniqMonths[uniqMonths.length - 1] : "";
  return { map, months: uniqMonths, month };
}

function _daysInMonth(year, month1to12) {
  return new Date(year, month1to12, 0).getDate();
}

function _buildMonthCells(ym, dailyMap) {
  // ym: "YYYY-MM"
  if (!ym) return [];
  const y = Number(ym.slice(0, 4));
  const m = Number(ym.slice(5, 7));
  if (!Number.isFinite(y) || !Number.isFinite(m)) return [];
  const first = new Date(y, m - 1, 1);
  const startDow = first.getDay(); // 0=Sun..6=Sat (Sun is first day)
  const nDays = _daysInMonth(y, m);

  const cells = [];
  for (let i = 0; i < startDow; i++) {
    cells.push({ k: `e-${i}`, kind: "empty", dayText: "", retText: "" });
  }
  for (let d = 1; d <= nDays; d++) {
    const dd = String(d).padStart(2, "0");
    const ds = `${ym}-${dd}`;
    const v = dailyMap[ds];
    const isTrade = (v != null) && Number.isFinite(Number(v));
    const vv = Number(v);
    const signKind = isTrade ? (vv > 0 ? "pos" : (vv < 0 ? "neg" : "")) : "";
    const kind = isTrade ? `trade ${signKind}` : "";
    cells.push({ k: ds, kind, dayText: String(d), retText: isTrade ? _pctText(vv) : "" });
  }
  while (cells.length % 7 !== 0) {
    const i = cells.length;
    cells.push({ k: `t-${i}`, kind: "empty", dayText: "", retText: "" });
  }
  return cells;
}

function _buildMonthlyYearCells(calMonthly) {
  const dates = (calMonthly && calMonthly.dates) ? calMonthly.dates : [];
  const values = (calMonthly && calMonthly.values) ? calMonthly.values : [];
  const map = {}; // year -> month -> value
  const years = [];
  for (let i = 0; i < dates.length; i++) {
    const s = String(dates[i]); // "YYYY-MM-DD"
    const y = Number(s.slice(0, 4));
    const m = Number(s.slice(5, 7));
    const v = Number(values[i]);
    if (!Number.isFinite(y) || !Number.isFinite(m)) continue;
    if (!map[y]) map[y] = {};
    map[y][m] = v;
    years.push(y);
  }
  const uniqYears = Array.from(new Set(years)).sort((a, b) => a - b);
  const out = [];
  for (const y of uniqYears) {
    const cells = [];
    for (let m = 1; m <= 12; m++) {
      const v = map[y] ? map[y][m] : null;
      const ok = v != null && Number.isFinite(Number(v));
      const vv = Number(v);
      const kind = ok ? (vv > 0 ? "pos" : (vv < 0 ? "neg" : "")) : "";
      cells.push({
        k: `${y}-${m}`,
        kind,
        title: `${m}月`,
        retText: ok ? _pctText(vv) : "-",
      });
    }
    out.push({ year: String(y), cells });
  }
  return out;
}

function _buildYearlyCells(calYearly) {
  const dates = (calYearly && calYearly.dates) ? calYearly.dates : [];
  const values = (calYearly && calYearly.values) ? calYearly.values : [];
  const out = [];
  for (let i = 0; i < dates.length; i++) {
    const s = String(dates[i]); // "YYYY-MM-DD"
    const y = s.slice(0, 4);
    const v = Number(values[i]);
    const ok = Number.isFinite(v);
    const kind = ok ? (v > 0 ? "pos" : (v < 0 ? "neg" : "")) : "";
    out.push({ year: y, kind, retText: ok ? _pctText(v) : "-" });
  }
  return out;
}

function pickRangeDates(dates, key) {
  const n = (dates || []).length;
  if (!n) return { start: null, end: null, label: "-" };
  const cfg = RANGE_MAP[key] || RANGE_MAP.all;
  if (!cfg.days) return { start: dates[0], end: dates[n - 1], label: cfg.label };
  const i0 = Math.max(0, n - cfg.days);
  return { start: dates[i0], end: dates[n - 1], label: cfg.label };
}

async function measure(page, id) {
  return new Promise((resolve) => {
    wx.createSelectorQuery()
      .in(page)
      .select(`#${id}`)
      .boundingClientRect((r) => resolve(r))
      .exec();
  });
}

async function drawAll(page, payload) {
  const dates = payload.dates || [];
  const nav = payload.nav || [];
  const ema252 = payload.ema252 || [];
  const bbU = payload.bb_upper || [];
  const bbL = payload.bb_lower || [];
  const dd = payload.drawdown || [];
  const rsi = payload.rsi24 || [];
  const rr3y = payload.roll3y_return || [];
  const rdd3y = payload.roll3y_mdd || [];

  const rNav = await measure(page, "cNav");
  drawLineChart(wx.createCanvasContext("cNav", page), {
    width: Math.max(10, Math.floor(rNav.width || 320)),
    height: Math.max(10, Math.floor(rNav.height || 200)),
    x: dates,
    yMode: "log",
    title: "EW NAV (log)",
    series: [
      { name: "NAV", y: nav, color: "#1677ff", width: 1.6 },
      { name: "EMA252", y: ema252, color: "#ff7a00", width: 1.2, dash: true },
      { name: "BBU", y: bbU, color: "#999", width: 1.0, dash: true },
      { name: "BBL", y: bbL, color: "#999", width: 1.0, dash: true },
    ],
  });

  const rDD = await measure(page, "cDD");
  drawLineChart(wx.createCanvasContext("cDD", page), {
    width: Math.floor(rDD.width || 320),
    height: Math.floor(rDD.height || 180),
    x: dates,
    yMode: "linear",
    title: "Drawdown",
    series: [{ name: "DD", y: dd, color: "#e53935", width: 1.4 }],
  });

  const rRSI = await measure(page, "cRSI");
  drawLineChart(wx.createCanvasContext("cRSI", page), {
    width: Math.floor(rRSI.width || 320),
    height: Math.floor(rRSI.height || 160),
    x: dates,
    yMode: "linear",
    yFixed: [0, 100],
    title: "RSI24",
    series: [{ name: "RSI24", y: rsi, color: "#7b61ff", width: 1.4 }],
  });

  const rRR = await measure(page, "cRR3Y");
  drawLineChart(wx.createCanvasContext("cRR3Y", page), {
    width: Math.floor(rRR.width || 320),
    height: Math.floor(rRR.height || 160),
    x: dates,
    yMode: "linear",
    title: "Rolling 3Y Return",
    series: [{ name: "R3Y", y: rr3y, color: "#2e7d32", width: 1.4 }],
  });

  const rRDD = await measure(page, "cRDD3Y");
  drawLineChart(wx.createCanvasContext("cRDD3Y", page), {
    width: Math.floor(rRDD.width || 320),
    height: Math.floor(rRDD.height || 160),
    x: dates,
    yMode: "linear",
    title: "Rolling 3Y MDD",
    series: [{ name: "DD3Y", y: rdd3y, color: "#ad1457", width: 1.4 }],
  });

  const corr = payload.correlation || {};
  const rCorr = await measure(page, "cCorr");
  drawHeatmap4(wx.createCanvasContext("cCorr", page), {
    width: Math.floor(rCorr.width || 320),
    height: Math.floor(rCorr.height || 220),
    title: "Correlation",
    labels: corr.codes || [],
    matrix: corr.matrix || [],
  });

  // return contribution (bar)
  const attr = payload.attribution || {};
  const rcRows = ((attr.return || {}).by_code || []).slice();
  const rcLabels = rcRows.map((x) => x.code);
  const rcVals = rcRows.map((x) => (x.return_share == null ? NaN : Number(x.return_share)));
  const rRC = await measure(page, "cRC");
  drawBarChart(wx.createCanvasContext("cRC", page), {
    width: Math.floor(rRC.width || 320),
    height: Math.floor(rRC.height || 180),
    title: "Return Share",
    labels: rcLabels,
    values: rcVals,
    colors: ["#1677ff", "#00bfa5", "#ff7a00", "#7b61ff"],
    valueFmt: (x) => (x * 100).toFixed(1) + "%",
    vmin: 0,
    vmax: 1,
  });

  // risk contribution (bar)
  const rkRows = ((attr.risk || {}).by_code || []).slice();
  const rkLabels = rkRows.map((x) => x.code);
  const rkVals = rkRows.map((x) => (x.risk_share == null ? NaN : Number(x.risk_share)));
  const rRisk = await measure(page, "cRiskC");
  drawBarChart(wx.createCanvasContext("cRiskC", page), {
    width: Math.floor(rRisk.width || 320),
    height: Math.floor(rRisk.height || 180),
    title: "Risk Share",
    labels: rkLabels,
    values: rkVals,
    colors: ["#e53935", "#d81b60", "#8e24aa", "#3949ab"],
    valueFmt: (x) => (x * 100).toFixed(1) + "%",
    vmin: 0,
    vmax: 1,
  });
}

function attachWeekdayPage({ anchor, title }) {
  return {
    data: {
      anchor,
      title,
      status: "",
      rangeKey: "all",
      rangeLabel: "全区间",
      raw: null,
      m: {},
      attrReturn: [],
      attrRisk: [],
      calMode: "daily",
      calModeLabel: "日度",
      calDailyMonth: "",
      calDailyCells: [],
      calDailyMonths: [],
      calDailyMap: {},
      calMonthlyYears: [],
      calYearlyCells: [],
    },

    onShow() {
      // default load
      if (!this.data.raw) this.loadRange("all");
    },

    async setRange(e) {
      const k = e.currentTarget.dataset.k;
      await this.loadRange(k);
    },

    async loadRange(k) {
      try {
        this.setData({ status: "加载中...", rangeKey: k });

        // First time: fetch a wide window (10y) to support all range buttons.
        // Subsequent range picks: slice by dates and re-fetch for accurate metrics/attribution/correlation.
        const today = new Date();
        const end0 = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
        const start0 = `${today.getFullYear() - 12}0101`;
        const base = await request("/analysis/baseline/weekly5-ew-dashboard", {
          method: "POST",
          data: { start: start0, end: end0, rebalance_shift: "prev" },
        });

        const full = (base.by_anchor || {})[String(anchor)];
        const fullDates = full.dates || [];
        const picked = pickRangeDates(fullDates, k);
        const rangeLabel = picked.label;

        let data = full;
        if (picked.start && picked.end && k !== "all") {
          const res = await request("/analysis/baseline/weekly5-ew-dashboard", {
            method: "POST",
            data: { start: ymd(picked.start), end: ymd(picked.end), rebalance_shift: "prev" },
          });
          data = (res.by_anchor || {})[String(anchor)];
        }

        const m = data.metrics || {};
        const cal = data.calendar || {};
        const dm = _buildDailyMonthState(cal.daily || {});
        const dailyCells = _buildMonthCells(dm.month, dm.map);
        const monthlyYears = _buildMonthlyYearCells(cal.monthly || {});
        const yearlyCells = _buildYearlyCells(cal.yearly || {});
        const mode = this.data.calMode || "daily";
        const modeLabel = mode === "monthly" ? "月度" : (mode === "yearly" ? "年度" : "日度");

        this.setData({
          raw: data,
          rangeLabel,
          status: "OK",
          m: {
            cumulative_return: pct(m.cumulative_return),
            annualized_return: pct(m.annualized_return),
            annualized_volatility: pct(m.annualized_volatility),
            max_drawdown: pct(m.max_drawdown),
            max_drawdown_recovery_days: m.max_drawdown_recovery_days,
            sharpe_ratio: num(m.sharpe_ratio),
            calmar_ratio: num(m.calmar_ratio),
            sortino_ratio: num(m.sortino_ratio),
            ulcer_index: num(m.ulcer_index),
            ulcer_performance_index: num(m.ulcer_performance_index),
          },
          attrReturn: [],
          attrRisk: [],
          calMode: mode,
          calModeLabel: modeLabel,
          calDailyMonth: dm.month,
          calDailyCells: dailyCells,
          calDailyMonths: dm.months,
          calDailyMap: dm.map,
          calMonthlyYears: monthlyYears,
          calYearlyCells: yearlyCells,
        });

        await drawAll(this, data);
      } catch (e) {
        console.error(e);
        this.setData({ status: e.message || "加载失败" });
      }
    },

    setCalMode(e) {
      const m = e.currentTarget.dataset.m;
      const mode = (m === "monthly" || m === "yearly" || m === "daily") ? m : "daily";
      const modeLabel = mode === "monthly" ? "月度" : (mode === "yearly" ? "年度" : "日度");
      this.setData({ calMode: mode, calModeLabel: modeLabel });
    },

    prevCalMonth() {
      const months = this.data.calDailyMonths || [];
      const cur = this.data.calDailyMonth;
      if (!months.length || !cur) return;
      const i = months.indexOf(cur);
      const j = Math.max(0, i - 1);
      const m = months[j];
      const cells = _buildMonthCells(m, this.data.calDailyMap || {});
      this.setData({ calDailyMonth: m, calDailyCells: cells });
    },

    nextCalMonth() {
      const months = this.data.calDailyMonths || [];
      const cur = this.data.calDailyMonth;
      if (!months.length || !cur) return;
      const i = months.indexOf(cur);
      const j = Math.min(months.length - 1, i + 1);
      const m = months[j];
      const cells = _buildMonthCells(m, this.data.calDailyMap || {});
      this.setData({ calDailyMonth: m, calDailyCells: cells });
    },
  };
}

module.exports = { attachWeekdayPage };

