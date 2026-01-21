const { request } = require("../utils/api");
const { drawLineChart, drawHeatmap4, drawBarChart, lineIndexFromTouchX } = require("../utils/charts");

// Display names (avoid showing codes in UI)
const CODE_NAME = {
  "159915": "创业板",
  "511010": "国债",
  "513100": "纳指",
  "518880": "黄金",
};
const CODE_ORDER = ["159915", "511010", "513100", "518880"];
const IDX_LABELS = ["1", "2", "3", "4"];
const IDX_MAP_LINES = [
  "1=创业板(159915)",
  "2=国债(511010)",
  "3=纳指(513100)",
  "4=黄金(518880)",
];

const THEME_MODE_KEY = "theme_mode_v1"; // "light" | "dark"

function _safeGetStorage(key, fallback) {
  try {
    const v = wx.getStorageSync(key);
    return v == null || v === "" ? fallback : v;
  } catch (e) {
    return fallback;
  }
}

function _safeSetStorage(key, value) {
  try {
    wx.setStorageSync(key, value);
  } catch (e) {
    // ignore
  }
}

function dispCode(code) {
  const c = String(code || "");
  return CODE_NAME[c] || c;
}

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

function _drawdownFromNav(nav) {
  let peak = null;
  const out = [];
  for (const v of nav || []) {
    const x = Number(v);
    if (!Number.isFinite(x)) { out.push(null); continue; }
    if (peak == null || x > peak) peak = x;
    out.push(peak > 0 ? (x / peak - 1) : null);
  }
  return out;
}

function _emaArr(arr, span) {
  const n = (arr || []).length;
  const s = Math.max(2, Number(span || 2));
  const alpha = 2 / (s + 1);
  const out = new Array(n).fill(null);
  let prev = null;
  for (let i = 0; i < n; i++) {
    const x = Number(arr[i]);
    if (!Number.isFinite(x)) { out[i] = null; continue; }
    if (prev == null) { prev = x; out[i] = x; continue; }
    prev = alpha * x + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

function _rollingStdArr(arr, window) {
  const n = (arr || []).length;
  const w = Math.max(2, Number(window || 2));
  const out = new Array(n).fill(null);
  for (let i = w - 1; i < n; i++) {
    let xs = [];
    for (let j = i - w + 1; j <= i; j++) {
      const x = Number(arr[j]);
      if (Number.isFinite(x)) xs.push(x);
    }
    if (xs.length < w) { out[i] = null; continue; }
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const var1 = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, xs.length - 1);
    out[i] = Math.sqrt(var1);
  }
  return out;
}

function _rollingReturn(nav, windowDays) {
  const n = (nav || []).length;
  const w = Math.max(1, Number(windowDays || 1));
  const out = new Array(n).fill(null);
  for (let i = w; i < n; i++) {
    const a = Number(nav[i - w]);
    const b = Number(nav[i]);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0) continue;
    out[i] = b / a - 1;
  }
  return out;
}

function _rollingMaxDrawdown(nav, windowDays) {
  const n = (nav || []).length;
  const w = Math.max(2, Number(windowDays || 2));
  const out = new Array(n).fill(null);
  for (let i = w - 1; i < n; i++) {
    let peak = -Infinity;
    let mdd = 0;
    for (let j = i - w + 1; j <= i; j++) {
      const x = Number(nav[j]);
      if (!Number.isFinite(x)) continue;
      if (x > peak) peak = x;
      if (peak > 0) {
        const dd = x / peak - 1;
        if (dd < mdd) mdd = dd;
      }
    }
    out[i] = mdd;
  }
  return out;
}

function _rollingDrawdown(nav, windowDays) {
  const n = (nav || []).length;
  const w = Math.max(2, Number(windowDays || 2));
  const out = new Array(n).fill(null);
  for (let i = w - 1; i < n; i++) {
    let peak = -Infinity;
    for (let j = i - w + 1; j <= i; j++) {
      const x = Number(nav[j]);
      if (!Number.isFinite(x) || x <= 0) { peak = NaN; break; }
      if (x > peak) peak = x;
    }
    const cur = Number(nav[i]);
    if (!Number.isFinite(cur) || !Number.isFinite(peak) || peak <= 0) { out[i] = null; continue; }
    out[i] = cur / peak - 1;
  }
  return out;
}

function lineXFromIndex(idx, width, n) {
  const padL = 46, padR = 16;
  const W = Number(width || 0);
  const N = Number(n || 0);
  if (!Number.isFinite(W) || !Number.isFinite(N) || N <= 1) return padL;
  const plotW = W - padL - padR;
  if (plotW <= 1) return padL;
  return padL + (Number(idx) / (N - 1)) * plotW;
}

function _rsiWilder(arr, window) {
  const n = (arr || []).length;
  const w = Math.max(2, Number(window || 14));
  const out = new Array(n).fill(null);
  let avgGain = null;
  let avgLoss = null;
  for (let i = 1; i < n; i++) {
    const p0 = Number(arr[i - 1]);
    const p1 = Number(arr[i]);
    if (!Number.isFinite(p0) || !Number.isFinite(p1)) continue;
    const diff = p1 - p0;
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    if (avgGain == null || avgLoss == null) {
      // initialize after first w samples
      if (i < w) continue;
      let g = 0, l = 0, cnt = 0;
      for (let k = i - w + 1; k <= i; k++) {
        const a = Number(arr[k - 1]);
        const b = Number(arr[k]);
        if (!Number.isFinite(a) || !Number.isFinite(b)) { cnt = 0; break; }
        const d = b - a;
        g += d > 0 ? d : 0;
        l += d < 0 ? -d : 0;
        cnt++;
      }
      if (cnt !== w) continue;
      avgGain = g / w;
      avgLoss = l / w;
    } else {
      // Wilder smoothing
      avgGain = (avgGain * (w - 1) + gain) / w;
      avgLoss = (avgLoss * (w - 1) + loss) / w;
    }
    if (avgLoss === 0) { out[i] = 100; continue; }
    const rs = avgGain / avgLoss;
    out[i] = 100 - (100 / (1 + rs));
  }
  return out;
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
  const rdd3y = payload.roll3y_dd || payload.roll3y_mdd || [];
  const theme = (page && page.data && page.data.theme) ? page.data.theme : "light";

  const rNav = await measure(page, "cNav");
  // store rects for touch interaction
  page.__rects = page.__rects || {};
  page.__rects.cNav = rNav;
  drawLineChart(wx.createCanvasContext("cNav", page), {
    width: Math.max(10, Math.floor(rNav.width || 320)),
    height: Math.max(10, Math.floor(rNav.height || 200)),
    x: dates,
    yMode: "linear",
    title: "EW 净值",
    theme,
    series: [
      { name: "NAV", y: nav, color: "#1677ff", width: 1.6 },
      { name: "EMA252", y: ema252, color: "#ff7a00", width: 1.2, dash: true },
      { name: "BBU", y: bbU, color: "#999", width: 1.0, dash: true },
      { name: "BBL", y: bbL, color: "#999", width: 1.0, dash: true },
    ],
  });

  const rDD = await measure(page, "cDD");
  page.__rects.cDD = rDD;
  drawLineChart(wx.createCanvasContext("cDD", page), {
    width: Math.floor(rDD.width || 320),
    height: Math.floor(rDD.height || 180),
    x: dates,
    yMode: "linear",
    title: "Drawdown",
    yLabelFmt: (v) => pct(v),
    theme,
    series: [{ name: "DD", y: dd, color: "#e53935", width: 1.4 }],
  });

  const rRSI = await measure(page, "cRSI");
  page.__rects.cRSI = rRSI;
  drawLineChart(wx.createCanvasContext("cRSI", page), {
    width: Math.floor(rRSI.width || 320),
    height: Math.floor(rRSI.height || 160),
    x: dates,
    yMode: "linear",
    yFixed: [0, 100],
    title: "RSI24",
    theme,
    series: [{ name: "RSI24", y: rsi, color: "#7b61ff", width: 1.4 }],
  });

  const rRR = await measure(page, "cRR3Y");
  page.__rects.cRR3Y = rRR;
  drawLineChart(wx.createCanvasContext("cRR3Y", page), {
    width: Math.floor(rRR.width || 320),
    height: Math.floor(rRR.height || 160),
    x: dates,
    yMode: "linear",
    title: "Rolling 3Y Return",
    yLabelFmt: (v) => pct(v),
    theme,
    series: [{ name: "R3Y", y: rr3y, color: "#2e7d32", width: 1.4 }],
  });

  const rRDD = await measure(page, "cRDD3Y");
  page.__rects.cRDD3Y = rRDD;
  drawLineChart(wx.createCanvasContext("cRDD3Y", page), {
    width: Math.floor(rRDD.width || 320),
    height: Math.floor(rRDD.height || 160),
    x: dates,
    yMode: "linear",
    title: "Rolling 3Y DD",
    yLabelFmt: (v) => pct(v),
    theme,
    series: [{ name: "DD3Y", y: rdd3y, color: "#ad1457", width: 1.4 }],
  });

  const corr = payload.correlation || {};
  const rCorr = await measure(page, "cCorr");
  page.__rects.cCorr = rCorr;
  drawHeatmap4(wx.createCanvasContext("cCorr", page), {
    width: Math.floor(rCorr.width || 320),
    height: Math.floor(rCorr.height || 220),
    title: "Correlation",
    theme,
    labels: IDX_LABELS,
    matrix: corr.matrix || [],
  });

  // return contribution (bar)
  const attr = payload.attribution || {};
  const rcRows = ((attr.return || {}).by_code || []).slice();
  const rcLabels = rcRows.map((x) => dispCode(x.code));
  const rcVals = rcRows.map((x) => (x.return_share == null ? NaN : Number(x.return_share)));
  const rRC = await measure(page, "cRC");
  drawBarChart(wx.createCanvasContext("cRC", page), {
    width: Math.floor(rRC.width || 320),
    height: Math.floor(rRC.height || 180),
    title: "Return Share",
    theme,
    labels: rcLabels,
    values: rcVals,
    colors: ["#1677ff", "#00bfa5", "#ff7a00", "#7b61ff"],
    valueFmt: (x) => (x * 100).toFixed(1) + "%",
    vmin: 0,
    vmax: 1,
  });

  // risk contribution (bar)
  const rkRows = ((attr.risk || {}).by_code || []).slice();
  const rkLabels = rkRows.map((x) => dispCode(x.code));
  const rkVals = rkRows.map((x) => (x.risk_share == null ? NaN : Number(x.risk_share)));
  const rRisk = await measure(page, "cRiskC");
  drawBarChart(wx.createCanvasContext("cRiskC", page), {
    width: Math.floor(rRisk.width || 320),
    height: Math.floor(rRisk.height || 180),
    title: "Risk Share",
    theme,
    labels: rkLabels,
    values: rkVals,
    colors: ["#e53935", "#d81b60", "#8e24aa", "#3949ab"],
    valueFmt: (x) => (x * 100).toFixed(1) + "%",
    vmin: 0,
    vmax: 1,
  });
}

function attachWeekdayPage({ anchor, title }) {
  const pageKey = (String(anchor) === "mix") ? "mix" : `wd${Number(anchor)}`;
  return {
    data: {
      anchor,
      title,
      pageKey,
      status: "",
      themeMode: "light",
      theme: "light",
      themeClass: "theme-light",
      rangeKey: "all",
      rangeLabel: "全区间",
      raw: null, // kept for backward-compat in data shape; large payloads are stored on page.__raw
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
      calMonthlyYearIdx: 0,
      calMonthlyYear: "",
      calMonthlyCells: [],
      calYearlyCells: [],
      calYearlyAll: [],
      calYearlyPage: 1,
      calYearlyPageSize: 12,
      corrMapLines: IDX_MAP_LINES,
      // tip: tooltip + vertical guide line (all px within canvasWrap)
      tip: { show: false, id: "", x: 0, y: 0, text: "", cx: 0, cw: 0, ch: 0 },

      // rotation section
      rot: null, // kept for backward-compat in data shape; large payloads are stored on page.__rot
      rm: {},
      rotCalMode: "daily",
      rotCalModeLabel: "日度",
      rotCalDailyMonth: "",
      rotCalDailyCells: [],
      rotCalDailyMonths: [],
      rotCalDailyMap: {},
      rotCalMonthlyYears: [],
      rotCalMonthlyYearIdx: 0,
      rotCalMonthlyYear: "",
      rotCalMonthlyCells: [],
      rotCalYearlyAll: [],
      rotCalYearlyCells: [],
      rotCalYearlyPage: 1,
      rotCalYearlyPageSize: 12,

      // period comparison
      rotPeriods: [],
      rotPeriodPage: 1,
      rotPeriodPageSize: 12,
      rotPeriodPageText: "",
      rotPeriodsView: [],

      // next rebalance plan
      nextPlan: null,
    },

    onLoad() {
      // init theme from cache
      this.applyTheme();
    },

    onShow() {
      // keep in sync with cached mode
      this.applyTheme();
      // default load
      if (!this.__raw) this.loadRange(this.data.rangeKey || "all");
    },

    onUnload() {
      // Prevent background async tasks from calling setData/draw after page is destroyed.
      this.__unloaded = true;
      this.__loadToken = (this.__loadToken || 0) + 1; // invalidate in-flight tasks
      if (this.__tipTimer) {
        clearTimeout(this.__tipTimer);
        this.__tipTimer = null;
      }
    },

    applyTheme(nextMode) {
      const modeRaw = String(nextMode || this.data.themeMode || _safeGetStorage(THEME_MODE_KEY, "light"));
      const mode = (modeRaw === "dark") ? "dark" : "light";
      const themeClass = mode === "dark" ? "theme-dark" : "theme-light";

      const changed = (this.data.theme !== mode) || (this.data.themeMode !== mode);
      this.setData({ themeMode: mode, theme: mode, themeClass });
      _safeSetStorage(THEME_MODE_KEY, mode);

      // Nav bar should match the theme for readability.
      try {
        wx.setNavigationBarColor({
          backgroundColor: mode === "dark" ? "#0f1115" : "#ffffff",
          frontColor: mode === "dark" ? "#ffffff" : "#000000",
        });
      } catch (e) {
        // ignore older base libs
      }

      // Redraw charts with new palette.
      if (changed) {
        try {
          if (this.__raw) drawAll(this, this.__raw);
          if (this.__rot) this._drawRotation(this.__rot);
        } catch (e3) {
          // best effort
        }
      }
    },

    setThemeMode(e) {
      const m = (e && e.currentTarget && e.currentTarget.dataset) ? String(e.currentTarget.dataset.m || "") : "light";
      this.applyTheme(m);
    },

    toggleTheme() {
      const cur = String(this.data.themeMode || "light");
      const next = (cur === "dark") ? "light" : "dark";
      this.applyTheme(next);
    },

    async setRange(e) {
      const k = e.currentTarget.dataset.k;
      await this.loadRange(k);
    },

    switchPage(e) {
      const p = (e && e.currentTarget && e.currentTarget.dataset) ? String(e.currentTarget.dataset.p || "") : "";
      if (!p || p === String(this.data.pageKey || "")) return;
      // No tabBar now (we use a custom bottom nav); use redirectTo to avoid stacking history.
      if (p === "mix") return wx.redirectTo({ url: "/pages/mix/index" });
      if (p === "wd0") return wx.redirectTo({ url: "/pages/wd0/index" });
      if (p === "wd1") return wx.redirectTo({ url: "/pages/wd1/index" });
      if (p === "wd2") return wx.redirectTo({ url: "/pages/wd2/index" });
      if (p === "wd3") return wx.redirectTo({ url: "/pages/wd3/index" });
      if (p === "wd4") return wx.redirectTo({ url: "/pages/wd4/index" });
    },

    async loadRange(k) {
      try {
        this.__unloaded = false;
        const token = (this.__loadToken = (this.__loadToken || 0) + 1);
        const isMix = String(anchor) === "mix";
        this.setData({ status: "加载中...", rangeKey: k });
        // bump version when backend semantics/UI change to avoid stale cached rotations/period tables
        const cacheKey = `wd_dash_v3_${String(anchor)}_${String(k)}`;
        try {
          const cached = wx.getStorageSync(cacheKey);
          if (cached && cached.base && cached.end0) {
            // quick render from cache for better UX
            this.__raw = cached.base;
            this.__rot = cached.rot || null;
            this.__fullBase = cached.fullBase || null;
            this.__fullRot = cached.fullRot || null;
            this.setData({ status: "使用缓存加载中..." });
          }
        } catch (e0) {
          // ignore cache errors
        }

        // First time: fetch a wide window (10y) to support all range buttons.
        // Subsequent range picks: slice by dates and re-fetch for accurate metrics/attribution/correlation.
        const today = new Date();
        const end0 = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
        // Fetch a sufficiently early start so backend can derive the true max-common-history range
        // (e.g. 20130729 for the fixed 4-ETF universe), rather than being truncated by request start.
        const start0 = "20000101";
        const baseLitePath = isMix ? "/analysis/baseline/weekly5-ew-dashboard-combo-lite" : "/analysis/baseline/weekly5-ew-dashboard-lite";
        const baseFullPath = isMix ? "/analysis/baseline/weekly5-ew-dashboard-combo" : "/analysis/baseline/weekly5-ew-dashboard";
        const rotLitePath = isMix ? "/analysis/rotation/weekly5-open-combo-lite" : "/analysis/rotation/weekly5-open-lite";
        const rotFullPath = isMix ? "/analysis/rotation/weekly5-open-combo" : "/analysis/rotation/weekly5-open";

        // Stage 1 (lite): fetch full-range once (per page instance) to get the trading-date axis fast.
        let baseLite = this.__fullBaseLite;
        if (!baseLite || !baseLite.by_anchor || !(baseLite.by_anchor[String(anchor)]) || !baseLite.meta || baseLite.meta.end !== end0 || baseLite.meta.start !== start0) {
          baseLite = await request(baseLitePath, {
            method: "POST",
            data: isMix
              ? { start: start0, end: end0, rebalance_shift: "prev" }
              : { start: start0, end: end0, rebalance_shift: "prev", anchor_weekday: Number(anchor) },
          });
          if (this.__unloaded || token !== this.__loadToken) return;
          this.__fullBaseLite = baseLite;
        }

        const full = (baseLite.by_anchor || {})[String(anchor)];
        const fullDates = full.dates || [];
        const picked = pickRangeDates(fullDates, k);
        const rangeLabel = picked.label;

        // Stage 1 (lite): render charts ASAP
        let dataLite = full;
        if (picked.start && picked.end && k !== "all") {
          const resLite = await request(baseLitePath, {
            method: "POST",
            data: isMix
              ? { start: ymd(picked.start), end: ymd(picked.end), rebalance_shift: "prev" }
              : { start: ymd(picked.start), end: ymd(picked.end), rebalance_shift: "prev", anchor_weekday: Number(anchor) },
          });
          if (this.__unloaded || token !== this.__loadToken) return;
          dataLite = (resLite.by_anchor || {})[String(anchor)];
        }

        // rotation (fixed strategy, open execution) — use the same range window
        let rotFull = null;
        try {
          // Stage 1 (lite): nav-only, faster & smaller
          let rotBaseLite = this.__fullRotLite;
          if (!rotBaseLite || !rotBaseLite.by_anchor || !(rotBaseLite.by_anchor[String(anchor)]) || !rotBaseLite.meta || rotBaseLite.meta.end !== end0 || rotBaseLite.meta.start !== start0) {
            rotBaseLite = await request(rotLitePath, { method: "POST", data: isMix ? { start: start0, end: end0 } : { start: start0, end: end0, anchor_weekday: Number(anchor) } });
            if (this.__unloaded || token !== this.__loadToken) return;
            this.__fullRotLite = rotBaseLite;
          }
          rotFull = (rotBaseLite.by_anchor || {})[String(anchor)];
          if (picked.start && picked.end && k !== "all") {
            const rotRes = await request(rotLitePath, { method: "POST", data: isMix ? { start: ymd(picked.start), end: ymd(picked.end) } : { start: ymd(picked.start), end: ymd(picked.end), anchor_weekday: Number(anchor) } });
            if (this.__unloaded || token !== this.__loadToken) return;
            rotFull = (rotRes.by_anchor || {})[String(anchor)];
          }
        } catch (e2) {
          rotFull = null;
        }

        // For stage-1 lite payload, metrics/calendar may be absent.
        const m = (dataLite && dataLite.metrics) ? dataLite.metrics : {};
        const cal = (dataLite && dataLite.calendar) ? dataLite.calendar : {};
        const dm = _buildDailyMonthState(cal.daily || {});
        const dailyCells = dm.month ? _buildMonthCells(dm.month, dm.map) : [];
        const monthlyYears = _buildMonthlyYearCells(cal.monthly || {});
        const yearlyAll = _buildYearlyCells(cal.yearly || {});
        const yearPageSize = 12;
        const yearPage = 1;
        const yearlyCells = yearlyAll.slice(Math.max(0, yearlyAll.length - yearPageSize), yearlyAll.length);

        const myIdx = monthlyYears.length ? (monthlyYears.length - 1) : 0;
        const my = monthlyYears[myIdx] || { year: "", cells: [] };
        const mode = this.data.calMode || "daily";
        const modeLabel = mode === "monthly" ? "月度" : (mode === "yearly" ? "年度" : "日度");

        // Keep large series off setData (setData is expensive with big arrays).
        this.__raw = dataLite;
        this.__rot = rotFull;

        this.setData({
          rangeLabel,
          status: "图表已加载，指标计算中...",
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
          calMonthlyYearIdx: myIdx,
          calMonthlyYear: my.year,
          calMonthlyCells: my.cells,
          calYearlyCells: yearlyCells,
          calYearlyAll: yearlyAll,
          calYearlyPage: yearPage,
          calYearlyPageSize: yearPageSize,
        });

        if (this.__unloaded || token !== this.__loadToken) return;
        await drawAll(this, dataLite);
        if (this.__unloaded || token !== this.__loadToken) return;
        await this._drawRotation(rotFull);

        // Stage 2 (full): fetch metrics/attribution/correlation/calendar in background and then redraw.
        (async () => {
          try {
            const s1 = (picked.start && picked.end && k !== "all") ? ymd(picked.start) : start0;
            const e1 = (picked.start && picked.end && k !== "all") ? ymd(picked.end) : end0;

            const fullRes = await request(baseFullPath, {
              method: "POST",
              data: isMix ? { start: s1, end: e1, rebalance_shift: "prev" } : { start: s1, end: e1, rebalance_shift: "prev", anchor_weekday: Number(anchor) },
            });
            if (this.__unloaded || token !== this.__loadToken) return;
            const dataFull = (fullRes.by_anchor || {})[String(anchor)];
            if (dataFull) {
              this.__raw = dataFull;
              const mm = dataFull.metrics || {};
              const cc = dataFull.calendar || {};
              const dm2 = _buildDailyMonthState(cc.daily || {});
              const dailyCells2 = dm2.month ? _buildMonthCells(dm2.month, dm2.map) : [];
              const monthlyYears2 = _buildMonthlyYearCells(cc.monthly || {});
              const yearlyAll2 = _buildYearlyCells(cc.yearly || {});
              const myIdx2 = monthlyYears2.length ? (monthlyYears2.length - 1) : 0;
              const my2 = monthlyYears2[myIdx2] || { year: "", cells: [] };
              const yearPageSize2 = 12;
              const yearPage2 = 1;
              const yearlyCells2 = yearlyAll2.slice(Math.max(0, yearlyAll2.length - yearPageSize2), yearlyAll2.length);

              this.setData({
                status: "OK",
                m: {
                  cumulative_return: pct(mm.cumulative_return),
                  annualized_return: pct(mm.annualized_return),
                  annualized_volatility: pct(mm.annualized_volatility),
                  max_drawdown: pct(mm.max_drawdown),
                  max_drawdown_recovery_days: mm.max_drawdown_recovery_days,
                  sharpe_ratio: num(mm.sharpe_ratio),
                  calmar_ratio: num(mm.calmar_ratio),
                  sortino_ratio: num(mm.sortino_ratio),
                  ulcer_index: num(mm.ulcer_index),
                  ulcer_performance_index: num(mm.ulcer_performance_index),
                },
                calDailyMonth: dm2.month,
                calDailyCells: dailyCells2,
                calDailyMonths: dm2.months,
                calDailyMap: dm2.map,
                calMonthlyYears: monthlyYears2,
                calMonthlyYearIdx: myIdx2,
                calMonthlyYear: my2.year,
                calMonthlyCells: my2.cells,
                calYearlyCells: yearlyCells2,
                calYearlyAll: yearlyAll2,
                calYearlyPage: yearPage2,
                calYearlyPageSize: yearPageSize2,
              });
              if (this.__unloaded || token !== this.__loadToken) return;
              await drawAll(this, dataFull);
            }

            const rotFullRes = await request(rotFullPath, { method: "POST", data: isMix ? { start: s1, end: e1 } : { start: s1, end: e1, anchor_weekday: Number(anchor) } });
            if (this.__unloaded || token !== this.__loadToken) return;
            const rotDataFull = (rotFullRes.by_anchor || {})[String(anchor)];
            if (rotDataFull) {
              this.__rot = rotDataFull;
              if (this.__unloaded || token !== this.__loadToken) return;
              await this._drawRotation(rotDataFull);
            }

            // persist cache for instant next open (store full payloads)
            try {
              wx.setStorageSync(cacheKey, { end0, base: this.__raw, rot: this.__rot });
            } catch (e4) {
              // ignore cache quota errors
            }
          } catch (eFull) {
            // keep stage-1 result; best-effort
          }
        })();

        // next rebalance plan (real-time): use "today" as asof.
        // Backend will map asof -> last available close <= asof, so:
        // - before close: shows today's execution plan (decision from previous close)
        // - after close: shows next trading day's plan (decision from today's close)
        try {
          const plan = isMix
            ? await request("/analysis/rotation/next-plan-auto", { method: "POST", data: { asof: end0 } })
            : await request("/analysis/rotation/next-plan", { method: "POST", data: { anchor_weekday: Number(anchor), asof: end0 } });
          if (this.__unloaded || token !== this.__loadToken) return;
          this.setData({ nextPlan: plan || null });
        } catch (e3) {
          if (this.__unloaded || token !== this.__loadToken) return;
          this.setData({ nextPlan: null });
        }
      } catch (e) {
        console.error(e);
        this.setData({ status: e.message || "加载失败" });
      }
    },

    async _drawRotation(rotFull) {
      if (!rotFull || !rotFull.nav || !rotFull.nav.dates) {
        this.setData({ rm: {}, rotPeriods: [], rotPeriodPageText: "", nextPlan: this.data.nextPlan });
        return;
      }
      const dates = (rotFull.nav.dates || []);
      const s = (rotFull.nav.series || {});
      const navRot = s.ROTATION || [];
      const navEw = s.EW_REBAL || [];
      const navEx = s.EXCESS || [];

      const ddRot = _drawdownFromNav(navRot);
      const ddEw = _drawdownFromNav(navEw);
      const ddEx = _drawdownFromNav((navEx && navEx.length) ? navEx : (navRot.map((v, i) => {
        const a = Number(v);
        const b = Number(navEw[i]);
        if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
        return a / b;
      })));
      const rsiRot24 = (((rotFull.nav_rsi || {}).series || {}).ROTATION || {})["24"] || _rsiWilder(navRot, 24);
      const rr3y = _rollingReturn(navRot, 3 * 252);
      const rdd3y = _rollingDrawdown(navRot, 3 * 252);

      // ratio curve ROT / EW
      const ratio = navRot.map((v, i) => {
        const a = Number(v);
        const b = Number(navEw[i]);
        if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
        return a / b;
      });
      const ratioEma = _emaArr(ratio, 252);
      const ratioSd = _rollingStdArr(ratio, 252);
      const ratioBbu = ratioEma.map((v, i) => (v != null && ratioSd[i] != null) ? (Number(v) + 2 * Number(ratioSd[i])) : null);
      const ratioBbl = ratioEma.map((v, i) => (v != null && ratioSd[i] != null) ? (Number(v) - 2 * Number(ratioSd[i])) : null);
      const ratioRsi24 = _rsiWilder(ratio.map((x) => (x == null ? NaN : x)), 24);

      // 40d rolling return diff: ROT - EW
      const rot40 = _rollingReturn(navRot, 40);
      const ew40 = _rollingReturn(navEw, 40);
      const diff40 = rot40.map((v, i) => (v == null || ew40[i] == null) ? null : (Number(v) - Number(ew40[i])));

      // cache derived series for smooth hover interaction (avoid O(n*window) recomputation on touchmove)
      this.__rotCache = {
        dates,
        navRot,
        navEw,
        navEx,
        ddRot,
        ddEw,
        ddEx,
        rr3y,
        dd3y: rdd3y,
        ratio,
        ratioEma,
        ratioBbu,
        ratioBbl,
        ratioRsi24,
        diff40,
      };

      // metrics text
      const ms = (rotFull.metrics || {}).strategy || {};
      const mex = (rotFull.metrics || {}).excess_vs_equal_weight || {};
      const wp = rotFull.win_payoff || {};
      const annualTurnover = (ms.avg_daily_turnover == null) ? NaN : (Number(ms.avg_daily_turnover) * 252);
      const rm = {
        cumulative_return: pct(ms.cumulative_return),
        annualized_return: pct(ms.annualized_return),
        annualized_volatility: pct(ms.annualized_volatility),
        max_drawdown: pct(ms.max_drawdown),
        max_drawdown_recovery_days: (ms.max_drawdown_recovery_days == null ? "-" : String(ms.max_drawdown_recovery_days)),
        sharpe_ratio: num(ms.sharpe_ratio),
        sortino_ratio: num(ms.sortino_ratio),
        calmar_ratio: num(ms.calmar_ratio),
        ulcer_index: num(ms.ulcer_index),
        ulcer_performance_index: num(ms.ulcer_performance_index),
        avg_daily_turnover: pct(ms.avg_daily_turnover),
        annual_turnover: pct(annualTurnover),
        excess_cum: pct(mex.cumulative_return),
        excess_ann: pct(mex.annualized_return),
        excess_volatility: pct(mex.annualized_volatility),
        excess_ir: num(mex.information_ratio),
        excess_max_drawdown: pct(mex.max_drawdown),
        excess_max_drawdown_recovery_days: (mex.max_drawdown_recovery_days == null ? "-" : String(mex.max_drawdown_recovery_days)),
        excess_win_rate: pct(wp.win_rate),
        excess_avg_win: pct(wp.avg_win_excess),
        excess_avg_loss: pct(wp.avg_loss_excess),
      };

      // calendar (strategy daily/monthly/yearly returns)
      const daily = [];
      for (let i = 0; i < dates.length; i++) {
        if (i === 0) { daily.push(0); continue; }
        const a = Number(navRot[i - 1]);
        const b = Number(navRot[i]);
        daily.push((Number.isFinite(a) && Number.isFinite(b) && a > 0) ? (b / a - 1) : null);
      }
      // monthly/yearly returns from last nav per month/year
      const monthLast = {}; // "YYYY-MM" -> {i, nav}
      const yearLast = {}; // "YYYY" -> {i, nav}
      for (let i = 0; i < dates.length; i++) {
        const ds = String(dates[i]);
        const ym = ds.slice(0, 7);
        const y = ds.slice(0, 4);
        monthLast[ym] = { i, nav: Number(navRot[i]) };
        yearLast[y] = { i, nav: Number(navRot[i]) };
      }
      const monthKeys = Object.keys(monthLast).sort();
      const monDates = [];
      const monVals = [];
      for (let k = 0; k < monthKeys.length; k++) {
        const ym = monthKeys[k];
        const cur = monthLast[ym];
        const prev = k > 0 ? monthLast[monthKeys[k - 1]] : null;
        if (!prev || !Number.isFinite(cur.nav) || !Number.isFinite(prev.nav) || prev.nav <= 0) continue;
        monDates.push(String(dates[cur.i]));
        monVals.push(cur.nav / prev.nav - 1);
      }
      const yearKeys = Object.keys(yearLast).sort();
      const yrDates = [];
      const yrVals = [];
      for (let k = 0; k < yearKeys.length; k++) {
        const y = yearKeys[k];
        const cur = yearLast[y];
        const prev = k > 0 ? yearLast[yearKeys[k - 1]] : null;
        if (!prev || !Number.isFinite(cur.nav) || !Number.isFinite(prev.nav) || prev.nav <= 0) continue;
        yrDates.push(String(dates[cur.i]));
        yrVals.push(cur.nav / prev.nav - 1);
      }
      const dm = _buildDailyMonthState({ dates, values: daily });
      const dailyCells = _buildMonthCells(dm.month, dm.map);
      const monthlyYears = _buildMonthlyYearCells({ dates: monDates, values: monVals });
      const yearlyAll = _buildYearlyCells({ dates: yrDates, values: yrVals });
      const myIdx = monthlyYears.length ? (monthlyYears.length - 1) : 0;
      const my = monthlyYears[myIdx] || { year: "", cells: [] };
      const yearPageSize = 12;
      const yearPage = 1;
      const yearlyCells = yearlyAll.slice(Math.max(0, yearlyAll.length - yearPageSize), yearlyAll.length);

      // periods table
      const rows = (rotFull.period_details || []).slice().sort((a, b) => {
        // newest first by end_date then start_date
        const ae = String(a.end_date || "");
        const be = String(b.end_date || "");
        if (ae !== be) return be.localeCompare(ae);
        const as = String(a.start_date || "");
        const bs = String(b.start_date || "");
        return bs.localeCompare(as);
      });
      const ps = Number(this.data.rotPeriodPageSize || 12);
      const pg = 1;
      const pages = Math.max(1, Math.ceil(rows.length / ps));
      const start = (pg - 1) * ps;
      const end = Math.min(rows.length, start + ps);
      const pageText = rows.length ? `第 ${pg}/${pages} 页（每页 ${ps} 期）` : "无逐期数据";

      this.setData({
        rm,
        rotCalDailyMonth: dm.month,
        rotCalDailyCells: dailyCells,
        rotCalDailyMonths: dm.months,
        rotCalDailyMap: dm.map,
        rotCalMonthlyYears: monthlyYears,
        rotCalMonthlyYearIdx: myIdx,
        rotCalMonthlyYear: my.year,
        rotCalMonthlyCells: my.cells,
        rotCalYearlyAll: yearlyAll,
        rotCalYearlyCells: yearlyCells,
        rotCalYearlyPage: yearPage,
        rotCalYearlyPageSize: yearPageSize,
        rotPeriods: rows,
        rotPeriodPage: pg,
        rotPeriodPageText: pageText,
      });
      this._setRotPeriodPage(1);

      // draw charts
      this.__rects = this.__rects || {};
      const measure = async (id) => {
        const r = await new Promise((resolve) => {
          wx.createSelectorQuery().in(this).select(`#${id}`).boundingClientRect((x) => resolve(x)).exec();
        });
        this.__rects[id] = r;
        return r;
      };
      const fmtPct = (v) => pct(v);

      // 1) nav compare
      const r1 = await measure("rNav");
      drawLineChart(wx.createCanvasContext("rNav", this), { width: Math.floor(r1.width || 320), height: Math.floor(r1.height || 220), x: dates, yMode: "linear", title: "轮动净值 vs 等权", theme: this.data.theme, series: [
        { name: "ROT", y: navRot, color: "#1677ff", width: 1.6 },
        { name: "EW", y: navEw, color: "#ff7a00", width: 1.4, dash: true },
      ]});

      // 2) dd compare
      const r2 = await measure("rDD");
      drawLineChart(wx.createCanvasContext("rDD", this), { width: Math.floor(r2.width || 320), height: Math.floor(r2.height || 200), x: dates, yMode: "linear", title: "回撤对比", yLabelFmt: fmtPct, theme: this.data.theme, series: [
        { name: "ROT DD", y: ddRot, color: "#e53935", width: 1.4 },
        { name: "EW DD", y: ddEw, color: "#999", width: 1.2, dash: true },
      ]});

      // 3) RSI
      const r3 = await measure("rRSI");
      drawLineChart(wx.createCanvasContext("rRSI", this), { width: Math.floor(r3.width || 320), height: Math.floor(r3.height || 180), x: dates, yMode: "linear", yFixed: [0,100], title: "策略 RSI24", theme: this.data.theme, series: [
        { name: "RSI24", y: rsiRot24, color: "#7b61ff", width: 1.4 },
      ]});

      // 4) rolling 3y return
      const r4 = await measure("rRR3Y");
      drawLineChart(wx.createCanvasContext("rRR3Y", this), { width: Math.floor(r4.width || 320), height: Math.floor(r4.height || 180), x: dates, yMode: "linear", title: "策略滚动三年收益率", yLabelFmt: fmtPct, theme: this.data.theme, series: [
        { name: "R3Y", y: rr3y, color: "#2e7d32", width: 1.4 },
      ]});

      // 5) rolling 3y mdd
      const r5 = await measure("rMDD3Y");
      drawLineChart(wx.createCanvasContext("rMDD3Y", this), { width: Math.floor(r5.width || 320), height: Math.floor(r5.height || 180), x: dates, yMode: "linear", title: "策略滚动三年回撤", yLabelFmt: fmtPct, theme: this.data.theme, series: [
        { name: "DD3Y", y: rdd3y, color: "#ad1457", width: 1.4 },
      ]});

      // 6) ratio + EMA/BB
      const r6 = await measure("rRatio");
      drawLineChart(wx.createCanvasContext("rRatio", this), { width: Math.floor(r6.width || 320), height: Math.floor(r6.height || 220), x: dates, yMode: "linear", title: "净值比值 ROT/EW + EMA/BB", theme: this.data.theme, series: [
        { name: "RATIO", y: ratio, color: "#1677ff", width: 1.6 },
        { name: "EMA252", y: ratioEma, color: "#ff7a00", width: 1.2, dash: true },
        { name: "BBU", y: ratioBbu, color: "#999", width: 1.0, dash: true },
        { name: "BBL", y: ratioBbl, color: "#999", width: 1.0, dash: true },
      ]});

      // 6.1) excess drawdown (based on ratio/excess nav)
      const r6b = await measure("rExDD");
      drawLineChart(wx.createCanvasContext("rExDD", this), { width: Math.floor(r6b.width || 320), height: Math.floor(r6b.height || 180), x: dates, yMode: "linear", title: "超额回撤（ROT/EW）", yLabelFmt: fmtPct, theme: this.data.theme, series: [
        { name: "EX DD", y: ddEx, color: "#ad1457", width: 1.4 },
      ]});

      // 7) ratio rsi
      const r7 = await measure("rRatioRSI");
      drawLineChart(wx.createCanvasContext("rRatioRSI", this), { width: Math.floor(r7.width || 320), height: Math.floor(r7.height || 180), x: dates, yMode: "linear", yFixed: [0,100], title: "比值 RSI24", theme: this.data.theme, series: [
        { name: "RSI24", y: ratioRsi24, color: "#7b61ff", width: 1.4 },
      ]});

      // 8) 40d diff
      const r8 = await measure("rDiff40");
      drawLineChart(wx.createCanvasContext("rDiff40", this), { width: Math.floor(r8.width || 320), height: Math.floor(r8.height || 180), x: dates, yMode: "linear", title: "40日收益差（ROT-EW）", yLabelFmt: fmtPct, theme: this.data.theme, series: [
        { name: "DIFF40", y: diff40, color: "#1677ff", width: 1.2 },
      ]});

      // 10/11 attribution bars
      const attr = rotFull.attribution || {};
      const rcRows = ((attr.return || {}).by_code || []).slice();
      const rkRows = ((attr.risk || {}).by_code || []).slice();
      const rcLabels = rcRows.map((x) => dispCode(x.code));
      const rcVals = rcRows.map((x) => (x.return_share == null ? NaN : Number(x.return_share)));
      const rkLabels = rkRows.map((x) => dispCode(x.code));
      const rkVals = rkRows.map((x) => (x.risk_share == null ? NaN : Number(x.risk_share)));
      const r10 = await measure("rRC");
      drawBarChart(wx.createCanvasContext("rRC", this), { width: Math.floor(r10.width || 320), height: Math.floor(r10.height || 180), title: "收益贡献", theme: this.data.theme, labels: rcLabels, values: rcVals, colors: ["#1677ff", "#00bfa5", "#ff7a00", "#7b61ff"], valueFmt: (x) => (x * 100).toFixed(1) + "%", vmin: 0, vmax: 1 });
      const r11 = await measure("rRiskC");
      drawBarChart(wx.createCanvasContext("rRiskC", this), { width: Math.floor(r11.width || 320), height: Math.floor(r11.height || 180), title: "风险贡献", theme: this.data.theme, labels: rkLabels, values: rkVals, colors: ["#e53935", "#d81b60", "#8e24aa", "#3949ab"], valueFmt: (x) => (x * 100).toFixed(1) + "%", vmin: 0, vmax: 1 });
    },

    setCalMode(e) {
      const m = e.currentTarget.dataset.m;
      const mode = (m === "monthly" || m === "yearly" || m === "daily") ? m : "daily";
      const modeLabel = mode === "monthly" ? "月度" : (mode === "yearly" ? "年度" : "日度");
      this.setData({ calMode: mode, calModeLabel: modeLabel });
    },

    // ----- rotation calendar controls -----
    setRotCalMode(e) {
      const m = (e && e.currentTarget && e.currentTarget.dataset) ? e.currentTarget.dataset.m : "";
      const mode = (m === "monthly" || m === "yearly") ? m : "daily";
      const label = (mode === "daily") ? "日度" : (mode === "monthly" ? "月度" : "年度");
      this.setData({ rotCalMode: mode, rotCalModeLabel: label });
    },

    prevRotCalMonth() {
      const ms = this.data.rotCalDailyMonths || [];
      if (!ms.length) return;
      const cur = String(this.data.rotCalDailyMonth || "");
      const i = ms.indexOf(cur);
      const j = (i <= 0) ? 0 : (i - 1);
      const month = ms[j];
      const cells = _buildMonthCells(month, this.data.rotCalDailyMap || {});
      this.setData({ rotCalDailyMonth: month, rotCalDailyCells: cells });
    },

    nextRotCalMonth() {
      const ms = this.data.rotCalDailyMonths || [];
      if (!ms.length) return;
      const cur = String(this.data.rotCalDailyMonth || "");
      const i = ms.indexOf(cur);
      const j = (i < 0) ? (ms.length - 1) : Math.min(ms.length - 1, i + 1);
      const month = ms[j];
      const cells = _buildMonthCells(month, this.data.rotCalDailyMap || {});
      this.setData({ rotCalDailyMonth: month, rotCalDailyCells: cells });
    },

    prevRotCalYear() {
      const ys = this.data.rotCalMonthlyYears || [];
      if (!ys.length) return;
      const i = Number(this.data.rotCalMonthlyYearIdx || 0);
      const j = Math.max(0, i - 1);
      const y = ys[j] || { year: "", cells: [] };
      this.setData({ rotCalMonthlyYearIdx: j, rotCalMonthlyYear: y.year, rotCalMonthlyCells: y.cells });
    },

    nextRotCalYear() {
      const ys = this.data.rotCalMonthlyYears || [];
      if (!ys.length) return;
      const i = Number(this.data.rotCalMonthlyYearIdx || 0);
      const j = Math.min(ys.length - 1, i + 1);
      const y = ys[j] || { year: "", cells: [] };
      this.setData({ rotCalMonthlyYearIdx: j, rotCalMonthlyYear: y.year, rotCalMonthlyCells: y.cells });
    },

    prevRotYearPage() {
      const all = this.data.rotCalYearlyAll || [];
      const size = Number(this.data.rotCalYearlyPageSize || 12);
      if (!all.length) return;
      const pages = Math.max(1, Math.ceil(all.length / size));
      const cur = Number(this.data.rotCalYearlyPage || 1);
      const next = Math.max(1, cur - 1);
      const start = Math.max(0, all.length - next * size);
      const end = Math.min(all.length, start + size);
      this.setData({ rotCalYearlyPage: next, rotCalYearlyCells: all.slice(start, end) });
    },

    nextRotYearPage() {
      const all = this.data.rotCalYearlyAll || [];
      const size = Number(this.data.rotCalYearlyPageSize || 12);
      if (!all.length) return;
      const pages = Math.max(1, Math.ceil(all.length / size));
      const cur = Number(this.data.rotCalYearlyPage || 1);
      const next = Math.min(pages, cur + 1);
      const start = Math.max(0, all.length - next * size);
      const end = Math.min(all.length, start + size);
      this.setData({ rotCalYearlyPage: next, rotCalYearlyCells: all.slice(start, end) });
    },

    // ----- rotation period pagination -----
    _setRotPeriodPage(pg) {
      const rows = this.data.rotPeriods || [];
      const ps = Number(this.data.rotPeriodPageSize || 12);
      const pages = Math.max(1, Math.ceil(rows.length / ps));
      const p = Math.max(1, Math.min(pages, Number(pg || 1)));
      const start = (p - 1) * ps;
      const end = Math.min(rows.length, start + ps);
      const view = rows.slice(start, end).map((r, i) => {
        const execDate = String(r.start_date || r.end_date || "");
        const buys = Array.isArray(r.buys) ? r.buys : [];
        const sells = Array.isArray(r.sells) ? r.sells : [];
        const cashText = "空";
        const abbr = (code) => {
          const c = String(code || "").trim();
          if (!c) return cashText;
          // fixed pool abbreviations
          if (c === "159915") return "创";
          if (c === "511010") return "债";
          if (c === "513100") return "纳";
          if (c === "518880") return "金";
          // fallback: first char of display name
          const nm = dispCode(c);
          return String(nm || c).slice(0, 1);
        };
        // In this mini-program fixed strategy, turnover is effectively 0% or 100%.
        // Show "无调仓" or "X->Y" only.
        const hasTrade = buys.length > 0 || sells.length > 0;
        let fromCode = null;
        let toCode = null;
        if (sells.length) {
          // pick the main sold leg (largest previous weight)
          let best = sells[0];
          for (const leg of sells) {
            if (Number(leg.from_weight || 0) > Number(best.from_weight || 0)) best = leg;
          }
          fromCode = best.code;
        }
        if (buys.length) {
          // pick the main bought leg (largest target weight)
          let best = buys[0];
          for (const leg of buys) {
            if (Number(leg.to_weight || 0) > Number(best.to_weight || 0)) best = leg;
          }
          toCode = best.code;
        }
        const tradeTxt = hasTrade ? `${abbr(fromCode)}->${abbr(toCode)}` : "无";
        return {
        k: `${p}-${i}`,
        exec_date: execDate,
        start_date: r.start_date,
        end_date: r.end_date,
        strategy_return_text: pct(r.strategy_return),
        equal_weight_return_text: pct(r.equal_weight_return),
        excess_return_text: pct(r.excess_return),
        trade_text: tradeTxt,
      };
      });
      const pageText = rows.length ? `第 ${p}/${pages} 页（每页 ${ps} 期）` : "无逐期数据";
      this.setData({ rotPeriodPage: p, rotPeriodPageText: pageText, rotPeriodsView: view });
    },

    prevRotPeriodPage() {
      this._setRotPeriodPage(Number(this.data.rotPeriodPage || 1) - 1);
    },

    nextRotPeriodPage() {
      this._setRotPeriodPage(Number(this.data.rotPeriodPage || 1) + 1);
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

    prevCalYear() {
      const ys = this.data.calMonthlyYears || [];
      if (!ys.length) return;
      const i = Number(this.data.calMonthlyYearIdx || 0);
      const j = Math.max(0, i - 1);
      const y = ys[j] || { year: "", cells: [] };
      this.setData({ calMonthlyYearIdx: j, calMonthlyYear: y.year, calMonthlyCells: y.cells });
    },

    nextCalYear() {
      const ys = this.data.calMonthlyYears || [];
      if (!ys.length) return;
      const i = Number(this.data.calMonthlyYearIdx || 0);
      const j = Math.min(ys.length - 1, i + 1);
      const y = ys[j] || { year: "", cells: [] };
      this.setData({ calMonthlyYearIdx: j, calMonthlyYear: y.year, calMonthlyCells: y.cells });
    },

    prevYearPage() {
      const all = this.data.calYearlyAll || [];
      const size = Number(this.data.calYearlyPageSize || 12);
      if (!all.length) return;
      const pages = Math.max(1, Math.ceil(all.length / size));
      const cur = Number(this.data.calYearlyPage || 1);
      const next = Math.max(1, cur - 1);
      const start = Math.max(0, all.length - next * size);
      const end = Math.min(all.length, start + size);
      this.setData({ calYearlyPage: next, calYearlyCells: all.slice(start, end) });
    },

    nextYearPage() {
      const all = this.data.calYearlyAll || [];
      const size = Number(this.data.calYearlyPageSize || 12);
      if (!all.length) return;
      const pages = Math.max(1, Math.ceil(all.length / size));
      const cur = Number(this.data.calYearlyPage || 1);
      const next = Math.min(pages, cur + 1);
      const start = Math.max(0, all.length - next * size);
      const end = Math.min(all.length, start + size);
      this.setData({ calYearlyPage: next, calYearlyCells: all.slice(start, end) });
    },

    onChartTouch(e) {
      try {
        // If we scheduled auto-hide on touch end, cancel it when touching again.
        if (this.__tipTimer) {
          clearTimeout(this.__tipTimer);
          this.__tipTimer = null;
        }
        const id = e.currentTarget.id;
        const raw = (id && id.startsWith("r")) ? (this.__rot || {}) : (this.__raw || {});
        const dates = (raw.dates || (raw.nav && raw.nav.dates) || []);
        const n = dates.length;
        if (!id || !n) return;
        const rect = (this.__rects && this.__rects[id]) ? this.__rects[id] : null;
        const touch = (e.touches && e.touches[0]) ? e.touches[0] : ((e.changedTouches && e.changedTouches[0]) ? e.changedTouches[0] : null);
        if (!rect || !touch) return;
        // Touch coordinate fields vary across devices/WeChat versions:
        // - iOS/Android may provide pageX/pageY instead of clientX/clientY
        // - some versions provide x/y
        const absX =
          (touch.clientX != null) ? Number(touch.clientX)
          : ((touch.pageX != null) ? Number(touch.pageX)
            : ((touch.x != null) ? Number(touch.x) : NaN));
        const absY =
          (touch.clientY != null) ? Number(touch.clientY)
          : ((touch.pageY != null) ? Number(touch.pageY)
            : ((touch.y != null) ? Number(touch.y) : NaN));
        const xRel = absX - Number(rect.left);
        const yRel = absY - Number(rect.top);
        const idx = lineIndexFromTouchX(xRel, Number(rect.width), n);
        if (idx == null) return;
        // Deduplicate updates: touchmove fires a lot; only update when data-index changes.
        this.__tipLast = this.__tipLast || {};
        if (this.__tipLast.id === id && this.__tipLast.idx === idx) return;
        this.__tipLast = { id, idx };

        const d = String(dates[idx]);
        const fmtPct = (v) => pct(v);
        const fmtNum = (v) => {
          const x = Number(v);
          if (!Number.isFinite(x)) return "-";
          return x.toFixed(4);
        };

        let lines = [];
        const rotCache = this.__rotCache || null;
        if (id === "cNav") {
          lines = [
            `NAV=${fmtNum((raw.nav || [])[idx])}`,
            `EMA252=${fmtNum((raw.ema252 || [])[idx])}`,
            `BBU=${fmtNum((raw.bb_upper || [])[idx])}`,
            `BBL=${fmtNum((raw.bb_lower || [])[idx])}`,
          ];
        } else if (id === "cDD") {
          lines = [`DD=${fmtPct((raw.drawdown || [])[idx])}`];
        } else if (id === "cRSI") {
          lines = [`RSI24=${fmtNum((raw.rsi24 || [])[idx])}`];
        } else if (id === "cRR3Y") {
          lines = [`R3Y=${fmtPct((raw.roll3y_return || [])[idx])}`];
        } else if (id === "cRDD3Y") {
          const arr = (raw.roll3y_dd || raw.roll3y_mdd || []);
          lines = [`DD3Y=${fmtPct(arr[idx])}`];
        } else if (id === "rNav") {
          const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
          lines = [`ROT=${fmtNum((s.ROTATION || [])[idx])}`, `EW=${fmtNum((s.EW_REBAL || [])[idx])}`];
        } else if (id === "rDD") {
          if (rotCache && Array.isArray(rotCache.ddRot) && Array.isArray(rotCache.ddEw)) {
            lines = [`ROT_DD=${fmtPct(rotCache.ddRot[idx])}`, `EW_DD=${fmtPct(rotCache.ddEw[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            lines = [`ROT_DD=${fmtPct(_drawdownFromNav(s.ROTATION || [])[idx])}`, `EW_DD=${fmtPct(_drawdownFromNav(s.EW_REBAL || [])[idx])}`];
          }
        } else if (id === "rRSI") {
          const rsiSeries = (((raw.nav_rsi || {}).series || {}).ROTATION || {})["24"] || [];
          lines = [`RSI24=${fmtNum(rsiSeries[idx])}`];
        } else if (id === "rRR3Y") {
          if (rotCache && Array.isArray(rotCache.rr3y)) {
            lines = [`R3Y=${fmtPct(rotCache.rr3y[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            lines = [`R3Y=${fmtPct(_rollingReturn(s.ROTATION || [], 3 * 252)[idx])}`];
          }
        } else if (id === "rMDD3Y") {
          if (rotCache && Array.isArray(rotCache.dd3y)) {
            lines = [`DD3Y=${fmtPct(rotCache.dd3y[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            lines = [`DD3Y=${fmtPct(_rollingDrawdown(s.ROTATION || [], 3 * 252)[idx])}`];
          }
        } else if (id === "rRatio") {
          if (rotCache && Array.isArray(rotCache.ratio) && Array.isArray(rotCache.ratioEma)) {
            lines = [
              `RATIO=${fmtNum(rotCache.ratio[idx])}`,
              `EMA252=${fmtNum(rotCache.ratioEma[idx])}`,
              `BBU=${fmtNum((rotCache.ratioBbu || [])[idx])}`,
              `BBL=${fmtNum((rotCache.ratioBbl || [])[idx])}`,
            ];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            const ratio = (s.ROTATION || []).map((v, i) => {
              const a = Number(v), b = Number((s.EW_REBAL || [])[i]);
              if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
              return a / b;
            });
            const ema = _emaArr(ratio, 252);
            const sd = _rollingStdArr(ratio, 252);
            const bbu = ema.map((v, i) => (v != null && sd[i] != null) ? (Number(v) + 2 * Number(sd[i])) : null);
            const bbl = ema.map((v, i) => (v != null && sd[i] != null) ? (Number(v) - 2 * Number(sd[i])) : null);
            lines = [`RATIO=${fmtNum(ratio[idx])}`, `EMA252=${fmtNum(ema[idx])}`, `BBU=${fmtNum(bbu[idx])}`, `BBL=${fmtNum(bbl[idx])}`];
          }
        } else if (id === "rExDD") {
          if (rotCache && Array.isArray(rotCache.ddEx)) {
            lines = [`EX_DD=${fmtPct(rotCache.ddEx[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            const ex = (s.EXCESS || []).length ? (s.EXCESS || []) : (s.ROTATION || []).map((v, i) => {
              const a = Number(v), b = Number((s.EW_REBAL || [])[i]);
              if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
              return a / b;
            });
            lines = [`EX_DD=${fmtPct(_drawdownFromNav(ex)[idx])}`];
          }
        } else if (id === "rRatioRSI") {
          if (rotCache && Array.isArray(rotCache.ratioRsi24)) {
            lines = [`RSI24=${fmtNum(rotCache.ratioRsi24[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            const ratio = (s.ROTATION || []).map((v, i) => {
              const a = Number(v), b = Number((s.EW_REBAL || [])[i]);
              if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return NaN;
              return a / b;
            });
            const rsi = _rsiWilder(ratio, 24);
            lines = [`RSI24=${fmtNum(rsi[idx])}`];
          }
        } else if (id === "rDiff40") {
          if (rotCache && Array.isArray(rotCache.diff40)) {
            lines = [`DIFF40=${fmtPct(rotCache.diff40[idx])}`];
          } else {
            const s = (raw.nav && raw.nav.series) ? raw.nav.series : {};
            const rot40 = _rollingReturn(s.ROTATION || [], 40);
            const ew40 = _rollingReturn(s.EW_REBAL || [], 40);
            const v = (rot40[idx] == null || ew40[idx] == null) ? null : (Number(rot40[idx]) - Number(ew40[idx]));
            lines = [`DIFF40=${fmtPct(v)}`];
          }
        }
        if (!lines.length) return;

        // tooltip position inside canvas wrap (px)
        const w = Number(rect.width || 0);
        const h = Number(rect.height || 0);
        const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
        const cx = clamp(lineXFromIndex(idx, w, n), 0, Math.max(0, w));
        // Estimate tooltip box size to prevent overflow on right/bottom.
        const maxChars = Math.max(String(d || "").length, ...lines.map((s) => String(s || "").length));
        const estW = Math.max(120, Math.min(w * 0.8, 24 + maxChars * 7)); // px (rough)
        const estH = Math.max(46, Math.min(h * 0.6, 20 + (lines.length + 1) * 16)); // px (rough)
        const margin = 8;

        // Prefer to show on the right; if it would overflow, flip to left.
        const tx0 = (Number(xRel) + margin + estW <= w) ? (Number(xRel) + margin) : (Number(xRel) - margin - estW);
        const tx = clamp(tx0, 0, Math.max(0, w - estW));

        // Prefer above the finger; if it would overflow, place below.
        const ty0 = (Number(yRel) - margin - estH >= 0) ? (Number(yRel) - margin - estH) : (Number(yRel) + margin);
        const ty = clamp(ty0, 0, Math.max(0, h - estH));
        this.setData({ tip: { show: true, id, x: tx, y: ty, text: `${d}\n${lines.join("\n")}`, cx, cw: w, ch: h } });
      } catch (err) {
        // best effort
      }
    },

    onChartTouchEnd() {
      const tip = this.data.tip || {};
      // For taps, hiding immediately makes it look like "no tooltip".
      // Keep it briefly so users can read it.
      if (!tip.show) return;
      if (this.__tipTimer) clearTimeout(this.__tipTimer);
      this.__tipTimer = setTimeout(() => {
        this.setData({ tip: { show: false, id: "", x: 0, y: 0, text: "", cx: 0, cw: 0, ch: 0 } });
        this.__tipTimer = null;
      }, 1200);
    },
  };
}

module.exports = { attachWeekdayPage };

