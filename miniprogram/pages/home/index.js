const { request } = require("../../utils/api");

function fmt(x) {
  if (x == null) return "-";
  if (typeof x === "number") return x.toFixed(4);
  return String(x);
}

function fmtPct(x) {
  if (x == null) return "-";
  const v = Number(x);
  if (!Number.isFinite(v)) return "-";
  return (v * 100).toFixed(1) + "%";
}

Page({
  data: {
    statusText: "",
    variants: [],
  },

  onShow() {
    this.onRefresh();
  },

  async _ensurePid() {
    const app = getApp();
    if (app && typeof app.ensurePortfolioReady === "function") {
      await app.ensurePortfolioReady();
    } else if (app && app.globalData && app.globalData.portfolioReady) {
      await app.globalData.portfolioReady;
    }
    return (app && app.globalData && app.globalData.portfolioId) || wx.getStorageSync("portfolio_id");
  },

  async onRefresh() {
    try {
      this.setData({ statusText: "加载中..." });
      const pid = await this._ensurePid();
      if (!pid) throw new Error("初始化账户失败，请稍后重试");
      const vs = await request(`/sim/portfolio/${pid}/variants`);
      const items = vs.variants || [];

      // fetch each variant status (sequential for simplicity)
      const out = [];
      for (const v of items) {
        const st = await request(`/sim/variant/${v.id}/status`);
        const positions = st.positions || {};
        const posText = Object.keys(positions).length === 0
          ? "现金"
          : Object.entries(positions).map(([k, q]) => `${k}:${Number(q).toFixed(2)}`).join(" ");
        out.push({
          id: v.id,
          label: v.label,
          is_active: v.is_active,
          asof: st.asof,
          nav: fmt(st.nav),
          mdd: fmtPct(st.mdd),
          posText,
        });
      }
      this.setData({ variants: out, statusText: "OK" });
    } catch (e) {
      console.error(e);
      this.setData({ statusText: e.message || "加载失败" });
    }
  },

  goVariant(e) {
    const id = e.currentTarget.dataset.id;
    wx.navigateTo({ url: `/pages/variant/index?id=${id}` });
  },

  goExecute(e) {
    const id = e.currentTarget.dataset.id;
    wx.navigateTo({ url: `/pages/execute/index?id=${id}` });
  },

  async setActive(e) {
    const id = e.currentTarget.dataset.id;
    try {
      await request(`/sim/variant/${id}/set-active`, { method: "POST" });
      await this.onRefresh();
    } catch (err) {
      wx.showToast({ title: err.message || "失败", icon: "none" });
    }
  },

  async onGenerate() {
    try {
      const pid = await this._ensurePid();
      if (!pid) throw new Error("初始化账户失败，请稍后重试");
      const end = new Date();
      const start = new Date(end.getTime() - 365 * 24 * 3600 * 1000);
      const fmtYmd = (d) => {
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, "0");
        const dd = String(d.getDate()).padStart(2, "0");
        return `${y}${m}${dd}`;
      };
      wx.showLoading({ title: "生成中..." });
      await request("/sim/decision/generate", { method: "POST", data: { portfolio_id: pid, start: fmtYmd(start), end: fmtYmd(end) } });
      wx.hideLoading();
      wx.showToast({ title: "OK", icon: "success" });
    } catch (e) {
      wx.hideLoading();
      wx.showToast({ title: e.message || "失败", icon: "none" });
    }
  },

  async onMark() {
    try {
      const end = new Date();
      const start = new Date(end.getTime() - 30 * 24 * 3600 * 1000);
      const fmtYmd = (d) => {
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, "0");
        const dd = String(d.getDate()).padStart(2, "0");
        return `${y}${m}${dd}`;
      };
      wx.showLoading({ title: "更新中..." });
      for (const v of (this.data.variants || [])) {
        await request(`/sim/mark-to-market?variant_id=${v.id}&start=${fmtYmd(start)}&end=${fmtYmd(end)}`, { method: "POST" });
      }
      wx.hideLoading();
      await this.onRefresh();
    } catch (e) {
      wx.hideLoading();
      wx.showToast({ title: e.message || "失败", icon: "none" });
    }
  },
});

