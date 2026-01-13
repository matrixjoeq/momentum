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
    id: null,
    label: "",
    asof: "",
    nav: "-",
    mdd: "-",
    posText: "",
    decisions: [],
    trades: [],
  },

  onLoad(query) {
    const id = Number(query.id);
    this.setData({ id });
  },

  onShow() {
    this.refresh();
  },

  async refresh() {
    try {
      const id = this.data.id;
      const st = await request(`/sim/variant/${id}/status`);
      const positions = st.positions || {};
      const posText = Object.keys(positions).length === 0
        ? "现金"
        : Object.entries(positions).map(([k, q]) => `${k}:${Number(q).toFixed(2)}`).join(" ");
      this.setData({
        label: st.label,
        asof: st.asof || "-",
        nav: fmt(st.nav),
        mdd: fmtPct(st.mdd),
        posText,
      });

      // decisions/trades (last 20)
      const dec = await request(`/sim/variant/${id}/decisions`);
      const trs = await request(`/sim/variant/${id}/trades`);
      this.setData({
        decisions: (dec.decisions || []).slice(-20).reverse(),
        trades: (trs.trades || []).slice(-20).reverse(),
      });
    } catch (e) {
      console.error(e);
      wx.showToast({ title: e.message || "加载失败", icon: "none" });
    }
  },

  goExecute() {
    wx.navigateTo({ url: `/pages/execute/index?id=${this.data.id}` });
  },
});

