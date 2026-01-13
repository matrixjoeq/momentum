const { request } = require("../../utils/api");

Page({
  data: {
    id: null,
    label: "",
    decisionId: null,
    tradeDate: "",
    currentCode: null,
    targetCode: null,
    sells: [],
    buys: [],
  },

  onLoad(query) {
    const id = Number(query.id);
    this.setData({ id });
  },

  onShow() {
    this.onRefresh();
  },

  async _pickLatestDecisionId() {
    const id = this.data.id;
    const dec = await request(`/sim/variant/${id}/decisions`);
    const xs = dec.decisions || [];
    if (xs.length === 0) return null;
    return xs[xs.length - 1].id;
  },

  async onRefresh() {
    try {
      const id = this.data.id;
      const st = await request(`/sim/variant/${id}/status`);
      const decisionId = await this._pickLatestDecisionId();
      if (!decisionId) {
        this.setData({ label: st.label, decisionId: null, tradeDate: "-", currentCode: null, targetCode: null, sells: [], buys: [] });
        wx.showToast({ title: "暂无决策，请先生成决策", icon: "none" });
        return;
      }
      const prev = await request("/sim/trade/preview", { method: "POST", data: { variant_id: id, decision_id: decisionId } });
      this.setData({
        label: st.label,
        decisionId,
        tradeDate: prev.trade_date,
        currentCode: prev.current_code,
        targetCode: prev.target_code,
        sells: prev.sells || [],
        buys: prev.buys || [],
      });
    } catch (e) {
      console.error(e);
      wx.showToast({ title: e.message || "失败", icon: "none" });
    }
  },

  async onConfirm() {
    const id = this.data.id;
    const decisionId = this.data.decisionId;
    if (!decisionId) return;
    try {
      wx.showLoading({ title: "提交中..." });
      await request("/sim/trade/confirm", { method: "POST", data: { variant_id: id, decision_id: decisionId } });
      wx.hideLoading();
      wx.showToast({ title: "已确认", icon: "success" });
      await this.onRefresh();
    } catch (e) {
      wx.hideLoading();
      wx.showToast({ title: e.message || "失败", icon: "none" });
    }
  },
});

