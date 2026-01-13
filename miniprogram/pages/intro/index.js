const KEY_SKIP = "intro_skip";
const KEY_ACK_AT = "intro_ack_at";

Page({
  data: {
    acknowledged: false,
    skipNext: false,
  },

  onLoad() {
    const skip = !!wx.getStorageSync(KEY_SKIP);
    if (skip) {
      wx.switchTab({ url: "/pages/wd0/index" });
      return;
    }
  },

  onAcknowledgeChange(e) {
    const vals = (e.detail && e.detail.value) ? e.detail.value : [];
    this.setData({ acknowledged: vals.includes("ack") });
  },

  onSkipChange(e) {
    const vals = (e.detail && e.detail.value) ? e.detail.value : [];
    this.setData({ skipNext: vals.includes("skip") });
  },

  onEnter() {
    if (!this.data.acknowledged) return;
    if (this.data.skipNext) {
      wx.setStorageSync(KEY_SKIP, true);
    } else {
      wx.removeStorageSync(KEY_SKIP);
    }
    wx.setStorageSync(KEY_ACK_AT, Date.now());
    wx.switchTab({ url: "/pages/wd0/index" });
  },
});

