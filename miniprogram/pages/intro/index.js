const KEY_SKIP = "intro_skip";
const KEY_ACK_AT = "intro_ack_at";

function defaultTabUrl() {
  // JS Date.getDay(): 0=Sun..6=Sat
  const d = new Date();
  const wd = Number(d.getDay());
  // weekend -> mix page
  if (wd === 0 || wd === 6) return "/pages/mix/index";
  // Mon..Fri -> wd0..wd4
  return `/pages/wd${wd - 1}/index`;
}

function goDefault() {
  const url = defaultTabUrl();
  // tab pages must use switchTab
  if (url.startsWith("/pages/wd")) {
    wx.switchTab({ url });
  } else {
    wx.redirectTo({ url });
  }
}

Page({
  data: {
    acknowledged: false,
    skipNext: false,
  },

  onLoad() {
    const skip = !!wx.getStorageSync(KEY_SKIP);
    if (skip) {
      goDefault();
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
    goDefault();
  },
});

