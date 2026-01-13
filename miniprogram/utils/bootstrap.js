const { request } = require("./api");

const KEY_PID = "portfolio_id";

async function ensureBootstrap() {
  // 1) use cached portfolio id if exists
  const cached = wx.getStorageSync(KEY_PID);
  if (cached) return cached;

  // 2) find existing
  const ports = await request("/sim/portfolio");
  if (Array.isArray(ports) && ports.length > 0) {
    const pid = ports[0].id;
    wx.setStorageSync(KEY_PID, pid);
    return pid;
  }

  // 3) create default portfolio
  const created = await request("/sim/portfolio", { method: "POST", data: { name: "默认账户", initial_cash: 1000000 } });
  const pid = created.id;
  wx.setStorageSync(KEY_PID, pid);

  // 4) init fixed strategy variants
  await request(`/sim/portfolio/${pid}/init-fixed-strategy`, { method: "POST" });
  return pid;
}

module.exports = { ensureBootstrap };

