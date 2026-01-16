const { request } = require("./api");

const KEY_PID = "portfolio_id";

async function ensureBootstrap() {
  async function ensureVariants(pid) {
    const vs = await request(`/sim/portfolio/${pid}/variants`, { method: "GET" });
    const items = (vs && vs.variants) ? vs.variants : [];
    if (Array.isArray(items) && items.length > 0) return pid;
    await request(`/sim/portfolio/${pid}/init-fixed-strategy`, { method: "POST", data: {} });
    return pid;
  }

  // 1) use cached portfolio id if exists (and still valid)
  const cached = wx.getStorageSync(KEY_PID);
  if (cached) {
    try {
      await ensureVariants(cached);
      return cached;
    } catch (e) {
      // cached pid might be invalid (DB reset) or missing variants; fall through
      try { wx.removeStorageSync(KEY_PID); } catch (_) {}
    }
  }

  // 2) find existing
  const ports = await request("/sim/portfolio");
  if (Array.isArray(ports) && ports.length > 0) {
    const pid = ports[0].id;
    await ensureVariants(pid);
    wx.setStorageSync(KEY_PID, pid);
    return pid;
  }

  // 3) create default portfolio
  const created = await request("/sim/portfolio", { method: "POST", data: { name: "默认账户", initial_cash: 1000000 } });
  const pid = created.id;
  wx.setStorageSync(KEY_PID, pid);

  // 4) init fixed strategy variants
  await ensureVariants(pid);
  return pid;
}

module.exports = { ensureBootstrap };

