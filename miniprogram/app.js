const { ensureBootstrap } = require("./utils/bootstrap");

App({
  globalData: {
    portfolioId: null,
    portfolioReady: null,
  },

  onLaunch() {
    // Kick off bootstrap ASAP; pages can await ensurePortfolioReady().
    this.globalData.portfolioReady = this.ensurePortfolioReady();
  },

  async ensurePortfolioReady() {
    if (this.globalData.portfolioId) return this.globalData.portfolioId;
    try {
      const pid = await ensureBootstrap();
      this.globalData.portfolioId = pid;
      return pid;
    } catch (e) {
      console.error("bootstrap failed", e);
      return null;
    }
  },
});

