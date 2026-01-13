const { ensureBootstrap } = require("./utils/bootstrap");

App({
  globalData: {
    portfolioId: null,
  },

  onLaunch() {
    ensureBootstrap()
      .then((pid) => {
        this.globalData.portfolioId = pid;
      })
      .catch((e) => {
        console.error("bootstrap failed", e);
      });
  },
});

