const { BASE_URL } = require("./config");

function request(path, { method = "GET", data = null } = {}) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${BASE_URL}/api${path}`,
      method,
      data,
      header: { "content-type": "application/json" },
      success: (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data);
        } else {
          const msg = (res.data && (res.data.detail || res.data.message)) ? (res.data.detail || res.data.message) : JSON.stringify(res.data);
          reject(new Error(`${res.statusCode}: ${msg}`));
        }
      },
      fail: (err) => reject(err),
    });
  });
}

module.exports = { request };

