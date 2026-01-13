## 微信小程序（MVP）

本目录是 `docs/WECHAT_MINIPROGRAM_PROJECT.md` 对应的小程序端最小骨架（P0/P2/P3）。

### 运行方式（开发态）
- 使用微信开发者工具打开 `miniprogram/` 目录
- 在 `miniprogram/utils/config.js` 配置后端 `BASE_URL`
  - 需要是微信小程序“合法域名”（HTTPS）或本地调试代理域名

### 依赖后端接口（MVP）
- `POST /api/sim/portfolio`
- `GET /api/sim/portfolio`
- `POST /api/sim/portfolio/{id}/init-fixed-strategy`
- `GET /api/sim/portfolio/{id}/variants`
- `POST /api/sim/decision/generate`
- `GET /api/sim/variant/{id}/status`
- `GET /api/sim/variant/{id}/decisions`
- `GET /api/sim/variant/{id}/trades`
- `GET /api/sim/variant/{id}/nav`
- `POST /api/sim/trade/preview`
- `POST /api/sim/trade/confirm`
- `POST /api/sim/mark-to-market`

