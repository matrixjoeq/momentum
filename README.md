# momentum
ETF动量轮动策略研究

## 开始使用（候选池配置 + 数据入库 MVP）

### 环境准备
- 使用项目内虚拟环境（避免 macOS PEP668 限制）：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

> 说明：目前 `akshare` 固定使用 `1.16.72`，以避免其新版本在部分环境中引入的 `curl_cffi` 构建问题。

### Windows PowerShell 使用说明（仍以 python3 为前提）

> 前提：系统已能在 PowerShell 中运行 `python3`（例如 `python3 --version`）。
>
> 说明：PowerShell 常见的执行策略（ExecutionPolicy）可能会阻止运行 `.ps1` 脚本（例如 venv 的 `Activate.ps1`）。
> 为了避免策略问题，下面的命令 **不依赖激活 venv**，而是直接调用 venv 里的 Python。

- 创建虚拟环境并安装依赖（在项目根目录执行）：

```powershell
python3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- 运行服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn etf_momentum.app:app --reload --port 8000
```

- 运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 运行服务

```bash
.venv/bin/python -m uvicorn etf_momentum.app:app --reload --port 8000
```

- Web UI：打开 `http://127.0.0.1:8000/`
- API 文档：打开 `http://127.0.0.1:8000/docs`

### API（简要）
- `GET /api/etf`：列出候选池
- `POST /api/etf`：新增/更新候选（code, name, start_date, end_date）
- `DELETE /api/etf/{code}`：删除候选
- `POST /api/etf/{code}/fetch`：抓取单个 ETF 日频前复权并写入 SQLite
- `POST /api/fetch-all`：抓取全部候选并入库

### 数据落地
- 默认 SQLite 路径：`data/etf_momentum.sqlite3`
- 可通过环境变量覆盖：
  - `MOMENTUM_SQLITE_PATH`：SQLite 文件路径
  - `MOMENTUM_DEFAULT_START_DATE` / `MOMENTUM_DEFAULT_END_DATE`：默认抓取区间（YYYYMMDD）

### 运行测试

```bash
.venv/bin/python -m pytest
```
