from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import qlib
import matplotlib.pyplot as plt
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord


@dataclasses.dataclass(frozen=True)
class QlibConfig:
    provider_uri: str
    region: str


def _load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def init_qlib(cfg: QlibConfig) -> None:
    qlib.init(provider_uri=cfg.provider_uri, region=cfg.region)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def build_dataset(handler_cfg: dict, segments: dict) -> dict:
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_cfg,
            "segments": {
                "train": tuple(segments["train"]),
                "valid": tuple(segments["valid"]),
                "test": tuple(segments["test"]),
            },
        },
    }


def build_port_config(cfg: dict, signal, strategy: dict) -> dict:
    automl = cfg["automl"]
    backtest = cfg["backtest"]
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": signal,
                "topk": int(strategy["topk"]),
                "n_drop": int(strategy["drop"]),
            },
        },
        "backtest": {
            "start_time": backtest["start"],
            "end_time": backtest["end"],
            "account": float(automl["account"]),
            "benchmark": automl["benchmark"],
            "exchange_kwargs": {"deal_price": "close", "limit_threshold": 0.095},
        },
    }

def _artifact_dir(recorder) -> Path | None:
    try:
        uri = recorder.get_artifact_uri()
    except Exception:  # pragma: no cover
        uri = None
    if uri:
        if uri.startswith("file://"):
            return Path(uri.replace("file://", "", 1))
        if uri.startswith("file:"):
            return Path(uri.replace("file:", "", 1))
        p = Path(uri)
        if p.exists():
            return p
    try:
        p = Path(recorder.get_local_dir())
        if p.exists():
            return p
    except Exception:  # pragma: no cover
        return None
    return None


def _extract_metrics_from_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    if "name" in cols and "value" in cols:
        out = {}
        for _, row in df.iterrows():
            k = str(row[cols["name"]])
            v = row[cols["value"]]
            if k:
                out[k] = v
        return out
    row = df.iloc[0].to_dict()
    return {str(k): row[k] for k in row.keys()}


def _find_metrics(recorder) -> dict:
    art = _artifact_dir(recorder)
    if art is None:
        return {}
    candidates = []
    for p in art.rglob("*.csv"):
        name = p.name.lower()
        score = 0
        if "analysis" in name:
            score += 3
        if "report" in name:
            score += 2
        if "indicator" in name or "metrics" in name or "result" in name:
            score += 2
        candidates.append((score, p))
    candidates.sort(reverse=True, key=lambda x: x[0])
    for _, p in candidates:
        try:
            metrics = _extract_metrics_from_csv(p)
            if metrics:
                return metrics
        except Exception:
            continue
    return {}


def _metric_value(metrics: dict, key: str) -> float | None:
    k = key.lower()
    alias = {
        "sharpe": ["sharpe", "sharpe_ratio"],
        "ir": ["information_ratio", "ir"],
        "ann_return": ["annualized_return", "annualized_return_rate"],
        "mdd": ["max_drawdown", "max_drawdown_rate"],
    }
    keys = alias.get(k, [key])
    for kk in keys:
        if kk in metrics:
            try:
                return float(metrics[kk])
            except Exception:
                return None
    return None


def _run_baseline(cfg: dict) -> dict:
    ds = cfg.get("data_source", {})
    db_path = str(ds.get("sqlite_path", "")).strip()
    if not db_path:
        return {}
    import os
    os.environ["MOMENTUM_DB_URL"] = f"sqlite:///{db_path}"
    from fastapi.testclient import TestClient
    from etf_momentum.app import app

    backtest = cfg["backtest"]
    automl = cfg["automl"]
    wd = int(automl.get("baseline_anchor_weekday", backtest.get("rebalance_weekday", 0)))
    payload = {
        "start": str(backtest["start"]).replace("-", ""),
        "end": str(backtest["end"]).replace("-", ""),
        "anchor_weekday": wd,
    }
    c = TestClient(app)
    r = c.post("/api/analysis/rotation/weekly5-open", json=payload)
    if r.status_code != 200:
        return {"error": f"status={r.status_code}", "detail": r.text}
    data = r.json()
    by_anchor = data.get("by_anchor", {}).get(str(wd), {})
    metrics = (by_anchor.get("metrics") or {}).get("strategy") or {}
    return metrics


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _write_html(df: pd.DataFrame, path: Path, title: str) -> None:
    html = df.to_html(index=False, escape=True)
    out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .muted {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="muted">Generated by qlib_workflow/automl_pipeline.py</div>
  {html}
</body>
</html>
"""
    path.write_text(out, encoding="utf-8")


def _write_png_ranking(df: pd.DataFrame, path: Path, metric: str) -> None:
    if df.empty:
        return
    top = df.head(10).copy()
    labels = top.apply(lambda r: f"{r['window']}|{r['handler']}|{r['model']}|{r['strategy']}", axis=1)
    scores = pd.to_numeric(top["score"], errors="coerce").fillna(0.0)
    plt.figure(figsize=(10, 4))
    plt.bar(labels, scores)
    plt.title(f"Top-10 by {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_png_compare(baseline: dict, best: dict, path: Path) -> None:
    b_sharpe = baseline.get("sharpe_ratio")
    b_cum = baseline.get("cumulative_return")
    a_sharpe = best.get("score")
    a_cum = best.get("cumulative_return")
    labels = ["Sharpe", "Cumulative"]
    b_vals = [b_sharpe if b_sharpe is not None else 0.0, b_cum if b_cum is not None else 0.0]
    a_vals = [a_sharpe if a_sharpe is not None else 0.0, a_cum if a_cum is not None else 0.0]
    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar([i - 0.2 for i in x], b_vals, width=0.4, label="Baseline")
    plt.bar([i + 0.2 for i in x], a_vals, width=0.4, label="Best AutoML")
    plt.xticks(list(x), labels)
    plt.title("Baseline vs Best")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_summary_md(out_dir: Path, metric: str, best: dict | None) -> None:
    lines = [
        "# AutoML Summary",
        "",
        f"- Metric: `{metric}`",
        f"- Best run: `{best['recorder_name']}`" if best else "- Best run: n/a",
        "",
        "## Artifacts",
        "- `runs.json`",
        "- `runs.csv`",
        "- `runs.html`",
        "- `best.json`",
        "- `best_config.json`",
        "- `baseline.csv`",
        "- `compare.csv`",
        "- `compare.html`",
        "- `ranking.png`",
        "- `compare.png`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_best_config(best: dict, path: Path) -> None:
    cfg = {
        "window": best.get("window"),
        "handler": best.get("handler"),
        "model": best.get("model"),
        "strategy": best.get("strategy"),
        "recorder_name": best.get("recorder_name"),
        "metric": best.get("metric"),
        "score": best.get("score"),
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    args = parser.parse_args()

    cfg = _load_toml(Path(args.config))
    init_qlib(QlibConfig(**cfg["qlib"]))

    universe = list(cfg["universe"]["symbols"])
    automl = cfg["automl"]
    segments_default = automl.get("segments") or cfg.get("ml_pipeline", {}).get("segments", {})
    if not segments_default and not automl.get("windows"):
        raise ValueError("segments not found in automl.segments/ml_pipeline or automl.windows")

    handlers = automl["handlers"]
    models = automl["models"]
    strategies = automl["strategies"]
    metric_key = str(automl.get("metric", "sharpe"))

    results = []
    windows = automl.get("windows") or [{"name": "default", **segments_default}]

    for w in windows:
        segments = {"train": w["train"], "valid": w["valid"], "test": w["test"]}
        for h, m, s in itertools.product(handlers, models, strategies):
            handler_kwargs = dict((h.get("kwargs") or {}))
            handler_kwargs["instruments"] = universe
            handler_cfg = {
                "class": h["class"],
                "module_path": h["module_path"],
                "kwargs": handler_kwargs,
            }
            dataset_cfg = build_dataset(handler_cfg, segments=segments)
            model_cfg = {
                "class": m["class"],
                "module_path": m["module_path"],
                "kwargs": m.get("kwargs", {}),
            }
            task = {"model": model_cfg, "dataset": dataset_cfg, "strategy": s, "window": w["name"]}

            recorder_name = f"{automl['recorder_prefix']}_{w['name']}_{h['name']}_{m['name']}_{s['name']}"
            status = "success"
            err_msg = ""
            metrics = {}
            try:
                model = init_instance_by_config(model_cfg)
                dataset = init_instance_by_config(dataset_cfg)

                with R.start(experiment_name=automl["experiment_name"], recorder_name=recorder_name):
                    R.log_params(**_flatten(task))
                    model.fit(dataset)
                    recorder = R.get_recorder()

                    sr = SignalRecord(model, dataset, recorder)
                    sr.generate()
                    signal = recorder.load_object("pred.pkl")

                    port_cfg = build_port_config(cfg, signal=signal, strategy=s)
                    pr = PortAnaRecord(recorder, config=port_cfg)
                    pr.generate()
                    metrics = _find_metrics(recorder)
                    if metrics:
                        R.log_metrics(**{f"report_{k}": v for k, v in metrics.items()})
            except Exception as e:  # pylint: disable=broad-exception-caught
                status = "failed"
                err_msg = str(e)

            score = _metric_value(metrics, metric_key)
            results.append(
                {
                    "window": w["name"],
                    "handler": h["name"],
                    "model": m["name"],
                    "strategy": s["name"],
                    "recorder_name": recorder_name,
                    "metric": metric_key,
                    "score": score,
                    "status": status,
                    "error": err_msg,
                }
            )

    out_dir = Path(__file__).with_name("outputs") / "automl_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runs.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_runs = [r for r in results if r.get("status") == "success" and r.get("score") is not None]
    if ok_runs:
        ok_runs.sort(key=lambda r: float(r.get("score") or -1e18), reverse=True)
        (out_dir / "best.json").write_text(json.dumps(ok_runs[0], ensure_ascii=False, indent=2), encoding="utf-8")
        _write_best_config(ok_runs[0], out_dir / "best_config.json")

    # Ranking table (CSV + HTML)
    ranked = sorted(ok_runs, key=lambda r: float(r.get("score") or -1e18), reverse=True)
    _write_csv(ranked, out_dir / "runs.csv")
    if ranked:
        _write_html(pd.DataFrame(ranked), out_dir / "runs.html", title="AutoML Ranking")
        _write_png_ranking(pd.DataFrame(ranked), out_dir / "ranking.png", metric=metric_key)

    # Baseline comparison
    baseline = _run_baseline(cfg)
    if baseline:
        _write_csv([baseline], out_dir / "baseline.csv")
        if ranked:
            best = ranked[0]
            compare = {
                "baseline_sharpe": baseline.get("sharpe_ratio"),
                "best_sharpe": best.get("score"),
                "baseline_cum": baseline.get("cumulative_return"),
                "best_cum": best.get("cumulative_return"),
            }
            _write_csv([compare], out_dir / "compare.csv")
            _write_html(pd.DataFrame([compare]), out_dir / "compare.html", title="Baseline vs Best")
            _write_png_compare(baseline, best, out_dir / "compare.png")

    _write_summary_md(out_dir, metric_key, ranked[0] if ranked else None)
    print(f"Automl finished. Run summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
