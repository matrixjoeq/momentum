from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import qlib
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


def build_task(cfg: dict, universe: list[str]) -> dict:
    ml = cfg["ml_pipeline"]
    handler = ml["handler"]
    handler_kwargs = dict(handler.get("kwargs", {}))
    handler_kwargs["instruments"] = universe
    handler_cfg = {
        "class": handler["class"],
        "module_path": handler["module_path"],
        "kwargs": handler_kwargs,
    }

    segments = ml["segments"]
    dataset_cfg = {
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

    model_cfg = {
        "class": ml["model"]["class"],
        "module_path": ml["model"]["module_path"],
        "kwargs": ml["model"].get("kwargs", {}),
    }

    return {"model": model_cfg, "dataset": dataset_cfg}


def build_port_config(cfg: dict, signal) -> dict:
    ml = cfg["ml_pipeline"]
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
                "topk": int(ml["topk"]),
                "n_drop": int(ml["drop"]),
            },
        },
        "backtest": {
            "start_time": backtest["start"],
            "end_time": backtest["end"],
            "account": float(ml["account"]),
            "benchmark": ml["benchmark"],
            "exchange_kwargs": {"deal_price": "close", "limit_threshold": 0.095},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    args = parser.parse_args()

    cfg = _load_toml(Path(args.config))
    init_qlib(QlibConfig(**cfg["qlib"]))

    universe = list(cfg["universe"]["symbols"])
    task = build_task(cfg, universe=universe)

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    ml = cfg["ml_pipeline"]
    with R.start(experiment_name=ml["experiment_name"], recorder_name=ml["recorder_name"]):
        R.log_params(**task)
        model.fit(dataset)
        recorder = R.get_recorder()

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        signal = recorder.load_object("pred.pkl")

        port_cfg = build_port_config(cfg, signal=signal)
        pr = PortAnaRecord(recorder, config=port_cfg)
        pr.generate()

    print("ML pipeline finished. Reports saved in Qlib recorder artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
