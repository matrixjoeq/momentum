from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


def _load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


DUMP_BIN_URLS = [
    "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py",
    "https://fastly.jsdelivr.net/gh/microsoft/qlib@main/scripts/dump_bin.py",
    "https://cdn.jsdelivr.net/gh/microsoft/qlib@main/scripts/dump_bin.py",
]


def _find_dump_bin() -> Path | None:
    try:
        import qlib  # type: ignore
    except Exception:
        return None
    root = Path(qlib.__file__).resolve().parent
    for p in root.rglob("dump_bin.py"):
        return p
    return None


def _download_dump_bin(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in DUMP_BIN_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            dest.write_text(content, encoding="utf-8")
            return dest
        except Exception as err:  # pragma: no cover - network dependent
            last_err = err
            continue
    raise RuntimeError(f"failed to download dump_bin.py: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    args = parser.parse_args()

    cfg = _load_toml(Path(args.config))
    ds = cfg.get("data_source", {})
    csv_dir = str(ds.get("csv_dir", "")).strip()
    qlib_dir = str(ds.get("qlib_dir", "")).strip()
    if not csv_dir or not qlib_dir:
        raise ValueError("data_source.csv_dir and data_source.qlib_dir are required")

    dump_bin = _find_dump_bin()
    if dump_bin is None:
        dump_bin = _download_dump_bin(Path(__file__).with_name("vendor") / "dump_bin.py")

    cmd = [
        sys.executable,
        str(dump_bin),
        "dump_all",
        "--data_path",
        csv_dir,
        "--qlib_dir",
        qlib_dir,
        "--include_fields",
        "open,high,low,close,volume,factor",
        "--date_field_name",
        "date",
        "--file_suffix",
        ".csv",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
