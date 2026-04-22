from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .baseline import load_close_prices


@dataclass(frozen=True)
class AssetGroupSuggestInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    adjust: str = "hfq"
    lookback_days: int = 252
    corr_threshold: float = 0.75


def suggest_asset_groups(db: Session, inp: AssetGroupSuggestInputs) -> dict[str, Any]:
    codes = list(dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()]))
    if len(codes) < 2:
        raise ValueError("codes must have at least 2 assets")
    if int(inp.lookback_days) < 20:
        raise ValueError("lookback_days must be >= 20")
    thr = float(inp.corr_threshold)
    if (not np.isfinite(thr)) or thr < 0.0 or thr >= 1.0:
        raise ValueError("corr_threshold must be in [0,1)")

    close = load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust=str(inp.adjust or "hfq"))
    if close.empty:
        raise ValueError("no price data in selected range")
    close = close.sort_index().ffill()
    ret = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    if int(inp.lookback_days) > 0 and len(ret) > int(inp.lookback_days):
        ret = ret.iloc[-int(inp.lookback_days) :]
    ret = ret.dropna(how="all")
    if ret.empty:
        raise ValueError("insufficient returns after cleaning")
    corr = ret[codes].corr().fillna(0.0).astype(float)

    # Union-find by absolute correlation threshold.
    parent = {c: c for c in codes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    links: list[dict[str, Any]] = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            v = float(corr.loc[a, b]) if (a in corr.index and b in corr.columns) else 0.0
            if abs(v) >= thr:
                union(a, b)
                links.append({"a": a, "b": b, "corr": v})

    groups: dict[str, list[str]] = {}
    for c in codes:
        root = find(c)
        groups.setdefault(root, []).append(c)
    groups = {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}

    mapping: dict[str, str] = {}
    stability: dict[str, float] = {}
    for i, (_, members) in enumerate(groups.items(), start=1):
        gid = f"G{i:02d}"
        sub = corr.loc[members, members] if len(members) > 1 else pd.DataFrame([[1.0]], index=members, columns=members)
        vals = []
        for x in members:
            for y in members:
                if x >= y:
                    continue
                vals.append(abs(float(sub.loc[x, y])))
        score = float(np.mean(vals)) if vals else 1.0
        stability[gid] = score
        for c in members:
            mapping[c] = gid

    return {
        "meta": {
            "type": "asset_group_suggestion",
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "adjust": str(inp.adjust or "hfq"),
            "lookback_days": int(inp.lookback_days),
            "corr_threshold": float(inp.corr_threshold),
        },
        "asset_groups": mapping,
        "groups": [{"group_id": gid, "members": [c for c, g in mapping.items() if g == gid], "stability": float(stability.get(gid, 0.0))} for gid in sorted(set(mapping.values()))],
        "links": sorted(links, key=lambda x: abs(float(x["corr"])), reverse=True)[:300],
        "corr_matrix": {c: {k: float(corr.loc[c, k]) for k in codes} for c in codes},
    }
