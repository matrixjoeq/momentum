import datetime as dt

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _count_trade_periods(out: dict) -> int:
    rows = out.get("period_details") or []
    n = 0
    for r in rows:
        if (r.get("buys") or []) or (r.get("sells") or []):
            n += 1
    return n


def test_rotation_inertia_reduces_trade_frequency(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(40)]

        # Alternate daily returns so that Top1 flips frequently when lookback=1 and rebalance=daily.
        # AAA: up on even i, down on odd i; BBB does the opposite.
        px_a = 100.0
        px_b = 100.0
        for i, d in enumerate(dates):
            if i == 0:
                pass
            elif i % 2 == 0:
                px_a *= 1.01
                px_b *= 0.99
            else:
                px_a *= 0.99
                px_b *= 1.01

            for adj in ("hfq", "none"):
                db.add(EtfPrice(code="AAA", trade_date=d, close=float(px_a), source="eastmoney", adjust=adj))
                db.add(EtfPrice(code="BBB", trade_date=d, close=float(px_b), source="eastmoney", adjust=adj))
        db.commit()

        base_inp = RotationInputs(
            codes=codes,
            start=start,
            end=dates[-1],
            rebalance="daily",
            top_k=1,
            lookback_days=1,
            skip_days=0,
            risk_off=False,
            cost_bps=0.0,
        )

        out_no = backtest_rotation(db, base_inp)
        dct = dict(base_inp.__dict__)
        dct["inertia"] = True
        dct["inertia_min_hold_periods"] = 5
        out_in = backtest_rotation(
            db,
            RotationInputs(**dct),
        )

    assert _count_trade_periods(out_in) < _count_trade_periods(out_no)

