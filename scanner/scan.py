#!/usr/bin/env python3
"""
Daily Neil-style setup scanner.

    python scanner/scan.py                 # live scan (OKX / MEXC / Gate public APIs)
    python scanner/scan.py --fixtures DIR  # offline run on saved candles (tests)

Writes:
    output/latest.json            top setups + market context (machine readable)
    output/report.md              Neil-format alert text + short analysis per setup
    output/charts/<date>_<SYM>.png
    output/history/<date>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import setups as S  # noqa: E402
import chart as C  # noqa: E402

log = logging.getLogger("scan")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# Coins Neil trades repeatedly — always scanned even if they fall under the liquidity floor.
NEIL_FAVOURITES = [
    "INIT", "LIT", "HYPE", "FARTCOIN", "ETH", "TAO", "VVV", "ZEC", "SWARMS", "SOL", "HIGH", "BIO", "JTO",
    "XMR", "ZRO", "HEMI", "PEPE", "SKR", "TIA", "WLD", "AAVE", "EIGEN", "ENA", "ETHFI", "BTC", "XRP", "DOGE",
    "SEI", "NEAR", "PUMP", "ONDO", "KAITO", "ARB", "SAGA", "PIXEL", "TUT", "INJ", "AVNT", "XPL", "ASTER",
    "HBAR", "TRX", "QNT", "FET", "SYRUP", "UNI", "TON", "GRASS", "PENGU", "ME", "CETUS", "NOT", "PARTI",
]
MAX_PER_DAY = 3


def load_fixture(dirpath: Path, sym: str, tf: str):
    f = dirpath / f"{sym}_{tf}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.attrs["source"] = "fixture"
    return df


def fetch_pack(inst: D.Instrument, btc4: pd.DataFrame, fixtures: Path | None):
    if fixtures:
        df1, df4, df1d = (load_fixture(fixtures, inst.base, tf) for tf in ("1h", "4h", "1d"))
    else:
        df4 = D.candles(inst, "4h", 300)
        if df4 is None:
            return None
        df1 = D.candles(inst, "1h", 300)
        df1d = D.candles(inst, "1d", 250)
        time.sleep(0.15)
    if df1 is None or df4 is None:
        return None
    return S.Pack(inst.base, df1, df4, df1d, btc4, inst.usd_vol_24h)


def neil_alert(s: dict) -> str:
    """Write the call the way Neil words his alerts."""
    tps = " · ".join(f"TP{i+1} {C._fmt(t)} ({r:.1f}R)" for i, (t, r) in enumerate(zip(s["tps"], s["tp_r"])))
    dca = f"DCA @ {C._fmt(s['dca'])}, " if s.get("dca") else "No DCA, "
    tf = s["stop_tf"]
    flavour = {
        "sr_flip": "SR flip trade — level flipped and holding, looking for continuation higher.",
        "ma_retest": "First retest of the 200MA after the push — think we get a bounce.",
        "range_low": "Bottom side of the range, higher low in — playing the rotation back to range highs.",
        "compression": "4H compression should break to the upside imo — aiming to hit this breakout.",
        "daily_breakout": "Daily range breakout — not missing this one, still looks good.",
    }[s["kind"]]
    return (f"Market long {s['symbol']} here at CMP ({C._fmt(s['entry'])}). {dca}{tf} close under "
            f"{C._fmt(s['stop'])} for stops, TPs above — {tps}. Once TP1 hits, stops to BE. {flavour}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=None)
    ap.add_argument("--max-symbols", type=int, default=400)
    ap.add_argument("--min-vol", type=float, default=5e6)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.v else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    (OUT / "history").mkdir(parents=True, exist_ok=True)

    # ---- universe ----------------------------------------------------------
    if a.fixtures:
        syms = sorted({f.name.split("_")[0] for f in a.fixtures.glob("*_4h.csv")})
        universe = {s: D.Instrument(s, "fixture", s, 0.0, 2e7, 0.0) for s in syms}
    else:
        universe = D.build_universe(a.min_vol, extra=NEIL_FAVOURITES)
    log.info("universe: %d coins", len(universe))
    if "BTC" not in universe:
        universe["BTC"] = D.Instrument("BTC", "okx", "BTC-USDT-SWAP", 0, 1e9, 0)

    # ---- BTC context -------------------------------------------------------
    btc_inst = universe["BTC"]
    if a.fixtures:
        btc4, btc1d = load_fixture(a.fixtures, "BTC", "4h"), load_fixture(a.fixtures, "BTC", "1d")
    else:
        btc4, btc1d = D.candles(btc_inst, "4h", 300), D.candles(btc_inst, "1d", 250)
    if btc4 is None:
        log.error("no BTC data — aborting")
        sys.exit(2)
    ctx = S.btc_context(btc4, btc1d)
    log.info("BTC context: %s (%s)", ctx.btc_trend, ctx.btc_note)

    # ---- scan --------------------------------------------------------------
    order = sorted(universe.values(), key=lambda i: -i.usd_vol_24h)[: a.max_symbols]
    found: list[S.Setup] = []
    packs: dict[str, S.Pack] = {}
    scanned = 0
    for k, inst in enumerate(order, 1):
        pack = fetch_pack(inst, btc4, a.fixtures)
        if pack is None:
            log.debug("no data for %s", inst.base)
            continue
        scanned += 1
        hits = S.scan_pack(pack, ctx)
        if hits:
            packs[inst.base] = pack
            found += hits
        if k % 25 == 0:
            log.info("… %d/%d scanned, %d setups so far", k, len(order), len(found))
    log.info("scanned %d coins, %d raw setups", scanned, len(found))

    # ---- rank: best setup per coin, then prefer distinct setup types --------
    best_per_sym: dict[str, S.Setup] = {}
    for s in found:
        if s.symbol not in best_per_sym or s.score > best_per_sym[s.symbol].score:
            best_per_sym[s.symbol] = s
    ranked = sorted(best_per_sym.values(), key=lambda s: -s.score)
    picks: list[S.Setup] = []
    for s in ranked:
        if len(picks) >= MAX_PER_DAY:
            break
        same_kind = sum(1 for p in picks if p.kind == s.kind)
        if same_kind >= 2:
            continue
        picks.append(s)
    if len(picks) < MAX_PER_DAY:                      # top up if diversity rule left gaps
        for s in ranked:
            if s not in picks and len(picks) < MAX_PER_DAY:
                picks.append(s)

    # ---- charts + outputs --------------------------------------------------
    results = []
    for s in picks:
        d = s.to_dict()
        pack = packs[s.symbol]
        df = pack.df4 if s.chart_tf == "4h" else pack.df1
        chart = OUT / "charts" / f"{today}_{s.symbol}.png"
        C.render(d, df, str(chart), s.chart_tf, s.source)
        d["chart"] = str(chart.relative_to(ROOT))
        d["alert"] = neil_alert(d)
        results.append(d)

    watch = [{"symbol": s.symbol, "kind": s.kind, "score": round(s.score, 2), "entry": s.entry, "stop": s.stop}
             for s in ranked[MAX_PER_DAY:MAX_PER_DAY + 7]]
    payload = {
        "date": today,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "scanned": scanned,
        "btc": {"trend": ctx.btc_trend, "note": ctx.btc_note, "above_4h_200": ctx.btc_4h_above_200,
                "change_3d_pct": round(ctx.btc_3d_change * 100, 2), "last": float(btc4["close"].iloc[-1])},
        "setups": results,
        "watchlist": watch,
    }
    (OUT / "latest.json").write_text(json.dumps(payload, indent=2))
    (OUT / "history" / f"{today}.json").write_text(json.dumps(payload, indent=2))

    lines = [f"# Neil-style setups — {today}", "",
             f"**BTC context:** {ctx.btc_note} (3d {ctx.btc_3d_change*100:+.1f}%)", "",
             f"Scanned {scanned} USDT perps; {len(found)} raw setups; showing top {len(results)}.", ""]
    for i, d in enumerate(results, 1):
        lines += [f"## {i}. {d['symbol']} — {d['label']}  (score {d['score']:.1f})", "",
                  f"> {d['alert']}", "",
                  f"Risk to stop: {d['risk_pct']:.2f}% · TP1 {d['tp_r'][0]:.1f}R · TP2 {d['tp_r'][1]:.1f}R · TP3 {d['tp_r'][2]:.1f}R",
                  "", "Why it qualifies:"] + [f"- {f}" for f in d["facts"]] + ["", f"![chart]({d['chart']})", ""]
    if watch:
        lines += ["## Watchlist (next in line)", ""] + [f"- {w['symbol']} — {w['kind']} (score {w['score']})" for w in watch]
    (OUT / "report.md").write_text("\n".join(lines))
    log.info("wrote %d setups to %s", len(results), OUT)


if __name__ == "__main__":
    main()
