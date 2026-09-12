"""
Market data layer for the Neil-style scanner.

Public, key-less endpoints only.  Sources (in priority order):
  1. OKX   USDT-margined perpetual swaps  (best data quality)
  2. MEXC  USDT-margined futures          (widest alt listing)
  3. Gate  USDT-margined futures          (fallback)

Every function returns pandas DataFrames with columns
    ts (UTC datetime, bar open), open, high, low, close, volume
oldest -> newest.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("data")

UA = {"User-Agent": "Mozilla/5.0 (neil-style-scanner; public market data)"}
TIMEOUT = 20
RETRIES = 3

# Symbols that are not tradeable coins (stables, fiat-ish, leveraged tokens)
EXCLUDE_BASES = {
    "USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDE", "USD1", "PYUSD", "BUSD", "EUR", "GBP", "USDD",
    "BTCDOM", "DEFI", "SHIB1000", "1000SHIB",
    # commodities / FX / stock-index perps that MEXC lists alongside crypto
    "UKOIL", "USOIL", "BZ", "XAU", "XAG", "XAUT", "PAXG", "XPT", "XPD", "NATGAS", "COPPER",
    "SPY", "NDX", "NAS100", "US500", "US30", "DJI", "QQQ", "TSLA", "NVDA", "AAPL", "MSFT",
    "AMZN", "GOOGL", "META", "COIN", "MSTR", "HOOD", "CRCL", "GME", "AMC", "PLTR", "AMD", "INTC",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "JPY", "CHF", "AUD", "CAD",
}


def _get(url: str, params: dict | None = None) -> Optional[dict | list]:
    for i in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            if r.status_code != 200:
                log.debug("GET %s -> %s", r.url, r.status_code)
                return None
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("GET %s failed: %s", url, e)
            time.sleep(0.8 * (i + 1))
    return None


@dataclass
class Instrument:
    base: str          # e.g. "INIT"
    source: str        # okx | mexc | gate
    inst_id: str       # exchange-specific id
    last: float
    usd_vol_24h: float
    change_24h: float  # fraction, e.g. 0.055


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def okx_universe() -> dict[str, Instrument]:
    out: dict[str, Instrument] = {}
    j = _get("https://www.okx.com/api/v5/market/tickers", {"instType": "SWAP"})
    if not j or j.get("code") != "0":
        log.warning("OKX tickers unavailable")
        return out
    for t in j["data"]:
        inst = t["instId"]
        if not inst.endswith("-USDT-SWAP"):
            continue
        base = inst.split("-")[0]
        try:
            last = float(t["last"])
            vol_usd = float(t["volCcy24h"]) * last
            open24 = float(t["open24h"])
            chg = last / open24 - 1 if open24 else 0.0
        except (ValueError, KeyError):
            continue
        out[base] = Instrument(base, "okx", inst, last, vol_usd, chg)
    return out


def mexc_universe() -> dict[str, Instrument]:
    out: dict[str, Instrument] = {}
    j = _get("https://contract.mexc.com/api/v1/contract/ticker")
    if not j or not j.get("success"):
        log.warning("MEXC tickers unavailable")
        return out
    for t in j["data"]:
        sym = t.get("symbol", "")
        if not sym.endswith("_USDT"):
            continue
        base = sym[:-5]
        try:
            out[base] = Instrument(base, "mexc", sym, float(t["lastPrice"]),
                                   float(t.get("amount24", 0)), float(t.get("riseFallRate", 0)))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def gate_universe() -> dict[str, Instrument]:
    out: dict[str, Instrument] = {}
    j = _get("https://api.gateio.ws/api/v4/futures/usdt/tickers")
    if not j or not isinstance(j, list):
        log.warning("Gate tickers unavailable")
        return out
    for t in j:
        c = t.get("contract", "")
        if not c.endswith("_USDT"):
            continue
        base = c[:-5]
        try:
            out[base] = Instrument(base, "gate", c, float(t["last"]),
                                   float(t.get("volume_24h_settle", 0)),
                                   float(t.get("change_percentage", 0)) / 100.0)
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _norm_base(base: str) -> str:
    """Map 1000PEPE / kPEPE style names onto the plain coin name."""
    b = base.upper()
    for pref in ("1000000", "100000", "10000", "1000"):
        if b.startswith(pref) and len(b) > len(pref):
            return b[len(pref):]
    if b.startswith("K") and len(b) > 4 and b[1:] in {"PEPE", "BONK", "FLOKI", "SHIB", "LUNC", "DOGS", "NEIRO"}:
        return b[1:]
    return b


def build_universe(min_usd_vol: float = 5e6, extra: list[str] | None = None) -> dict[str, Instrument]:
    """
    Union of all sources.  For each coin keep the most liquid listing.
    Coins below `min_usd_vol` on every source are dropped unless in `extra`.
    """
    merged: dict[str, Instrument] = {}
    for fetch in (okx_universe, mexc_universe, gate_universe):
        try:
            src = fetch()
        except Exception as e:  # noqa: BLE001
            log.warning("universe source failed: %s", e)
            continue
        for base, inst in src.items():
            nb = _norm_base(base)
            if nb in EXCLUDE_BASES:
                continue
            cur = merged.get(nb)
            if cur is None or inst.usd_vol_24h > cur.usd_vol_24h:
                inst.base = nb
                merged[nb] = inst
        time.sleep(0.3)
    extra = set(x.upper() for x in (extra or []))
    return {b: i for b, i in merged.items() if i.usd_vol_24h >= min_usd_vol or b in extra}


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
_OKX_BAR = {"1h": "1H", "4h": "4H", "1d": "1Dutc"}
_MEXC_BAR = {"1h": "Min60", "4h": "Hour4", "1d": "Day1"}
_GATE_BAR = {"1h": "1h", "4h": "4h", "1d": "1d"}
_SECS = {"1h": 3600, "4h": 14400, "1d": 86400}


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def okx_candles(inst_id: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    rows: list[tuple] = []
    after = None
    while len(rows) < limit:
        params = {"instId": inst_id, "bar": _OKX_BAR[tf], "limit": str(min(300, limit - len(rows)))}
        if after:
            params["after"] = str(after)
        j = _get("https://www.okx.com/api/v5/market/candles", params)
        if not j or j.get("code") != "0" or not j.get("data"):
            break
        batch = j["data"]  # newest first
        for r in batch:
            # ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm
            rows.append((int(r[0]) // 1000, r[1], r[2], r[3], r[4], r[6]))
        after = int(batch[-1][0])
        if len(batch) < 100:
            break
        time.sleep(0.12)
    return _frame(rows) if rows else None


def mexc_candles(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    end = int(time.time())
    start = end - _SECS[tf] * (limit + 2)
    j = _get(f"https://contract.mexc.com/api/v1/contract/kline/{symbol}",
             {"interval": _MEXC_BAR[tf], "start": start, "end": end})
    if not j or not j.get("success") or not j.get("data") or not j["data"].get("time"):
        return None
    d = j["data"]
    rows = list(zip(d["time"], d["open"], d["high"], d["low"], d["close"], d.get("vol", [0] * len(d["time"]))))
    return _frame(rows)


def gate_candles(contract: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    j = _get("https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
             {"contract": contract, "interval": _GATE_BAR[tf], "limit": str(limit)})
    if not j or not isinstance(j, list) or not j:
        return None
    rows = [(int(r["t"]), r["o"], r["h"], r["l"], r["c"], r.get("v", 0)) for r in j]
    return _frame(rows)


def candles(inst: Instrument, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch candles for an instrument, trying its own source first, then the others."""
    order = {
        "okx": [("okx", inst.inst_id), ("mexc", f"{inst.base}_USDT"), ("gate", f"{inst.base}_USDT")],
        "mexc": [("mexc", inst.inst_id), ("okx", f"{inst.base}-USDT-SWAP"), ("gate", f"{inst.base}_USDT")],
        "gate": [("gate", inst.inst_id), ("mexc", f"{inst.base}_USDT"), ("okx", f"{inst.base}-USDT-SWAP")],
    }[inst.source]
    for src, sid in order:
        try:
            df = {"okx": okx_candles, "mexc": mexc_candles, "gate": gate_candles}[src](sid, tf, limit)
        except Exception as e:  # noqa: BLE001
            log.debug("%s %s %s failed: %s", src, sid, tf, e)
            df = None
        if df is not None and len(df) >= 60:
            df.attrs["source"] = src
            return df
        time.sleep(0.1)
    return None
