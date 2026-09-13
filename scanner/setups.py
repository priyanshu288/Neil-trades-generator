"""
Setup detection modelled on TraderNeil's calls (#neil-calls, Dec 2025 - Sep 2026).

What his entries have in common (173 leverage entries analysed):
  * ~99% longs, entered at market ("CMP"), sometimes with ONE DCA a few % lower.
  * Soft stop = candle CLOSE under a level.  4H close (50%), 1H close (28%),
    15/30-min close (15%) for scalps, 12H/daily for swings.  Stop distance is tight (1-5%).
  * TPs are prior highs / resistance levels drawn on the chart.  TP1 ~1.5-2.5R
    -> stops to breakeven, TP2 3-5R, TP3+ 5-12R with a runner.
  * Setup vocabulary: "SR flip / reclaim / retest", "first retest of the 200MA",
    "range low reclaim / higher low / range play", "4H compression / bull flag / breakout",
    "daily range breakout", "bull divs at daily support".
  * Context: coin must hold its LTF trend (above 15m/1H/4H 200MA), show relative
    strength vs BTC, and BTC must not be actively breaking down.

Each detector returns a Setup or None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def pivots(df: pd.DataFrame, left: int = 4, right: int = 4):
    """Return (pivot_high_idx, pivot_low_idx) lists of integer positions."""
    h, l = df["high"].values, df["low"].values
    n = len(df)
    ph, pl = [], []
    for i in range(left, n - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            ph.append(i)
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            pl.append(i)
    return ph, pl


def nice(x: float, direction: str = "down") -> float:
    """Round a price to ~4 significant digits the way a human marks a level."""
    if x <= 0 or not math.isfinite(x):
        return x
    mag = 10 ** (math.floor(math.log10(x)) - 3)
    v = math.floor(x / mag) * mag if direction == "down" else math.ceil(x / mag) * mag
    return float(f"{v:.10g}")


def cluster_levels(levels: list[float], tol: float = 0.012) -> list[float]:
    """Merge price levels closer than `tol` (fraction) into their mean."""
    if not levels:
        return []
    levels = sorted(levels)
    out, cur = [], [levels[0]]
    for x in levels[1:]:
        if x / cur[-1] - 1 <= tol:
            cur.append(x)
        else:
            out.append(float(np.mean(cur)))
            cur = [x]
    out.append(float(np.mean(cur)))
    return out


@dataclass
class Setup:
    symbol: str
    kind: str                 # sr_flip | ma_retest | range_low | compression | daily_breakout
    label: str                # human name Neil would use
    entry: float
    stop: float
    stop_tf: str              # "4H" | "1H" | "15min" | "12H"
    dca: Optional[float]
    tps: list[float]
    tp_r: list[float]
    risk_pct: float
    score: float
    facts: list[str] = field(default_factory=list)   # objective observations for the write-up
    chart_tf: str = "4h"
    level: Optional[float] = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["risk_pct"] = round(self.risk_pct * 100, 2)
        return d


# --------------------------------------------------------------------------- #
# take-profit construction
# --------------------------------------------------------------------------- #
def build_tps(entry: float, stop: float, df4: pd.DataFrame, df1d: Optional[pd.DataFrame],
              measured: Optional[float] = None) -> tuple[list[float], list[float]]:
    """
    TPs = prior swing highs / resistance clusters above entry, like the horizontal
    lines Neil draws.  Falls back to synthetic R multiples if structure is missing.
    """
    risk = entry - stop
    if risk <= 0:
        return [], []
    cands: list[float] = []
    ph, _ = pivots(df4, 3, 3)
    cands += [float(df4["high"].iloc[i]) for i in ph[-80:]]
    if df1d is not None and len(df1d) > 20:
        phd, _ = pivots(df1d, 3, 3)
        cands += [float(df1d["high"].iloc[i]) for i in phd[-40:]]
        cands.append(float(df1d["high"].iloc[-60:].max()))
    cands = [x for x in cands if entry * 1.012 < x < entry * 1.8]
    if measured:
        cands.append(measured)
    levels = cluster_levels(cands)

    # Neil's ladder: TP1 ~1.5-2.5R, TP2 ~3-5R, TP3 ~5-12R.  A structural level is used
    # when one sits inside the band; otherwise a synthetic R-multiple fills the slot.
    tps: list[float] = []
    bands = [(1.4, 3.0), (2.8, 6.0), (4.5, 12.0)]
    last_r = 0.0
    for lo_r, hi_r in bands:
        lo_r = max(lo_r, last_r + 1.0)
        nxt = [x for x in levels if lo_r <= (x - entry) / risk <= max(hi_r, lo_r + 1.0)]
        if nxt:
            tp = nice(nxt[0], "down")
        else:
            tp = nice(entry + risk * (lo_r + 0.4), "down")
        if tps and tp <= tps[-1] * 1.01:
            tp = nice(tps[-1] + risk * 1.5, "down")
        tps.append(tp)
        last_r = (tp - entry) / risk
    tp_r = [round((t - entry) / risk, 2) for t in tps]
    return tps, tp_r


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    btc_trend: str          # "up" | "chop" | "down"
    btc_note: str
    btc_4h_above_200: bool
    btc_3d_change: float


def btc_context(btc4: pd.DataFrame, btc1d: Optional[pd.DataFrame]) -> Context:
    c = btc4["close"]
    s50, s200 = sma(c, 50), sma(c, 200)
    last = float(c.iloc[-1])
    above200 = bool(last > s200.iloc[-1]) if not np.isnan(s200.iloc[-1]) else True
    above50 = bool(last > s50.iloc[-1]) if not np.isnan(s50.iloc[-1]) else True
    chg3d = last / float(c.iloc[-19]) - 1 if len(c) > 19 else 0.0
    lo20 = float(btc4["low"].iloc[-30:-1].min())
    if above50 and above200 and chg3d > -0.02:
        trend, note = "up", "BTC above its 4H 50/200MA — alts have a tailwind."
    elif (not above50) and (last < lo20 * 1.005 or chg3d < -0.05):
        trend, note = "down", "BTC below its 4H 50MA and pressing recent lows — Neil sits on hands here; scalp-size only."
    else:
        trend, note = "chop", "BTC chopping around its 4H MAs — no fresh breakdown, but keep size normal and stops honest."
    return Context(trend, note, above200, chg3d)


# --------------------------------------------------------------------------- #
# per-symbol feature pack
# --------------------------------------------------------------------------- #
@dataclass
class Pack:
    symbol: str
    df1: pd.DataFrame
    df4: pd.DataFrame
    df1d: Optional[pd.DataFrame]
    btc4: pd.DataFrame
    usd_vol: float

    def __post_init__(self):
        for df in (self.df1, self.df4):
            df["sma20"] = sma(df["close"], 20)
            df["sma50"] = sma(df["close"], 50)
            df["sma200"] = sma(df["close"], 200)
            df["atr"] = atr(df)
        if self.df1d is not None:
            self.df1d["sma200"] = sma(self.df1d["close"], 200)
            self.df1d["sma50"] = sma(self.df1d["close"], 50)
        self.last = float(self.df4["close"].iloc[-1])
        # relative strength vs BTC over 7 days (42 x 4H bars)
        n = 42 if len(self.df4) > 42 and len(self.btc4) > 42 else min(len(self.df4), len(self.btc4)) - 1
        self.ret7 = self.last / float(self.df4["close"].iloc[-n]) - 1
        self.btc7 = float(self.btc4["close"].iloc[-1]) / float(self.btc4["close"].iloc[-n]) - 1
        self.rs7 = self.ret7 - self.btc7

    # trend checks Neil calls out in his alerts
    def above_1h_200(self) -> Optional[bool]:
        v = self.df1["sma200"].iloc[-1]
        return None if np.isnan(v) else bool(self.df1["close"].iloc[-1] > v)

    def above_4h_200(self) -> Optional[bool]:
        v = self.df4["sma200"].iloc[-1]
        return None if np.isnan(v) else bool(self.last > v)

    def above_1d_200(self) -> Optional[bool]:
        if self.df1d is None:
            return None
        v = self.df1d["sma200"].iloc[-1]
        return None if np.isnan(v) else bool(self.last > v)

    def trend_facts(self) -> list[str]:
        f = []
        a1, a4, ad = self.above_1h_200(), self.above_4h_200(), self.above_1d_200()
        tags = [("1H", a1), ("4H", a4), ("daily", ad)]
        above = [t for t, v in tags if v]
        below = [t for t, v in tags if v is False]
        if above:
            f.append("above the " + "/".join(above) + " 200MA")
        if below:
            f.append("still under the " + "/".join(below) + " 200MA")
        f.append(f"7d: {self.ret7*100:+.1f}% vs BTC {self.btc7*100:+.1f}% (RS {self.rs7*100:+.1f}%)")
        if self.ret7 > 0.60:
            f.append(f"already {self.ret7*100:.0f}% up on the week — extended, size down or wait for a deeper pullback")
        return f


def _common_score(p: Pack) -> float:
    s = 0.0
    if p.above_1h_200():
        s += 1.0
    if p.above_4h_200():
        s += 1.2
    if p.above_1d_200():
        s += 0.6
    s += max(-1.5, min(2.0, p.rs7 * 10))           # relative strength
    s += min(1.0, math.log10(max(p.usd_vol, 1e6) / 1e6) * 0.4)  # liquidity
    # Neil buys retests and higher lows, not blow-off tops: fade anything already parabolic
    if p.ret7 > 0.60:
        s -= min(3.0, (p.ret7 - 0.60) * 4)
    return s


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
def detect_sr_flip(p: Pack) -> Optional[Setup]:
    """
    Resistance that price has just closed above and is now retesting from the top.
    Neil: "SR flip trade", "SR retest here playing it for a higher low", "reclaim setup".
    """
    df = p.df4
    n = len(df)
    if n < 120:
        return None
    ph, _ = pivots(df, 4, 4)
    ph = [i for i in ph if n - 130 <= i <= n - 8]
    if not ph:
        return None
    last = p.last
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    best = None
    for i in ph:
        lvl = float(high[i])
        if not (last * 0.985 <= lvl <= last * 0.995 + last * 0.03):   # price 0-4% above level
            continue
        # it acted as resistance: bars between i and the reclaim mostly closed below lvl
        seg = close[i + 1:n - 1]
        if len(seg) < 6:
            continue
        below = (seg < lvl).mean()
        if below < 0.6:
            continue
        # reclaim: first close above lvl within the last 2-18 bars
        recl = [k for k in range(max(i + 1, n - 18), n) if close[k] > lvl]
        if not recl or recl[0] > n - 2:
            continue
        r0 = recl[0]
        # since the reclaim, no 4H close back under the level (setup still valid)
        if (close[r0:] < lvl * 0.995).any():
            continue
        # retest: some low since reclaim came within 1.5% of the level
        touch = (low[r0:] <= lvl * 1.015).any()
        touches_before = int(((high[max(0, i - 60):i] >= lvl * 0.985) & (high[max(0, i - 60):i] <= lvl * 1.015)).sum())
        # a level price never traded into before is not a level — it is just a pivot high
        if touches_before < 1:
            continue
        freshness = n - r0
        sc = 2.0 + (1.0 if touch else 0.0) + min(1.5, touches_before * 0.3) + max(0, 1.2 - freshness * 0.08)
        vol_ratio = float(df["volume"].iloc[r0] / max(df["volume"].iloc[r0 - 20:r0].mean(), 1e-9))
        if vol_ratio > 1.5:
            sc += 0.6
        if best is None or sc > best[0]:
            best = (sc, lvl, touch, touches_before, freshness, vol_ratio)
    if best is None:
        return None
    sc, lvl, touch, touches_before, freshness, vol_ratio = best
    stop = nice(lvl * 0.988, "down")
    risk = (last - stop) / last
    if not (0.008 <= risk <= 0.07):
        return None
    dca = nice((last + lvl) / 2, "down") if last / lvl - 1 > 0.02 else None
    tps, tp_r = build_tps(last, stop, df, p.df1d)
    if not tps or tp_r[0] < 1.2:
        return None
    facts = [
        f"4H closed above the {lvl:.6g} resistance {freshness} bars ago and is holding above it"
        + (", low already retested the level" if touch else " — retest still pending"),
        f"level rejected price {touches_before}x before the flip",
        f"reclaim bar volume {vol_ratio:.1f}x the 20-bar average",
    ] + p.trend_facts()
    return Setup(p.symbol, "sr_flip", "4H SR flip / retest", last, stop, "4H", dca, tps, tp_r, risk,
                 sc + _common_score(p), facts, "4h", lvl)


def detect_ma_retest(p: Pack) -> Optional[Setup]:
    """
    First retest of a rising 200MA after an impulse.  Neil: "first retest of the 15min 200ma",
    "1H 200MA tap", "scalp longing HYPE after this 15min 200ma tap".
    We use the 1H 200MA (4H chart for context).
    """
    df = p.df1
    if len(df) < 220 or np.isnan(df["sma200"].iloc[-1]):
        return None
    ma = df["sma200"]
    close, low = df["close"], df["low"]
    last = float(close.iloc[-1])
    ma_now = float(ma.iloc[-1])
    if last < ma_now * 0.995:
        return None
    rising = ma_now > float(ma.iloc[-24])
    if not rising:
        return None
    # impulse: price was >= 6% above the MA in the last 60 bars
    dist = (close / ma - 1).iloc[-60:]
    if dist.max() < 0.06:
        return None
    # tap: a low within 1.2% of the MA in the last 4 bars, none in the 30 bars before that
    recent = (low.iloc[-4:] <= ma.iloc[-4:] * 1.012).any()
    earlier = (low.iloc[-34:-4] <= ma.iloc[-34:-4] * 1.005).any()
    if not recent or earlier:
        return None
    impulse_high = float(df["high"].iloc[-60:].max())
    stop = nice(ma_now * 0.985, "down")
    risk = (last - stop) / last
    if not (0.008 <= risk <= 0.05):
        return None
    tps, tp_r = build_tps(last, stop, p.df4, p.df1d, measured=impulse_high)
    if not tps or tp_r[0] < 1.2:
        return None
    sc = 2.6 + min(1.5, float(dist.max()) * 10) + (0.5 if last > ma_now * 1.005 else 0)
    facts = [
        f"first tap of the rising 1H 200MA ({ma_now:.6g}) after a {dist.max()*100:.0f}% extension above it",
        f"impulse high at {impulse_high:.6g} is the natural first target",
    ] + p.trend_facts()
    return Setup(p.symbol, "ma_retest", "1H 200MA first retest", last, stop, "1H", None, tps, tp_r, risk,
                 sc + _common_score(p), facts, "1h", ma_now)


def detect_range_low(p: Pack) -> Optional[Setup]:
    """
    Range play from the bottom: higher low forming near range lows with the range still intact.
    Neil: "Bottom side of range here", "range low reclaim", "looking for a higher low".
    """
    df = p.df4
    if len(df) < 200:
        return None
    win = df.iloc[-180:]
    r_hi, r_lo = float(win["high"].max()), float(win["low"].min())
    if r_hi / r_lo - 1 < 0.15:
        return None
    last = p.last
    pos = (last - r_lo) / (r_hi - r_lo)
    if pos > 0.35:
        return None
    _, pl = pivots(df, 4, 4)
    pl = [i for i in pl if i >= len(df) - 90]
    if len(pl) < 2:
        return None
    lo_prev, lo_last = float(df["low"].iloc[pl[-2]]), float(df["low"].iloc[pl[-1]])
    if lo_last <= lo_prev:            # need a higher low
        return None
    if len(df) - pl[-1] > 25:          # higher low must be recent
        return None
    # bounce underway: last close above 4H 20MA or above the higher-low bar's high
    if not (last > float(df["sma20"].iloc[-1]) or last > float(df["high"].iloc[pl[-1]])):
        return None
    stop = nice(lo_last * 0.99, "down")
    risk = (last - stop) / last
    if not (0.01 <= risk <= 0.08):
        return None
    mid = r_lo + (r_hi - r_lo) * 0.5
    vah = r_lo + (r_hi - r_lo) * 0.75
    tps, tp_r = build_tps(last, stop, df, p.df1d, measured=vah)
    if not tps or tp_r[0] < 1.2:
        return None
    dca = nice((last + lo_last) / 2, "down") if last / lo_last - 1 > 0.03 else None
    sc = 2.0 + (1.0 if p.above_4h_200() else 0) + max(0, 0.8 - pos * 2)
    facts = [
        f"30-day range {r_lo:.6g}–{r_hi:.6g}; price in the bottom {pos*100:.0f}% of it",
        f"higher low {lo_last:.6g} vs prior low {lo_prev:.6g} — range low holding",
        f"mid-range {mid:.6g} and top-quarter {vah:.6g} are the rotation targets",
    ] + p.trend_facts()
    return Setup(p.symbol, "range_low", "Range-low higher low", last, stop, "4H", dca, tps, tp_r, risk,
                 sc + _common_score(p), facts, "4h", lo_last, extra={"range": [r_lo, r_hi]})


def detect_compression(p: Pack) -> Optional[Setup]:
    """
    4H compression / bull flag after an impulse, price pressing the top of the box.
    Neil: "4H bull flag / accumulation move", "4H compression should break to the upside",
    "aiming to hit this 4H breakout".
    """
    df = p.df4
    if len(df) < 160:
        return None
    last = p.last
    box = df.iloc[-30:]
    b_hi, b_lo = float(box["high"].max()), float(box["low"].min())
    width = b_hi / b_lo - 1
    if width > 0.14:
        return None
    # ATR compression: current ATR% in bottom third of the last 120 bars
    atrp = (df["atr"] / df["close"]).iloc[-120:]
    if atrp.iloc[-1] > atrp.quantile(0.35):
        return None
    # prior impulse: >= 15% up from the low of the 100 bars before the box
    pre = df.iloc[-130:-30]
    imp_lo = float(pre["low"].min())
    if b_hi / imp_lo - 1 < 0.15:
        return None
    # price in the upper 40% of the box, above 4H 20 and 50MA
    pos = (last - b_lo) / max(b_hi - b_lo, 1e-9)
    if pos < 0.6:
        return None
    if not (last > float(df["sma20"].iloc[-1]) and last > float(df["sma50"].iloc[-1])):
        return None
    stop = nice(b_lo * 0.995, "down")
    risk = (last - stop) / last
    if risk > 0.07:
        stop = nice(float(df["sma50"].iloc[-1]) * 0.99, "down")
        risk = (last - stop) / last
    if not (0.01 <= risk <= 0.07):
        return None
    measured = b_lo + (b_hi - imp_lo)          # flag-pole projection
    tps, tp_r = build_tps(last, stop, df, p.df1d, measured=measured)
    if not tps or tp_r[0] < 1.2:
        return None
    sc = 2.2 + (1.0 - width * 5) + (0.5 if last >= b_hi * 0.99 else 0)
    facts = [
        f"30-bar 4H box {b_lo:.6g}–{b_hi:.6g} ({width*100:.1f}% wide) after a {(b_hi/imp_lo-1)*100:.0f}% impulse",
        f"ATR% at the {int(atrp.rank(pct=True).iloc[-1]*100)}th percentile of the last 120 bars — coiled",
        f"price in the top {int((1-pos)*100)}% of the box, above the 4H 20/50MA; measured move ~{measured:.6g}",
    ] + p.trend_facts()
    return Setup(p.symbol, "compression", "4H compression breakout", last, stop, "4H", None, tps, tp_r, risk,
                 sc + _common_score(p), facts, "4h", b_hi, extra={"box": [b_lo, b_hi]})


def detect_daily_breakout(p: Pack) -> Optional[Setup]:
    """
    Daily range breakout within the last 1-4 days, price still near the breakout level.
    Neil: "Daily range breakout. Send it higher", "Not missing this daily range breakout".
    """
    d = p.df1d
    if d is None or len(d) < 40:
        return None
    last = p.last
    hi20 = d["high"].rolling(20).max().shift(1)
    closes = d["close"]
    brk = [i for i in range(len(d) - 4, len(d)) if closes.iloc[i] > hi20.iloc[i] * 1.0 and not np.isnan(hi20.iloc[i])]
    if not brk:
        return None
    i0 = brk[0]
    lvl = float(hi20.iloc[i0])
    if last < lvl * 0.985 or last > lvl * 1.12:
        return None
    if (closes.iloc[i0:] < lvl * 0.98).any():
        return None
    stop = nice(lvl * 0.975, "down")
    risk = (last - stop) / last
    if not (0.01 <= risk <= 0.10):
        return None
    rng_lo = float(d["low"].iloc[i0 - 20:i0].min())
    measured = lvl + (lvl - rng_lo)
    tps, tp_r = build_tps(last, stop, p.df4, d, measured=measured)
    if not tps or tp_r[0] < 1.2:
        return None
    days_ago = len(d) - 1 - i0
    sc = 2.4 + max(0, 1.0 - days_ago * 0.25) + (0.6 if p.above_1d_200() else 0)
    facts = [
        f"daily close above the 20-day range high {lvl:.6g} ({days_ago} day(s) ago) and holding",
        f"range depth {(lvl/rng_lo-1)*100:.0f}% → measured move ~{measured:.6g}",
    ] + p.trend_facts()
    return Setup(p.symbol, "daily_breakout", "Daily range breakout", last, stop, "4H", None, tps, tp_r, risk,
                 sc + _common_score(p), facts, "4h", lvl)


DETECTORS = [detect_sr_flip, detect_ma_retest, detect_range_low, detect_compression, detect_daily_breakout]


def scan_pack(p: Pack, ctx: Context) -> list[Setup]:
    out = []
    for det in DETECTORS:
        try:
            s = det(p)
        except Exception:  # noqa: BLE001
            s = None
        if s is None:
            continue
        # market-context adjustment (Neil waits for BTC to stabilise before fresh longs)
        if ctx.btc_trend == "down":
            s.score -= 1.5
            s.facts.append("BTC is weak — treat as scalp size until it stabilises")
        elif ctx.btc_trend == "chop":
            s.score -= 0.4
        # relative-strength hard filter: Neil wants coins holding up against BTC
        if p.rs7 < -0.06:
            s.score -= 1.0
        s.source = p.df4.attrs.get("source", "")
        out.append(s)
    return out
