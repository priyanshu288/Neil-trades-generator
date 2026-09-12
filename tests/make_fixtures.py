"""
Builds synthetic OHLCV fixtures that reproduce the setup shapes the detectors look for,
so the pipeline (detect -> rank -> chart -> report) can run without exchange access.

    python tests/make_fixtures.py && python scanner/scan.py --fixtures tests/fixtures
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
END = pd.Timestamp("2026-09-12 08:00", tz="UTC")


def path_to_ohlc(closes: np.ndarray, tf: str, vol_scale: float = 1e5) -> pd.DataFrame:
    n = len(closes)
    step = {"1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    ts = pd.date_range(end=END, periods=n, freq=step)
    opens = np.r_[closes[0], closes[:-1]]
    noise = np.abs(rng.normal(0, 0.006, n)) * closes
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.006, n)) * closes
    vol = rng.uniform(0.5, 1.5, n) * vol_scale
    return pd.DataFrame({"ts": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": vol})


def write(sym: str, c4: np.ndarray, c1: np.ndarray, c1d: np.ndarray):
    path_to_ohlc(c4, "4h").to_csv(OUT / f"{sym}_4h.csv", index=False)
    path_to_ohlc(c1, "1h").to_csv(OUT / f"{sym}_1h.csv", index=False)
    path_to_ohlc(c1d, "1d").to_csv(OUT / f"{sym}_1d.csv", index=False)


def walk(n, start, drift, vol):
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def upsample(c4: np.ndarray, k: int = 4, vol: float = 0.004) -> np.ndarray:
    """1H path consistent with the 4H closes."""
    out = []
    for a, b in zip(np.r_[c4[0], c4[:-1]], c4):
        seg = np.linspace(a, b, k + 1)[1:] * np.exp(rng.normal(0, vol, k))
        seg[-1] = b
        out += list(seg)
    return np.array(out[-300:])


def downsample(c4: np.ndarray, k: int = 6, n: int = 250) -> np.ndarray:
    d = c4[::-1][::k][::-1]
    pre = walk(max(0, n - len(d)), d[0] / 1.3, 0.001, 0.03)
    return np.r_[pre, d][-n:]


# ---- BTC: mild uptrend, above its MAs -------------------------------------------
btc4 = walk(300, 60000, 0.0006, 0.012)
write("BTC", btc4, upsample(btc4), downsample(btc4))

# ---- SRF: resistance at ~1.00 tested 3x, reclaimed 6 bars ago, retesting from above
c = np.r_[walk(150, 0.80, 0.0008, 0.02)]
c = c / c[-1] * 0.93
res = 1.00
seg = []
for i in range(120):
    v = 0.90 + 0.07 * np.sin(i / 9) + rng.normal(0, 0.008)
    seg.append(min(v, res - 0.004))
seg = np.array(seg)
seg[[30, 60, 95]] = res - 0.002
reclaim = np.r_[np.linspace(0.93, 1.03, 8), np.array([1.035, 1.028, 1.012, 1.008, 1.015, 1.02, 1.024, 1.03])]
c4 = np.r_[c, seg, reclaim][-300:]
write("SRF", c4, upsample(c4), downsample(c4))

# ---- CMP: impulse then a tight 30-bar box, price at top of box, above 20/50MA
base = walk(170, 2.0, 0.0002, 0.015)
imp = np.linspace(base[-1], base[-1] * 1.45, 40) * np.exp(rng.normal(0, 0.01, 40))
top = imp[-1]
box = top * (0.965 + 0.035 * (0.5 + 0.5 * np.sin(np.arange(60) / 4))) * np.exp(rng.normal(0, 0.003, 60))
box[-5:] = np.linspace(box[-6], top * 0.995, 5)
c4 = np.r_[base, imp, box][-300:]
write("CMP", c4, upsample(c4), downsample(c4))

# ---- RNG: 30-day range, price near the lows with a higher low and a bounce
rlo, rhi = 10.0, 14.0
seg = []
for i in range(180):
    v = 12 + 1.9 * np.sin(i / 14) + rng.normal(0, 0.15)
    seg.append(float(np.clip(v, rlo, rhi)))
seg = np.array(seg)
tail = np.r_[np.linspace(12.5, 10.2, 20), np.linspace(10.2, 11.1, 12), np.linspace(11.1, 10.55, 10), np.linspace(10.55, 11.3, 10)]
c4 = np.r_[walk(100, 12, 0, 0.01), seg, tail][-300:]
write("RNG", c4, upsample(c4), downsample(c4))

# ---- MAR: strong 1H uptrend, extended >6% above the 1H 200MA, now tapping it for the first time
c1 = walk(200, 5.0, 0.0012, 0.006)
c1 = np.r_[c1, np.linspace(c1[-1], c1[-1] * 1.12, 40), np.linspace(c1[-1] * 1.12, c1[-1] * 1.02, 56), np.array([c1[-1] * 1.03] * 4)]
c1 = c1[-300:]
c4 = np.r_[walk(225, 3.5, 0.001, 0.02), c1[::4]][-300:]
write("MAR", c4, c1, downsample(c4))

# ---- DBO: 20-day range, daily close broke out 2 days ago, holding just above it
d = np.r_[walk(200, 0.30, 0.0005, 0.03), 0.42 + 0.03 * np.sin(np.arange(45) / 5) + rng.normal(0, 0.004, 45)]
d = np.r_[d, [0.47, 0.475, 0.472]][-250:]
c4 = np.r_[walk(282, 0.40, 0.0002, 0.012)[:282], np.linspace(0.45, 0.47, 6), [0.475, 0.478, 0.474, 0.471, 0.473, 0.472, 0.4725, 0.472, 0.4715, 0.472, 0.4725, 0.472]][-300:]
write("DBO", c4, upsample(c4), d)

# ---- DUD: choppy downtrend, should produce nothing
c4 = walk(300, 1.0, -0.0008, 0.02)
write("DUD", c4, upsample(c4), downsample(c4))
print("fixtures written to", OUT)
