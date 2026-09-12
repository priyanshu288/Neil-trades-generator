"""
Chart renderer that mimics the TradingView screenshots Neil posts:
dark background, white OHLC bars, SMA 20 (yellow) / 50 (red) / 200 (cyan),
long-position tool (green box to final TP, red box to stop), yellow TP lines with
price badges on the right axis, and a grey support zone when the setup has one.
"""
from __future__ import annotations

from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BG = "#0d1117"
PANEL = "#0d1117"
GRID = "#1c2230"
TXT = "#c9d1d9"
UP = "#ffffff"
DN = "#9aa4b2"
MA20, MA50, MA200 = "#f5c542", "#ef5350", "#26c6da"
GREEN, RED = "#26a65b", "#c0392b"


def _fmt(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4g}" if x < 100 else f"{x:.5g}"
    return f"{x:.5g}"


def _badge(ax, y, text, color, x=1.0, fc=None, tc="black"):
    ax.annotate(text, xy=(x, y), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8, color=tc, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=fc or color, ec="none"), annotation_clip=False, zorder=10)


def render(setup: dict, df: pd.DataFrame, out_path: str, tf_label: str, source: str) -> str:
    n_show = 130 if tf_label == "4h" else 170
    d = df.iloc[-n_show:].reset_index(drop=True)
    x = np.arange(len(d))
    pad = int(len(d) * 0.30)                       # room on the right for the position box

    fig = plt.figure(figsize=(12.5, 7.6), dpi=130, facecolor=BG)
    ax = fig.add_axes([0.035, 0.07, 0.86, 0.86], facecolor=PANEL)
    ax.set_xlim(-1, len(d) + pad)

    # ---- OHLC bars ---------------------------------------------------------
    for i, r in d.iterrows():
        c = UP if r["close"] >= r["open"] else DN
        ax.plot([i, i], [r["low"], r["high"]], color=c, lw=0.9, zorder=3)
        ax.plot([i - 0.35, i], [r["open"], r["open"]], color=c, lw=0.9, zorder=3)
        ax.plot([i, i + 0.35], [r["close"], r["close"]], color=c, lw=0.9, zorder=3)

    # ---- moving averages ---------------------------------------------------
    for col, colr, lw in (("sma20", MA20, 1.4), ("sma50", MA50, 1.4), ("sma200", MA200, 1.6)):
        if col in d and d[col].notna().any():
            ax.plot(x, d[col], color=colr, lw=lw, zorder=4)

    entry, stop = setup["entry"], setup["stop"]
    tps = setup["tps"]
    final_tp = tps[-1]
    x0 = len(d) - 1
    x1 = len(d) + pad - 1

    # ---- long-position tool ------------------------------------------------
    ax.add_patch(Rectangle((x0, entry), x1 - x0, final_tp - entry, fc=GREEN, ec="none", alpha=0.35, zorder=2))
    ax.add_patch(Rectangle((x0, stop), x1 - x0, entry - stop, fc=RED, ec="none", alpha=0.45, zorder=2))
    ax.hlines(entry, x0, x1, color="#e6e6e6", lw=0.8, zorder=5)

    # ---- TP lines + badges -------------------------------------------------
    for k, tp in enumerate(tps):
        ax.axhline(tp, color="#f5c542", lw=0.8, alpha=0.9, zorder=5)
        _badge(ax, tp, f"TP{k+1} {_fmt(tp)}", "#f5c542")
    _badge(ax, entry, f"{_fmt(entry)}", "#e6e6e6")
    _badge(ax, stop, f"SL {_fmt(stop)}", "#ef5350", tc="white")
    if setup.get("dca"):
        ax.hlines(setup["dca"], x0, x1, color="#ffffff", lw=0.8, ls=(0, (3, 3)), zorder=5)
        _badge(ax, setup["dca"], f"DCA {_fmt(setup['dca'])}", "#cfd8dc")

    # ---- structural extras -------------------------------------------------
    ex = setup.get("extra") or {}
    if "range" in ex:
        lo, hi = ex["range"]
        ax.axhline(hi, color="#8b949e", lw=0.7, ls="--", alpha=0.7)
        ax.add_patch(Rectangle((-1, lo * 0.985), len(d) + pad, lo * 0.03, fc="#8b949e", alpha=0.25, ec="none"))
    if "box" in ex:
        lo, hi = ex["box"]
        ax.add_patch(Rectangle((len(d) - 31, lo), 31, hi - lo, fc="none", ec="#8b949e", lw=0.8, ls="--"))
    if setup.get("level") and setup["kind"] in ("sr_flip", "daily_breakout"):
        lvl = setup["level"]
        ax.add_patch(Rectangle((-1, lvl * 0.992), len(d) + pad, lvl * 0.016, fc="#8b949e", alpha=0.25, ec="none"))

    # ---- axes cosmetics ----------------------------------------------------
    lo_y = min(stop, float(d["low"].min())) * 0.985
    if "sma200" in d and d["sma200"].notna().any():
        ma_lo = float(d["sma200"].dropna().min())
        if ma_lo > lo_y * 0.9:                     # keep the 200MA in frame when it is close
            lo_y = min(lo_y, ma_lo * 0.99)
    hi_y = max(final_tp, float(d["high"].max())) * 1.02
    ax.set_ylim(lo_y, hi_y)
    ax.yaxis.tick_right()
    ax.tick_params(colors=TXT, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    step = max(1, len(d) // 9)
    ticks = list(range(0, len(d), step))
    ax.set_xticks(ticks)
    fmt = "%d %b" if tf_label != "1h" else "%d %b %H:%M"
    ax.set_xticklabels([d["ts"].iloc[i].strftime(fmt) for i in ticks], color=TXT, fontsize=8)

    src = {"okx": "OKX", "mexc": "MEXC", "gate": "Gate"}.get(source, source or "")
    ax.text(0.01, 0.985, f"{setup['symbol']}USDT Perpetual · {tf_label} · {src}", transform=ax.transAxes,
            color=TXT, fontsize=10, va="top", fontweight="bold")
    ax.text(0.01, 0.955, "SMA 20 · 50 · 200   |   " + setup["label"], transform=ax.transAxes, color="#8b949e",
            fontsize=8, va="top")
    ax.text(0.01, 0.015, f"Neil-style scanner · {datetime.now(timezone.utc):%b %d, %Y %H:%M} UTC",
            transform=ax.transAxes, color="#5b6472", fontsize=7, va="bottom")
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path
