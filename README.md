# neil-scanner

Daily scan of USDT perpetuals for long setups in the style of TraderNeil's `#neil-calls`
alerts (SR flip / retest, 200MA first retest, range-low higher low, 4H compression breakout,
daily range breakout). Picks the top 3, draws a TradingView-style chart for each, and writes
the alert text the way Neil words it.

Runs on GitHub Actions every day at 12:30 UTC (6:00 PM IST / 8:30 AM New York) and commits
the results to `output/`. A Claude scheduled task then clones this repo, reads the results
and writes the short analysis.

## Layout

```
scanner/data.py      public market data (OKX -> MEXC -> Gate fallbacks, no API keys)
scanner/setups.py    the five detectors + BTC context + TP builder
scanner/chart.py     chart renderer (dark, white OHLC bars, SMA 20/50/200, position box)
scanner/scan.py      orchestrator: universe -> scan -> rank -> charts -> output/
output/latest.json   today's picks (machine readable)
output/report.md     today's picks (human readable, with charts)
output/charts/       PNG per pick, named <date>_<SYMBOL>.png
output/history/      one JSON per day
tests/               synthetic fixtures for an offline dry run
```

## One-time setup

1. Create a **public** GitHub repo (e.g. `neil-scanner`) and push these files to `main`.
2. In the repo: *Settings → Actions → General → Workflow permissions* → **Read and write**.
3. *Actions* tab → `daily-neil-scan` → **Run workflow** once to verify it goes green and
   `output/latest.json` appears.

GitHub disables cron workflows after 60 days without any commits from a person — the
daily bot commits do not count. Pushing any small change (even to this README) every couple
of months keeps it alive.

## Local run

```
pip install -r scanner/requirements.txt
python scanner/scan.py -v                      # live
python tests/make_fixtures.py
python scanner/scan.py --fixtures tests/fixtures   # offline smoke test
```

## Tuning knobs

* `MAX_PER_DAY`, `NEIL_FAVOURITES` in `scanner/scan.py`
* `--min-vol` (default $5M 24h volume floor), `--max-symbols`
* Detector thresholds live at the top of each `detect_*` function in `scanner/setups.py`

Nothing here is trade advice; it is a pattern screen that surfaces candidates for a human to judge.
