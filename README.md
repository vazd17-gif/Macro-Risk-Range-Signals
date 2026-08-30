# Fractal Trend / Range Dashboard

A re-runnable daily pipeline that computes, for a universe of tickers:

- **RISK RANGE** — the probable daily high/low envelope
- **TRADE / TREND / TAIL** — short/medium/long duration lines, each with a bull/bear state
- a derived **state** (trending long / short / counter-trend) and a **fractal phase** (1–5)

This is a reconstruction of Hedgeye's Risk Range™ and the Similar Set (@similar_set)
fractal indicator, fitted against their published levels. It is **not** their
proprietary math. Every parameter is fitted, every fit is reported, and every
reference level used is stored in `fractal/data/labels.csv`.

---

## What changed against the build spec

The spec came in with three claims. Two did not survive contact with a larger
label set; the third is confirmed and now measured.

| Spec said | Now |
|---|---|
| **RANGE = spot ± 1σ EWMA. LOCKED.** Validated on one SLV upper edge to 2 cents. | **Wrong on two counts.** The range is centred on a **5-day EMA**, not on spot, and it is **±1.9σ**, not ±1σ. |
| **TRADE/TREND are adaptive (KAMA-family)**; adaptivity is the open research item. | **No adaptivity.** TRADE is a plain **29-day SMA**, TREND a plain **64-day EMA**. Adaptive forms fit *worse*. |
| Volume ruled out of the lines. | Confirmed — nothing here needed it. |

### The RANGE was validated on the wrong edge

The spec locked the range on a single observation: SLV's published upper edge of
64.30 against a model value of 64.27. That match is real, and this repo reproduces
it (`range_spot(SLV, λ=0.94, m=1.0)` → 64.273). But the same model puts SLV's
**lower** edge at 61.30, and the published lower edge that day was **58.37** — a
5% miss. One matching edge was read as a validated model.

Fitting both edges against 720 traced range values gives:

```
RANGE_high[t+1] = A[t] * exp(+1.902 * sigma[t])
RANGE_low [t+1] = A[t] * exp(-1.841 * sigma[t])
A = EMA(close, 5)      sigma = close-to-close EWMA, lambda = 0.88
```

rmse **1.12%** over 1,440 edges. Two things fall out of this:

- **It is not spot-centred.** Forcing the anchor onto spot (the spec's form) fits
  at 1.77% and only works by making the widths asymmetric (m_dn/m_up = 1.27×).
  The apparent downside skew in the published charts is an *anchoring* artefact:
  after a rally the 5-day EMA sits below spot, which pushes the whole envelope
  down. Once you centre correctly, the widths are near-symmetric (1.90 vs 1.84),
  so the spec's optional "persistence skew" is not needed.
- **It is not one sigma.** ±1.9σ, and the realised coverage confirms it: 84% of
  next-day closes land inside, against ~50% for the spec's m=1 reading.

### The lines are not adaptive

Fitted separately against the published levels, over four MA families, two price
bases (close, HLC3) and a 0–6 bar staleness lag:

| Line | Winner | rmse | median abs err | max | labels |
|---|---|---|---|---|---|
| TRADE | **SMA 29** | 0.68% | 0.48% | 1.41% | 9 |
| TREND | **EMA 64** | 1.30% | 0.82% | 3.79% | 28 |

Runners-up for TREND: WMA 85 (1.35%), SMA 64 (1.68%), KAMA (2.13%). The adaptive
families lose outright. An efficiency-ratio EMA (`aema`, span interpolated by ER)
was added specifically to give adaptivity its best shot and it converges to
span_fast=50 / span_slow=65 — i.e. it flattens into a fixed ~59-span EMA and ties
the plain EMA at 1.31%.

The spec's outliers came from too few observations, not from adaptivity:

- **XLF wanted a slower TREND** (+3.07% at 50d). At the fitted 64d it is +2.21%,
  and the whole panel shifted the same way. The line was simply too fast.
- **SLV TRADE wanted ~30d** while the seed used 20d. The fit lands on 29d, and
  SLV's TRADE residual drops to −0.43%.

Price basis makes no difference (close and HLC3 tie), and the lag axis is
degenerate with span (span 59 + lag 2 ≡ span 64 + lag 0), so there is no evidence
the published levels are stale.

### What the RANGE does out of sample

Over 3 years and 13 liquid names, `calib/backtest_coverage.py`:

```
coverage            84.2%      median range width  4.13%
breaches            up 8.3%    down 7.4%
5d forward return   after up-breach +0.18%   after down-breach +1.06%   uncond +0.48%
excess vs uncond    after up -0.30%          after down +0.58%
```

Mean reversion is present at both edges. That is not something the fit was asked
to produce — the parameters were chosen only to reproduce published levels — so it
is independent evidence the reconstruction is a real risk band rather than a
curve-fit to someone else's numbers. It is also exactly the behaviour the Similar
Set newsletter trades on ("sell SOME at the top end", "buy the damn dip").

---

## How the labels were built

The spec shipped 7 hand-collected reference points. This repo has **36 TRADE/TREND
labels across 32 tickers** plus **720 traced RANGE edges across 16 tickers**, all
derived from the source PDFs in this folder and all machine-verified.

**TRADE / TREND** come from the axis badges on the newsletter charts
(`SS:TREND 59.320`). `calib/labels.py` verifies every row before it can be fitted:
each chart prints a last price and a daily change, so the implied prior close must
match the price history. All 36 rows reconcile, most to ~1e-6. That check caught a
genuine error — the Colombia ETF is `COLO`, not `GXG`, and the wrong symbol would
have silently contributed a 10% residual.

Four of the harvested values (SLV, NVDA, XLV, TLT) coincide exactly with the
spec's independently collected seed values, and two more (HYG 79.655, BTC 72,700)
match figures quoted in the newsletter prose.

**RANGE** has no badge — it is only drawn as the red and green lines — so
`calib/extract_ranges.py` traces them off the chart images. The interesting part
is that nothing is read by eye:

- **Geometry from the price history.** Recovering the bar grid from the image
  alone fails (bodies and wicks split, adjacent bars merge, autocorrelation locks
  onto the body width). Instead the pitch, grid anchor and calendar alignment are
  solved by minimising the residual between every candle's top/bottom pixel and
  that bar's real high/low. Only the true geometry makes ~80 candle extremes land
  on their actual prices.
- **The axis is then pinned by the badges**, whose values are already verified,
  which removes the few tenths of a percent of level bias the candle fit carries.
- **Lines are traced column by column, not by connected components** — the white
  candles are drawn over the lines and cut them into dozens of fragments. The
  RANGE line moves a few pixels per column while the dotted TRADE/TREND series sit
  far away, so nearest-neighbour tracking seeded on unambiguous single-run columns
  stays on the line.
- **Charts that fail either check are dropped**, not guessed at. 16 of 30 passed;
  14 were rejected by the guardrails.

The end-to-end check: SLV's traced upper edge for 2026-08-28 comes out at
**64.311** against the published **64.30** — 0.02%.

One structural detail the tracing exposed: the lines are plotted **one slot beyond
the last candle**. That slot is the level for the session about to trade, which is
what makes these levels quotable intraday off the prior close.

---

## Layout

```
fractal/
  data/      loader.py  yahoo_client.py  ib_client.py  universe.py
             labels.csv          published TRADE/TREND levels (ground truth)
             range_labels.csv    traced RANGE edges
             chart_map.csv       chart image -> ticker
  model/     range_ewma.py  adaptive_ma.py  state.py  persistence.py
  calib/     labels.py            load + verify the label set
             fit_lines.py         fit TRADE/TREND (family x price x lag)
             fit_range.py         fit the RANGE against traced edges
             extract_pdf_charts.py    pull chart images out of the PDFs
             extract_chart_levels.py  axis calibration + line tracing
             extract_ranges.py    geometry, dating, verification
             backtest_coverage.py
  app/       scan.py  dashboard.py
  config/    params.yaml      every fitted constant lives here
  reference/charts/           extracted chart images
  out/       dashboard.html
```

`config/params.yaml` holds all fitted constants; nothing numeric is hard-coded in
`model/`. Re-running the calibration with `--write` overwrites it.

## Running it

```bash
python -m fractal.app.dashboard
```

Other entry points:

```bash
python -m fractal.app.scan --sort range --only at_top
python -m fractal.calib.labels
python -m fractal.calib.fit_lines --lines trade,trend --families sma,ema,wma,kama,aema --wide
python -m fractal.calib.fit_range
python -m fractal.calib.backtest_coverage --years 3 --horizon 5

# validate against Hedgeye's own published ranges (out of sample)
python -m fractal.calib.validate_hedgeye
python -m fractal.calib.validate_hedgeye --csv fractal/reference/hedgeye_early_look_week.csv

# is volume driving the range width?
python -m fractal.calib.fit_range_volume --volume-only

# out-of-sample check against another Similar Set daily email
python -m fractal.calib.extract_similarset_email --charts <dir>     --chart-date 2026-08-27 --prior-close 2026-08-26 --candidates NVDA,MSFT,SLV,...
```

### Range profiles in `params.yaml`

| profile | target | use |
|---|---|---|
| `anchor_ewma` | Similar Set | default; fitted to 720 traced edges |
| `hedgeye_anchor` | **Hedgeye** | wider; matches Hedgeye's published ranges (NVDA -> [206.9, 231.6] vs their 207-230) |
| `hedgeye_vol` | Hedgeye + volume | **experimental, not validated** |
| `spot_ewma` | build spec | superseded; kept for comparison |

Adding new reference levels: forward a Similar Set chart, run
`python -m fractal.calib.extract_pdf_charts <pdf> --out fractal/reference/charts`,
append rows to `labels.csv` and `chart_map.csv`, then re-run the fits.

## Data

Interactive Brokers is the primary source (`data/ib_client.py`, via `ib_insync`
against a running TWS/Gateway); Yahoo is the fallback and is what the current fits
were run on. `load_prices` tries IB and falls back automatically. Set
`data.source: ib` in `params.yaml` to prefer IB.

Futures (`CL=F`, `6B=F`, `6E=F`) are carried in `labels.csv` but flagged `fit=0` —
continuous-contract roll makes their levels non-comparable. `BTC-USD` is likewise
held out: it trades 365 days a year, so a window measured in bars is not the same
duration as it is for equities.

## Open items

0. **Volume in the RANGE width is unresolved, not disproven.** See the volume
   section above: underpowered on the ~27 single-name ranges available. The code
   path is implemented and ready to re-fit on more labels.

1. **TAIL is unfitted.** No published TAIL level appears in any source document, so
   `tail` ships as an EMA-600 extrapolation of the TREND result and is flagged
   low-confidence below 750 bars of history. It needs labels, not more modelling.
2. **TRADE rests on 9 labels.** The 0.68% rmse is encouraging but thin. Every new
   chart with a visible `SS:TRADE` badge is worth adding.
3. **14 charts fail extraction.** Mostly annotation overlays swallowing candles, and
   a few where the traced lines are too short. Recovering them would roughly double
   the RANGE label set.
4. **EWMA gap handling.** `winsor_z` is implemented (clip the latest return at N prior
   sigmas before the variance update) but is off, because the fitted λ=0.88 already
   lands where the label set wants it. Worth revisiting if post-gap ranges look hot.
5. **The 5-phase execution overlay** is implemented as a phase label only
   (`model/state.py`). Entry/hedge/trim rules are not, and should not be built until
   the levels are trusted on more than one newsletter's worth of labels.

---

## Validation against Hedgeye's *own* published ranges (out of sample)

Everything above is fitted to Similar Set, which is itself a reconstruction of
Hedgeye. Hedgeye publishes its actual numbers in plain text — the ETF Pro Plus
weekly report, the daily ETF Pro change notes, and the "Our Levels" block at the
bottom of every Early Look — so those are the real target. None of these levels
were used to fit any parameter, so this is a true out-of-sample test. Harvested
from email into `reference/hedgeye_ranges.csv` and `reference/hedgeye_early_look.csv`;
run with `python -m fractal.calib.validate_hedgeye`.

**Early Look "Our Levels" — 17 macro names, ranges for 2026-08-28 off the 8/27 close**

| | result |
|---|---|
| RANGE edge error | median **0.27%**, all 34/34 edges within 5% |
| TREND direction | **15/15 (100%)** — SPX, NDX, RUT, VIX, USD, gold, silver, copper, oil, natgas, Nikkei, DAX, XLV, RSP, IGV, HYG, LQD |
| range width vs Hedgeye | **1.02×** — essentially identical |

On liquid, actually-moving instruments the reconstruction reproduces Hedgeye's
published range almost exactly (SPX 7620–7788 vs model 7612–7782; USD 98.50–99.66
vs 98.61–99.61; HYG 79.54–80.01 vs 79.59–80.05).

**ETF Pro Plus weekly — 35 ETFs, ranges off the 8/21 close**

| | result |
|---|---|
| RANGE edge error | median **0.65%**, 61/70 within 2%, all 70 within 5% |
| TREND direction | 28/35 (80%) |
| range width vs Hedgeye | **0.56×** — model too narrow |

The two sets disagree on exactly one thing, and it is informative: **width on
quiet instruments.** The edge levels match to well under 1% because that week was
calm, but on low-volatility names the model's band is about half Hedgeye's
(HDV 2.7% vs 6.7%, VYM 2.0% vs 3.7%, LVHI 1.2% vs 3.3%), while on names that
actually move the two agree (QTUM 8.0% vs 12.4%, ICOP 10.7% vs 12.8%, and the
whole macro set at 1.02×). Fitting Hedgeye's edges directly implies a multiplier
of **~2.1σ up / ~2.5σ down**, against the Similar-Set fit of 1.90 / 1.84 — Hedgeye
runs a **wider and more downside-skewed** range than Similar Set, and appears to
**floor the width** so calm instruments keep a few percent of range instead of
letting EWMA collapse. All 7 TREND-direction misses are near-cash bond/again
low-vol ETFs (CLOX, CLOZ, VTIP, VCSH, IVOL, HYG, EPHE) where price sits on the
line and "bull vs bear" is noise; every macro name matched.

**Takeaway.** The reconstruction is validated against Hedgeye directly, not just
against Similar Set: it reproduces the published TREND direction and both range
edges to a fraction of a percent on liquid names. The one real gap is that
Similar Set (and therefore this model) draws a tighter range than Hedgeye on quiet
instruments. Two fixes are worth testing: a longer vol memory (higher λ), and a
minimum range-width floor. `calib/validate_hedgeye.py` is the harness to check
either against.

### Sources used
- ETF Pro Plus – New Weekly Report (2026-08-23, data through 8/21) — 35 ETF ranges
- Early Look "Our Levels" (2026-08-28) — 17 macro ranges + TREND direction
- ETF Pro change notes (2026-08-27, -28) — range confirmations (FXY, QTUM, VXF, UUP)
- Macro Show summary notes — narrative only (no range table); yields the S&P 7,788
  lower-high and NVDA "Alpha Code 217/208" (TRADE 217 / TREND 208, matching the
  model's NVDA TREND to ~1%)
- Similar Set daily newsletter (2026-08-28) — the 36 TRADE/TREND labels and 720
  traced RANGE edges the model is fitted to

---

## Does volume drive the RANGE width? (tested against Hedgeye)

Hedgeye describes the Risk Range boundaries as adjusted by the *rate of change of
volume* against a 1-month (TRADE) and 3-month (TREND) baseline. The build spec
ruled volume out of the *lines*; it never tested the *range width*, which is a
separate question. `calib/fit_range_volume.py` tests it directly against Hedgeye's
own published ranges, with volume features `v1 = vol/SMA(vol,21)` and
`v3 = vol/SMA(vol,63)`. Findings, in order of how the evidence tightened:

1. **The first apparent signal was an artifact.** On a mixed set of 87 ranges,
   `v3` correlated −0.34 with the implied width-multiplier — until we noticed the
   cash indices and spot contracts (SPX, VIX, gold, …) report no real Yahoo volume,
   so their `v1=v3=1.0` constant rows were biasing the correlation. Volume applies
   only to stocks/ETFs (the source's own guidance), and those instruments have to
   be isolated.

2. **On the clean stock/ETF subset, the signal disappears.** Restricted to the 62
   volume-bearing edges, `v1` and `v3` correlate ~0 with the multiplier, and adding
   them makes the edge-level fit *worse* (pure σ 1.14% → +v1+v3 1.19%). A single σ
   multiplier reconstructs Hedgeye better than any volume model here.

3. **The one hint matches the described mechanism but is underpowered.** A proper
   within-ticker test (demeaning by ticker, which is what "volume dynamically
   adjusts the range over time" actually means) leaves every level-form volume
   feature near zero — except the **rate of change of the 1-month volume ratio**,
   which correlates **+0.26**. That is exactly the "rate of change vs baseline"
   form Hedgeye describes. But it rests on 25 observations dominated by
   stable-volume liquid ETFs (NVDA, the one high-variation name, has 2 points), so
   it is a hint, not a result.

**Conclusion.** On the ~27 single-name ranges available, a volume effect on the
range width **cannot be confirmed or refuted** — the sample is underpowered on
exactly the instruments where volume varies. The shippable Hedgeye target
(`hedgeye_anchor`) therefore uses a plain σ multiplier and no volume term; the
volume-adjusted profile (`hedgeye_vol`) is implemented and left in place but
flagged **experimental / not validated**. Settling this needs single-stock ranges
across many high-volume-variation days — which the Similar Set daily charts supply
(NVDA, MSFT, PLTR, SNOW, QTUM across the week), and which
`calib/extract_similarset_email.py` harvests for exactly this purpose.

The RANGE model (`model/range_ewma.py`) implements the volume path in full
(`range_anchor_vol`, using both the 1-month and 3-month baselines) so the
hypothesis can be re-tested the moment more single-stock labels are added — no code
change required, just `fit_range_volume.py --write` on a larger set.

---

## Out-of-sample check against another Similar Set day

The model is fitted to the 2026-08-28 Similar Set charts. The daily emails carry
the same charts as external Kit CDN images, so any other day is out of sample.
`calib/extract_similarset_email.py` fetches a day's charts, identifies each one,
extracts the RANGE and badges, and compares against the model.

**Identification is the hard part, and the obvious approach is wrong.** The first
version ranked candidate tickers by the candle-vs-OHLC percent fit error — the same
statistic that works so well for geometry — and it identified all 25 charts as HYG.
A low-volatility ~$80 instrument reconciles *any* chart, because the axis fit finds
a near-flat slope that maps every candle to roughly one price with a tiny percent
error. Percent error measures whether a level is plausible, not whether a pattern
matches. The fix is to rank by the **correlation of the candle high/low sequence**
with the ticker's real OHLC (`shape_score`); only the true ticker tracks.

Result on 2026-08-27 (levels for that session, off the 8/26 close):

| ticker | SS range | model range | low err | high err |
|---|---|---|---|---|
| TLT | 81.58–82.72 | 81.77–84.09 | +0.24% | +1.65% |
| SLV | 58.38–64.50 | 59.17–64.20 | +1.36% | −0.46% |
| NVDA | 207.34–227.24 | 205.10–220.76 | −1.08% | −2.85% |
| HECA | 28.39–29.02 | 27.68–28.13 | −2.49% | −3.08% |

RANGE edges: **median 1.51%**, 5/8 within 2%. The extracted badges corroborate the
lines independently — SLV's 56.52 / 59.59 against model TRADE 56.30 / TREND 58.90,
NVDA's 211.23 against model TREND 209.14.

That is meaningfully looser than the in-sample 8/28 fit, which is the expected and
honest result: 1–3% out of sample versus ~0.3% against Hedgeye's macro text levels.
Only 4 of 25 charts cleared the identification gate (`min_shape=0.85`), so this is a
spot check rather than a full second day. Loosening the gate trades yield for the
risk of mis-identified charts silently entering the label set, which is the one
failure mode this harness exists to prevent — so it is left strict.

---

## Macro Risk Range Signals (dashboard + newsletter)

`python run_daily.py` builds a report over a **188-name watchlist — 157 ETFs and
the 31 highest dollar-volume S&P 500 single names** — applying Hedgeye's own ETF Pro
decision rules. The model does not distinguish funds from stocks; the split exists
only so the two lists can be maintained separately (`RAW` and `STOCKS` in
`data/etf_universe.py`) and filtered apart in the dashboard.

Berkshire is carried as **BRK-B**: the data feed does not recognise the `BRK.B`
form. GOOGL and GOOG are both included, as supplied.

| Signal | Condition | Action |
|---|---|---|
| **ADD LONG** | at/near the **low** end of the Risk Range, bullish TRADE and/or TREND | buy |
| **REMOVE LONG** | has **broken** TRADE and/or TREND in the last 3 sessions | sell |
| **ADD SHORT** | at/near the **high** end of the Risk Range, bearish TRADE and/or TREND | short, or avoid |
| **COVER SHORT** | a broken name has **reclaimed** TRADE and/or TREND | cover |
| **WATCHLIST** | at the low end but the signal has broken | watch, nothing to act on yet |

`WATCHLIST` is the one rule not taken from Hedgeye. It comes from the Similar Set
handbook, which is explicit that a first break of TRADE means the immediate
direction is counter-trend: *don't buy the low end of the RANGE during the break,
wait for TREND support.* Without it, "buy the low end" fires exactly when it is
most likely to lose money. These names are worth watching; they are not yet
actionable.

### Volume

Every name carries its daily volume against its **1-month (21-day)** and
**3-month (63-day)** rolling averages — the two windows Hedgeye benches volume on,
matching the TRADE and TREND durations.

Volume is displayed as a **z-score**, not a percentage. Volume is lognormal and its
scale spans orders of magnitude across this list, so a +60% session is routine for
a thin fund and remarkable for a mega-ETF; a percentage is not comparable across
names, a z-score is. Two are computed — the z of log volume against the fund's own
**1-month** distribution and against its **3-month** distribution.

The 3-month z carries the `surge` / `dry` flag at ±2, because 21 observations give
a noisy standard deviation. The 1-month z is reported beside it because the two
diverging is itself informative: DUST on 2026-08-28 read z +1.8 against 1 month but
z +2.7 against 3 months, meaning its volume had already been building for weeks.

Where it appears:

- **Dashboard** — Volume / z vs 1m / z vs 3m columns, amber for heavy and blue for
  light, plus *Volume surge* and *Volume dry* filter buttons.
- **Newsletter signal entries** — a volume line with both z-scores.
- **VOLUME OUTLIERS section** — the absolute numbers spelled out: shares traded,
  then the 1-month average and the 3-month average each in shares, with the percent
  deviation and the z against that window.
- **Full list** — both z-scores as their own columns.

On 2026-08-28 that surfaced 10 outliers, and they were coherent: the whole precious
metals complex — PALL (z +2.9), DUST (+2.7), GLD (+2.7), WEAT (+2.1), AAAU (+2.1),
SLV (+2.1) on the 3-month window — traded heavy into Friday's gold selloff. That is
directly relevant context for the two ADD LONG signals, which are both gold: they
hit the low end of their range on two to three times normal volume.

The newsletter shows each fund's name in small grey type beside its ticker, in the
signal sections, the portfolio block and the full list. Names are fetched once and
cached in `data/etf_names.csv` (`python -m fractal.data.etf_names` refreshes them),
then trimmed of issuer boilerplate so they fit on one line: "iShares 20+ Year
Treasury Bond ETF" renders as *20+ Year Treasury Bond*. A name that reduces to the
ticker itself (Invesco QQQ Trust) is dropped rather than repeated.

Every ticker is colour-coded by its TREND state — **green when price is above
TREND, red when below** — in the dashboard table and in the newsletter's full list,
so the state of the whole book reads at a glance without checking numbers.

Outputs (in `fractal/out/`): `etf_dashboard.html` (sortable, filter by signal and
by group, range-position bar per name), `etf_newsletter.html` (inline-styled HTML
email — tables only, so it survives email clients), and a dated CSV.

```bash
python -m fractal.app.etf_report                     # ETF report only
python -m fractal.app.etf_report --profile anchor_ewma --edge 0.20
python run_daily.py --skip-etf                       # macro only
```

### Portfolio

`data/portfolio.csv` holds the book — one row per lot, with side, entry date and
entry price. Closing a lot stamps status and exit rather than deleting the row, so
the book keeps its own history.

**The starting point is the last completed close** — Friday's close on a weekend
run. Every level in the report comes off that bar, so the book uses the same one:
adding a position without a price books it at that close, on that date. That makes
"opened at the baseline" and "opened three weeks ago" directly comparable, because
both are measured against the same current spot.

```bash
python -m fractal.app.portfolio add GLD long                      # books at the last close
python -m fractal.app.portfolio add UUP short --price 28.24 --date 2026-08-20
python -m fractal.app.portfolio close IWM                         # exits at the last close
python -m fractal.app.portfolio list
python -m fractal.app.portfolio status            # reconcile against today
python -m fractal.app.portfolio status --live     # use live intraday quotes as spot
```

Each position carries two performance numbers: **since entry** (entry price to
current spot) and **since the baseline close** (Friday's close to current spot).
Outside market hours the current spot *is* that close, so the second reads 0.00%;
it starts moving as soon as the next session trades. `--live` pulls intraday
quotes instead.

The point of the book is that a signal is only actionable relative to what is
already held: a break of TREND is a sell only if you own it, and the low end of the
RANGE is a buy only if you do not. Reconciliation turns each signal into a
position-level action:

Actions are phrased as the order to place, not as a description of the position:

| Holding | Signal | Action |
|---|---|---|
| long | broke TRADE/TREND | **SELL** — exit the long |
| long | at the high end of the RANGE | **TRIM** — sell some into strength |
| long | at the low end, still bullish | **LONG** — add |
| short | reclaimed TRADE/TREND | **COVER** — buy the short back, closing it out |
| short | at the low end of the RANGE | **COVER SOME** — buy back part of it |
| short | at the high end, still bearish | **SHORT** — open or add to the short |
| either | nothing triggered | HOLD |

SELL and SHORT are different orders and are kept as different words: SELL exits a
long you hold, SHORT opens or adds to a short. A COVER is a buy that takes a short
out of the book — run `portfolio close <ticker>` to record it.

The book was opened from flat at the 2026-08-28 close:

| Ticker | Side | Entry | Why |
|---|---|---|---|
| GLD | long | 408.89 | low end of the RANGE, bullish TRADE and TREND |
| AAAU | long | 43.94 | low end of the RANGE, bullish TRADE and TREND |

Those were the only two new positions available. The other 38 signals that day were
REMOVE LONG (26) and COVER SHORT (10) — both act on holdings that did not exist —
plus 2 WATCHLIST, which by definition is not actionable.

Open positions appear at the top of both the dashboard and the newsletter with
entry price, the baseline close, current spot, both P&L numbers, days held and the
action.
Run the report against a different book with `--portfolio <path>`, or add `--live`
for intraday pricing.

### Two guards that matter

Both were added after the first run produced misleading output:

- **One report, one date.** The first run mixed as-of dates (123 names on 8/28,
  32 on 8/27, one stale since July) because the cache was written mid-session.
  Signals are now computed only on the latest common date; anything staler is
  reported but raises no signal. That caught **GXG**, which has not traded since
  2026-07-17 — Global X MSCI Colombia now trades as **COLO**.
- **Recently listed names are flagged.** An EMA needs roughly three times its span
  of history before it stops carrying the value it was seeded from, so a TREND on
  a young ETF is softer than it looks. Anything with under 3 x the TREND span
  (192 bars) is marked `young`, its bar count is reported, and any signal it raises
  carries a "short history" note. It is flagged, not suppressed — DRAM (Roundhill
  Memory, added later, 103 bars) is the only one today.
- **Cash-like names raise no signal.** The first run flagged 33 REMOVE LONG, but a
  third were near-cash bond funds (TBIL, SHY, BUXX, MTBA) "breaking TRADE" by five
  basis points. Any ETF whose entire Risk Range is narrower than
  `MIN_RANGE_PCT` (2%) is now shown but never signalled. This is the same lesson
  the Hedgeye validation produced from the other side: every TREND-direction
  mismatch there was a near-cash bond ETF where price sits on the line and
  bull/bear is noise. 19 of the 155 names are cash-like.

The ETF report defaults to the **`hedgeye_anchor`** RANGE profile, not the Similar
Set default, because these rules are Hedgeye's and that profile is the one fitted
to Hedgeye's published ranges. The Similar Set profile draws a narrower band and
would trip the edge rules more often.

### Watchlist resolution

The fund side of the watchlist is **157 ETFs**, all pricing. Seven symbols from the original list were
removed because they no longer price: CHIR, EGPT, MOON and WNDY (liquidated), RAYC
(one bar, last 2025-11-28), and EPG and PP (unrecognised). They are recorded in a
comment in `data/etf_universe.py` so the removal stays auditable.

Two symbols are substituted rather than dropped:

- **`ML PX` → MLPX** (Global X MLP & Energy Infrastructure) — typo in the source list.
- **`GXG` → COLO** — Global X MSCI Colombia stopped printing under GXG on
  2026-07-17 and now trades as COLO, which is also the symbol Hedgeye and Similar
  Set use for it.

---

## Running it live

### Daily refresh

`automation/daily_update.bat` recomputes everything from the latest completed
session, writes the dashboard, newsletter and CSV, and logs each run to
`fractal/out/logs/daily_<date>.log`. Register it with Windows Task Scheduler:

```powershell
powershell -ExecutionPolicy Bypass -File automation\install_task.ps1
```

That sets it to run weekdays at 17:45 local (`-Time "17:45"` to change). Anything
from about 30 minutes after the 4pm ET close is fine — the job needs the session's
bar to be complete. `-StartWhenAvailable` means a missed run (machine asleep)
catches up rather than being skipped.

```powershell
Start-ScheduledTask -TaskName FractalRiskRangeDaily      # run now
Get-ScheduledTaskInfo -TaskName FractalRiskRangeDaily    # last result
Unregister-ScheduledTask -TaskName FractalRiskRangeDaily -Confirm:$false
```

### Email

`fractal/app/publish.py` builds and sends the newsletter. It defaults to writing a
`.eml` file for review rather than sending:

```bash
python -m fractal.app.publish --to you@example.com            # writes out/newsletter.eml
python -m fractal.app.publish --to you@example.com --send     # actually sends
```

Two things keep it safe to run unattended:

- **No duplicate sends.** The report is keyed to the last completed session, so a
  scheduler firing on a holiday, or twice in an evening, would otherwise resend the
  same newsletter. The as-of date of the last successful send is kept in
  `out/.last_sent` and a send is skipped unless the data has moved on (`--force`
  overrides).
- **No credentials in the repo.** SMTP details come from the environment:
  `FRACTAL_SMTP_USER`, `FRACTAL_SMTP_PASS`, `FRACTAL_MAIL_TO`. Gmail needs an
  **app password** (myaccount.google.com/apppasswords), not the account password —
  set it yourself; nothing here asks for it or stores it.

```powershell
setx FRACTAL_SMTP_USER "you@gmail.com"
setx FRACTAL_SMTP_PASS "your-16-char-app-password"
setx FRACTAL_MAIL_TO   "you@gmail.com,someone@else.com"
```

The daily job emails only when `FRACTAL_MAIL_TO` is set, so it stays silent until
you opt in.

### Dashboard hosting

Published as a private Artifact — a shareable link that works on a phone, with the
narrow-screen layout dropping the reference columns and keeping name, spot, range
position and signal. It is a **snapshot**: republishing updates it in place, but the
scheduled job cannot do that itself.

For a URL that updates on its own, GitHub Pages is the better fit: commit
`fractal/out/` on each run and push. The repo is not under version control yet,
which is the first step either way.
