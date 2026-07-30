# Statistical-Trading-Bot (`sttbot`)

A modular, testable **systematic-trading ecosystem for the solo operator**. The
design premise is deliberately narrow: instead of competing with institutional
funds on liquid majors, target *fragmented, low-capacity, and behavioral*
niches where AUM capacity floors keep large capital out — micro-cap
post-earnings drift, prediction-market probability boundaries, cross-venue
crypto divergence, and quantitative sports pricing.

This repository is the software backbone for that thesis: a set of small,
independently testable modules mirroring the pipeline **data ingestion →
strategy/analysis → orchestration/execution → operations/risk**.

## Architecture

```
Data ingestion      →  sttbot.data.storage      DuckDB columnar tick store
Strategy / analysis →  sttbot.strategies         declarative params + signals
Execution           →  sttbot.execution          OMS interface, paper broker,
                                                  TTL + slippage order manager
Risk / ops          →  sttbot.risk               drawdown circuit breaker
                       sttbot.economics           net-edge / friction model
                       sttbot.monitoring          webhook alerting
```

## Modules

| Module | What it provides |
| --- | --- |
| `data.storage` | `TickStore` — DuckDB-backed tick/order-book storage with Parquet export; `:memory:` for tests. |
| `data.datasets` | Catalog of openly-licensed datasets + caching downloader (`STTBOT_DATA_DIR`), with checksum pinning; `LIVE_ENDPOINTS` records the direct-egress APIs and their access requirements. |
| `data.tokens` | DexScreener + GoPlus clients mapped into `TokenMetadata`, keeping *zero* and *unknown* distinct so a clean token isn't failed on a gap that doesn't exist. |
| `venues.polymarket` | Full-venue client: gamma keyset enumeration (with the `after_cursor` trap encoded), CLOB order books with depth walking, complete-basket arbitrage and executable sizing. |
| `data.probe` | `python -m sttbot.data.probe` — re-measures which data hosts this machine can actually reach. |
| `data.football` | Loaders for the bulk football dataset **and** football-data.co.uk season CSVs (the only wired-up source carrying a genuine closing line); overround and de-vig helpers. |
| `strategies.base` | `Strategy` + `Param` — declarative hyperparameters with grid-search introspection (the `# @param` convention, made programmatic). |
| `strategies.pead` | Micro-cap Post-Earnings Announcement Drift via Standardised Unexpected Earnings (SUE). |
| `strategies.dixon_coles` | Dixon-Coles bivariate-Poisson model for low-scoring sports, plus +EV and Closing Line Value helpers. |
| `strategies.dixon_coles_fit` | Weighted-MLE fitter for the above: per-team attack/defence, home advantage γ and dependence ρ, with exponential time decay, `mean(attack)=0` identifiability, and an analytic gradient. |
| `backtest.walk_forward` | Walk-forward engine — fit on a trailing window, bet the next matchday, settle after friction. Look-ahead is structurally impossible, not merely intended. |
| `backtest.metrics` | Sharpe, max drawdown, hit rate, ROI on staked capital. |
| `backtest.clv` | Closing-line-value **controls**: what CLV a random selection earns on the same matches (`clv_baseline`), and what an *identically-priced* selection earns (`clv_odds_matched_baseline`, `clv_excess_per_bet`), so neither mechanical CLV nor a longshot preference is mistaken for skill. |
| `strategies.prob_arbitrage` | Multi-outcome probability-boundary arbitrage for categorical prediction markets. |
| `venues.prediction` | Cross-venue prediction-market pricing: per-venue fee models (Kalshi's quadratic, Polymarket's zero), depth-bounded arbitrage sizing, Kelly staking, depth-weighted consensus. |
| `strategies.market_making` | Scalping/market making on binary contracts: fee-aware spreads, `untradeable_band`, Avellaneda-Stoikov inventory skew, one-sided quoting at inventory limits. |
| `backtest.mm_simulator` | Market-making simulator with fill mark-out, so adverse selection is measured rather than assumed away. |
| `strategies.amm` | Constant-product AMM mechanics: price impact, the no-arb fee band, profit-maximising CEX-DEX arbitrage sizing, impermanent loss. |
| `backtest.cohort` | Birth-cohort return measurement: resolves captured pools from candle history, separates fetch failures from genuine no-trades, defends against initialisation sweeps / dust ticks / unconfirmed wicks, Hill tail index, and clone-family (wash) detection. |
| `strategies.breadth` | Many-names/small-size portfolio engine: exact round-trip cost through the pool, depth-capped position sizing, breakeven-multiple and names-to-confidence arithmetic, and an assumption-driven Monte Carlo (explicitly not a backtest). |
| `strategies.token_screen` | Pre-trade rug/honeypot screening for low-cap tokens, plus exit sizing against real pool depth. |
| `economics.friction` | Net-edge formula, maker/taker routing rule, order-size/timing stealthing. |
| `execution.oms` | `OMS` protocol + in-memory `PaperBroker` for deterministic paper trading. |
| `execution.order_manager` | `DynamicOrderManager` — pre-trade slippage cap + Time-To-Live cancellation. |
| `risk.circuit_breaker` | High-water-mark **and** rolling-window drawdown kill switch. |
| `monitoring.alerts` | Fail-safe Discord/Telegram webhook notifier (stdlib only). |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the end-to-end paper-trading demo
python examples/run_pipeline.py

# run the tests
pip install -e .
pytest
```

## Example: a +EV sports signal

```python
from sttbot.strategies.dixon_coles import DixonColesModel, TeamRating, expected_value

model = DixonColesModel(
    ratings={"Home": TeamRating(0.35, -0.15), "Away": TeamRating(0.05, -0.05)},
    home_advantage=0.30,
    rho=-0.05,
)
probs = model.match_probabilities("Home", "Away")
print(expected_value(probs.home_win, decimal_odds=1.95))  # +EV if positive
```

## Example: prediction-market boundary arbitrage

```python
from sttbot.strategies.prob_arbitrage import Outcome, find_boundary_arbitrage

book = [
    Outcome("Cand-A", best_bid=0.40, best_ask=0.42),
    Outcome("Cand-B", best_bid=0.33, best_ask=0.35),
    Outcome("Cand-C", best_bid=0.19, best_ask=0.21),
]
opp = find_boundary_arbitrage(book, fee_rate=0.005)
print(opp)  # buy_basket when the asks sum below 1.00, net of fees
```

## Open data

`sttbot.data.datasets` catalogs openly-licensed, key-free datasets and caches
them locally on first use (override the location with `STTBOT_DATA_DIR`):

```python
from sttbot.data.datasets import available, describe, fetch

print(available())            # ['btcusd_1min', 'football_json_en1', 'football_matches']
path = fetch("football_matches")   # downloads once, then served from cache
```

| Dataset | License | Size | Contents |
| --- | --- | --- | --- |
| `football_matches` | MIT | 44 MB | 230,557 club matches (2000–2025, 38 divisions, 1,214 teams) with scores, match stats, Elo, and 1X2 / O-U / Asian-handicap odds from Bet365 plus best-of-book maxima |
| `btcusd_1min` | MIT | 95 MB | 6.85M one-minute BTC/USD OHLCV candles from Bitstamp (2012–2025) |
| `football_json_en1` | CC0 | <1 MB | EPL 2023-24 fixtures/results in JSON |

`python examples/explore_open_data.py` profiles the football data. Measured on
the full 230k rows:

- **Home advantage** — 1.488 vs 1.151 goals, i.e. γ ≈ **0.257**, which
  independently supports the model's 0.25 default.
- **Low-score cells** — 0-0, 0-1, 1-0 and 1-1 are 20.6 / 20.1 / 27.4 / 31.9% of
  all 0-or-1-goal games, the concentration the Dixon-Coles `tau` correction exists to fix.
- **Friction** — a single book's 1X2 margin averages **6.73%**, but the
  best-of-book margin is only **0.93%**; line-shopping recovers **5.8pp**, which
  is larger than most model edges. In **23.6%** of matches the best-of-book sum
  falls below 1.00 — a boundary arbitrage of the kind `find_boundary_arbitrage`
  detects. Treat that as an *upper bound*: those maxima are peak prices across
  ~17 books and assume simultaneous availability, accounts everywhere, and
  stake limits that don't bind.

### Direct-egress sources

Re-probed 2026-07-28 from an unrestricted network. An earlier sandbox returned
403 for all of these, which was its allowlisting proxy rather than any upstream
restriction — so a blanket 403 means "re-probe", not "the venue is gone". Run
`python -m sttbot.data.probe` to re-measure from your own machine.

| Host | Status | Notes |
| --- | --- | --- |
| `www.football-data.co.uk` | reachable | Per-season odds CSVs carrying **both** pre-match and closing prices — the only source here that supports real CLV |
| `gamma-api.polymarket.com` | reachable | Public read API, no key |
| `api.elections.kalshi.com` | reachable | Public read on `/trade-api/v2/markets`; trading needs a signed key |
| `data.sec.gov` | reachable | 403s unless the User-Agent carries a contact address — set `STTBOT_USER_AGENT` |
| `stooq.com` | reachable | Free daily OHLCV; rate-limited |
| `api.exchange.coinbase.com` | reachable | Public read endpoints |
| `api.binance.com` | **blocked** | HTTP 451 geo-restriction, not a proxy. Use Coinbase instead |

## Walk-forward results on real odds

`python examples/backtest_dixon_coles.py` fits the model on a trailing two-year
window, bets the next matchday, and settles after 2% commission. Seven seasons
(2018/19–2024/25), English tiers 1–4 plus Scotland, ~2,600 matches per division.

**The model does not beat these markets.** Every division loses money, and
against a single book its closing-line value is *negative* — it is
systematically taking worse prices than the market closes at, which is a
stronger indictment than the P&L, because CLV carries no settlement variance.

| Division | Entry | Bets | ROI | Mean CLV | Random CLV | Skill |
| --- | --- | --- | --- | --- | --- | --- |
| E0 (Premier League) | single-book | 1,675 | −9.43% | −0.0081 | −0.0079 | −0.0002 |
| E0 | best-of-book | 2,208 | −4.54% | +0.0078 | +0.0068 | +0.0010 |
| E2 (League One) | best-of-book | 2,738 | −6.54% | +0.0067 | +0.0057 | +0.0010 |
| SC2 (Scottish L1) | best-of-book | 556 | −2.65% | +0.0146 | +0.0067 | **+0.0078** |
| SC3 (Scottish L2) | best-of-book | 500 | +3.12% | +0.0135 | +0.0067 | **+0.0068** |

The `Random CLV` column is the reason this table is trustworthy. Best-of-book
entry prices beat a single book's close roughly 60% of the time **with no model
at all** — you are comparing the best of ~17 prices against one. A strategy
reporting "+0.007 mean CLV, 64% positive" on that basis has demonstrated
nothing. `backtest.clv.clv_skill` subtracts that free lunch.

What survives the control: in the two thinnest markets tested, the model shows
genuine selection skill (SC2 t = 5.1, SC3 t = 4.1). But the skill is worth
~0.7 percentage points of implied probability, and the margin plus commission
is worth more, so **it does not convert into profit**: SC3's +3.12% ROI is 0.5
standard errors from zero (SE 6.7%) and is noise, not a finding. Do not read
that number as an edge.

### Out-of-sample: the skill is real, and much broader than that

`python examples/clv_thin_markets.py` widens the test to all 22 divisions
football-data.co.uk publishes with both pre-match and closing odds (~53,000
matches). Pre-specified before looking at any result: **CLV skill increases
with market thinness**, thinness proxied by mean single-book overround
(computed from prices alone, independent of the skill metric), tested with
**one** Spearman correlation on the 15 divisions never examined when the
hypothesis was formed. Skill is measured against an *odds-matched* baseline —
each bet compared to the mean CLV of same-priced selections — so a preference
for longshots cannot masquerade as forecasting.

The hypothesis holds: **Spearman ρ = +0.539, p = 0.038**. More importantly, the
earlier conclusion was too narrow. Skill is not a Scottish curiosity — it is
present almost everywhere, and **14 of 15 out-of-sample divisions are
individually significant at t > 2**:

| Division | Overround | Bets | ROI | Skill | t |
| --- | --- | --- | --- | --- | --- |
| EC (National League) | 1.0793 | 2,467 | −3.34% | +0.0075 | **10.2** |
| SC1 (Scottish Champ.) | 1.0843 | 558 | −8.91% | +0.0072 | 5.7 |
| SP2 (Segunda) | 1.0625 | 2,501 | −10.96% | +0.0052 | 7.8 |
| B1 (Belgian Pro) | 1.0698 | 1,583 | −5.34% | +0.0044 | 5.0 |
| F1 (Ligue 1) | 1.0541 | 2,171 | −1.56% | +0.0030 | 5.3 |
| E0–E3 *(in-sample)* | ~1.051 | — | — | ~+0.0002 | <2 |

The English tiers are the *exception*, not the Scottish leagues being special.
They have the lowest overrounds in the sample and near-zero skill — which is
what "efficiently priced" looks like. Testing only England and Scotland made a
general pattern look like a local quirk.

**Caveat on the mechanism.** Thin markets also have wider CLV dispersion
(ρ = +0.661 with overround, p = 0.007), so part of the thinness–skill link is a
scale effect rather than pure inefficiency. Normalising skill by dispersion
barely moves the point estimate (ρ = +0.539 → +0.496) but loses significance at
n = 15 (p = 0.060). The direction is supported; "thinness causes skill" is not
cleanly separable from "thinness widens everything" at this sample size.

Honest summary: the fitter works, the backtest is sound, the model has real and
broadly significant forecasting skill against the closing line — and it is
**still not profitable in a single division out of 15**. Skill of ~0.3–0.75
percentage points does not cover a 5–9% overround. Beating the closing line by
a little is not the same as beating the book.

## Crypto, meme coins, and prediction markets

`python examples/crypto_and_prediction.py` demonstrates all three. Each module
targets a specific way naive implementations lose money.

**Fees that aren't flat.** Kalshi charges `0.07 · P · (1−P)` per contract —
1.75c at a 50c contract, 0.33c at 5c. An identical 1c gross edge is therefore
*unprofitable mid-book and profitable at the tail*. Modelling this as a flat
percentage gets the sign of the trade wrong. `venues.prediction` also refuses to
pair two markets unless they're explicitly declared `equivalent=True`: a
same-headline market with a different resolution source turns a "risk-free"
hedge into an unhedged bet on a technicality.

**Arbitrage sized to the wrong target.** The intuitive AMM arb trades until the
pool price equals the external price. That overshoots — the marginal unit earns
nothing while still paying fees. `optimal_arbitrage` maximises profit directly:

```
dy* = (√(p_ext · x · y · γ) − y) / γ
```

Tests verify the closed form against a brute-force search over ±50% of the
optimum, and confirm it beats price-equality sizing. `no_arb_band` gives the
range where fees make *any* size unprofitable — divergence inside it is not an
opportunity, however large it looks against mid.

**Position sizing that ignores the exit.** For thin tokens the dominant loss
mode isn't a bad entry, it's being unable to sell. `token_screen` covers the rug
and honeypot heuristics (LP lock, holder concentration, mint/freeze authority,
sell-tax asymmetry, simulated sell), and `position_limit` takes the *smaller* of
your risk budget and what the pool can actually absorb on exit. Missing data
counts as a failure, not a pass — absence of evidence isn't evidence of safety
for an asset like this.

The checks are research-led rather than invented. The literature keys on
**top-10 concentration** (~30%), not just the largest holder — ten coordinated
wallets at 5% each carry the same exit risk as one whale while passing a
single-holder check comfortably — and on **sniper/bundled wallets** holding the
launch float. Both are screened, with a test for the distributed-whale case
that a top-1 measure alone misses.

### Why the defaults are strict

Published base rates, 2025–2026:

| Metric | Figure |
| --- | --- |
| Meme coins that die or lose meaningful volume | ~97% (Binance Research) |
| pump.fun tokens whose last trade is their launch day | ~69% |
| pump.fun graduation to a DEX listing | <2% through 2025, ~0.26% by mid-2026 |
| Average rug pull | ~$510k, >$2.8bn total in 2025 (Chainalysis) |

Against a base rate that skewed, essentially all the achievable edge is in
**avoidance, not selection**. A screen that rejects almost everything is working
as designed: a false reject costs you a missed winner, a false accept costs the
whole position. Loosening thresholds to surface more candidates is usually a
mistake. The ~69% same-day death rate is also what justifies the 24h minimum
age — it removes most of the distribution at almost no opportunity cost.

These are heuristics over self-reported metadata, not a safety guarantee. A
token can pass every check and still go to zero; most will. Nothing here
estimates whether a token goes *up*.

## Scalping prediction markets

`python examples/scalp_prediction_markets.py`. The edge is liquidity
provision, not speed — which is deliberate: scalping liquid majors would be the
head-to-head latency competition this project exists to avoid.

**Fees decide where you can quote at all.** Kalshi's quadratic fee is paid twice
on a round trip, so the breakeven spread is 3.50c at a 50c contract and 0.67c
at 5c. `untradeable_band` turns that into the actionable number:

| Market spread | Blocked region (Kalshi) |
| --- | --- |
| 1c | 0.08–0.92 — **84% of the book** |
| 2c | 0.18–0.82 — 64% of the book |
| 4c | none |

So on Kalshi you scalp the tails, not the coin flips.

### Polymarket: makers pay nothing

`python examples/scalp_polymarket.py`. Polymarket's 2026 schedule charges
**takers only**, using the same quadratic shape at a category-dependent rate
(0.03 sports, 0.04 politics/finance/tech, 0.05 economics/culture/weather, 0.07
crypto; geopolitical and world-events markets are fee-free). Part of it is
rebated to makers. A market maker is passive on *both* legs, so it pays zero —
and with a rebate is paid to quote:

| Maker round-trip cost | Kalshi | Polymarket | Polymarket + rebate |
| --- | --- | --- | --- |
| at 0.05 | 0.67c | 0.00c | −0.12c |
| at 0.50 | 3.50c | 0.00c | −0.62c |

At a 2c market spread Kalshi blocks 64% of the book; Polymarket blocks **none**.
Use `polymarket_fees(category, maker_rebate=...)`; an unknown category falls
back to the `other` rate rather than silently assuming free trading.

This also fixed a real bug: `breakeven_spread` was pricing both legs at the
*taker* rate. On Kalshi maker and taker are equal so it made no difference; on
Polymarket it was the gap between paying 5% and earning a rebate.

**Then the tick becomes the binding constraint.** With fees gone, the 1c grid
is the floor: quotes snap to it (bid down, ask up, so snapping never tightens
the intended spread), the tightest round trip earns 1–2c, and sub-cent noise is
unmonetisable — a ±0.3c wobble produces **zero fills**, because the quote sits
at 0.49/0.51 and the price never reaches it.

**Inventory skew has to be scaled to the spread.** Quotes shift against
inventory (Avellaneda-Stoikov) using `p(1-p)` as the risk scale, tapering to
zero at expiry. The scale is easy to get wrong: at `risk_aversion=1.0` the shift
is 25c at full inventory against a ~1c spread, and the maker buys back its own
position far above fair value. Measured on identical flow:

| `risk_aversion` | Skew at full inventory | Net P&L |
| --- | --- | --- |
| 0.01 | 0.25c | +2.31 |
| 0.05 (default) | 1.25c | +2.01 |
| 0.25 | 6.25c | +0.52 |
| 1.00 | 25.0c | **−5.05** |

**Adverse selection is the real cost, and it's measured, not assumed.** A
simulator that fills you at your quote against an unreactive path reports a
profit almost regardless of strategy. `mm_simulator` marks out every fill —
price N steps later versus fill price, signed by direction — so being picked off
shows up explicitly, and before the P&L does.

Two properties the tests pin, both counterintuitive:

- **Quoting around the last mid loses whenever the price swings wider than your
  spread.** With ±3c oscillation and a 0.5c half-spread, the maker sells at
  0.475 while fair value is 0.50, every time. A maker needs a fair-value
  estimate better than "the last mid".
- **Slow drift never fills a re-quoting maker** — the quote re-centres and
  outruns it. Adverse selection comes from jumps that outpace re-quoting.

Fills assume queue priority and no partial fills, so all of this is an **upper
bound**. Real quoting is worse.

## Scanning all of Polymarket

`python examples/scan_polymarket.py` enumerates the entire open venue —
**16,241 events, 144,876 markets** — and looks for structure worth trading.

**The venue is enormous and almost entirely empty.** 91.1% of markets traded
$0 in the last 24 hours and median market liquidity is **$18**, so the $63M
of daily volume comes from a few hundred markets.

| Pass | Result |
| --- | --- |
| Complete multi-outcome baskets | 5,721 |
| Positive edge at top of book | 7 |
| Executable against real depth | 7, totalling **$66 profit on $2,334 capital** |
| Markets worth quoting (mid 0.05–0.95, ≥2 ticks, ≥$10k volume) | **57 of 144,876** |

Total boundary arbitrage available across the whole platform is $66, locked
until the events resolve in January 2027, and the single largest opportunity
caps out at 950 contracts. Date ladders ("X by July 31" nested inside "X by
Dec 31") were checked separately for monotonicity violations: **zero** across
140 ladders. Those markets are internally consistent.

Two traps are encoded in `venues.polymarket` because both produce confidently
wrong answers rather than errors, and both caught me first:

- **Pagination.** The gamma response field is `next_cursor` but the request
  parameter is `after_cursor`. Sending it back under the name it arrived with
  is accepted and ignored, so you re-read page one forever — my first
  "complete" pull was 200 events that were 100 unique and 100 duplicates.
  Plain `offset` isn't an escape either: it caps near 2,400 against a real
  universe of 16,241, truncating to ~15% while still looking finished.
- **Basket completeness.** Scoring only the legs that happen to be quoted
  reported **172 arbitrages with edges up to +0.85/contract**. Requiring the
  full outcome set leaves 7. The other 165 were unhedged shorts on the
  outcomes I'd dropped, wearing an arbitrage costume.

The scalping screen documents its own vacuity: because Polymarket rebates
makers, breakeven spread is negative and *all 108,160* live two-sided markets
"clear the round-trip fee". A screen everything passes measures nothing — the
binding constraint is adverse selection, which `backtest.mm_simulator` measures
directly.

## Screening live meme tokens

`python examples/screen_meme_tokens.py` joins DexScreener (discovery, pool
depth, age) with GoPlus (holder concentration, LP locks, mint/freeze authority,
transfer taxes) and runs `strategies.token_screen` over real Solana launches.

On a live run of 24 discovered tokens: **0 passed, 24 rejected**, almost all
for being under 24 hours old or under $25k of liquidity. That is the screen
working — it is built for *avoidance, not selection*, and ~97% of these tokens
die.

The more useful number is capacity. Sizing the exit against real pool depth at
a 2% price-impact budget:

| Token | Pool liquidity | 24h volume | Max exit |
| --- | --- | --- | --- |
| Faucina | $37,148 | $583,661 | **$323** |
| RIKA | $19,874 | $1,299,391 | **$173** |
| PIBBLE | $8,856 | $494,199 | **$77** |

RIKA turned over $1.3M in a day against a pool that can only absorb **$173**
on the way out. Volume in this asset class is not liquidity — it is the same
few dollars round-tripping. Meme-coin scalping is capacity-limited to a few
hundred dollars per token, which caps the strategy long before edge does.

Honest gap: GoPlus had no holder data for 22 of the 24 tokens (they were
minutes old), and neither source covers sniper/bundle concentration or a real
sell simulation. Those come back as **unknowns rather than passes** —
`token_screen` distinguishes "a check ran and the token lost" from "the data
was never there", because the two call for different responses.

### There is no "established but thin" sweet spot

The obvious next hypothesis was that tokens surviving their first month stay
small enough to be inefficient but grow big enough to trade.
`python examples/token_capacity_cohorts.py` tests it, scoring every token on
two axes at once — exit capacity inside a 2% impact budget, and turnover
(24h volume ÷ pool liquidity), because liquidity without flow is stranded
capital rather than an opportunity.

| Cohort | n | Median liquidity | Median exit | Median turnover |
| --- | --- | --- | --- | --- |
| < 1 day | 26 | $18,390 | $160 | **175.2×** |
| 1–7 days | 13 | $22,633 | $197 | 94.5× |
| 7–30 days | 16 | $308,731 | $2,686 | 0.04× |
| 1–6 months | 28 | $3,091,221 | $26,892 | **0.00×** |
| > 6 months | 68 | $809,169 | $7,039 | **0.00×** |

Capacity does improve with age — roughly 150× from fresh launch to one month.
But the flow disappears at exactly the same rate. **90% of established tokens
turn over less than 0.1× of their pool per day**, and within the thin band
($25k–$1M liquidity) it is also 90%. Of 96 established tokens, exactly **one**
had both a ≥$5k exit and ≥0.5× turnover — and it was SOL itself, at $26M of
liquidity, which is the opposite of thin.

The two properties are anticorrelated, and the reason is structural: a thin
market that is genuinely active does not stay thin. It either attracts
liquidity until it is no longer thin, or the activity dies and leaves stranded
depth behind. Fresh-and-thin has flow but no capacity; established-and-liquid
has both but is competitive; established-and-thin has capacity and no
counterparty. There is nowhere to sit.

This is the same shape as the Polymarket result — 91% of markets there traded
$0 — and it is the sharpest limit on the whole thin-markets thesis so far.
Inefficiency is easy to find in places nobody trades. That is not a
coincidence; it is the reason the inefficiency survives.

## Breadth: many tokens, small size

If per-name capacity is the ceiling, the only remaining shape is width.
`strategies.breadth` sizes a book across many names at once, and
`python examples/breadth_portfolio.py` runs the full pipeline — discover,
screen, size against real pool depth, print intended orders.

The engine separates three things that are usually blurred together:

- **`round_trip_cost` is exact**, simulated through the constant-product pool.
  Both legs are priced against the *same* pool state, and that choice matters:
  simulating the exit against the reserves your own entry just moved lets your
  impact reverse itself, and round-trip cost then *falls* with size (0.598% at
  $50 down to 0.480% at $5,000 in a $20k pool) — which would tell a
  capacity-constrained strategy to trade bigger. That reversal is only real if
  you exit in the same instant, which a held position does not.
- **`breakeven_multiple` and `names_for_confidence` are arithmetic** on an
  assumed hit rate. They say what the world must look like.
- **`simulate` is a Monte Carlo over an assumed payoff tail.** It is *not* a
  backtest and nothing here is fitted to realised token returns.

The arithmetic is unforgiving. At a 3% survival rate and ~2–4% round-trip cost:

| Quantity | Value |
| --- | --- |
| Multiple winners must return to break even | **33.3×** |
| Concurrent names for a 95% chance of holding one | **99** |
| A 20-name book misses entirely | 54% of the time |

A live run built a book of **2 positions** — because the DexScreener profile
feed yields ~30 tokens per refresh and only ~10% clear even a relaxed screen.
A 2-name book misses entirely 94% of the time. **The binding constraint has
moved from capacity to discovery throughput.** GeckoTerminal's `new_pools`
endpoint serves 200 per refresh and is the obvious fix; the profile feed cannot
sustain this strategy.

Simulated over 12 rounds at three assumed tails, from a $10,000 bankroll:

| Assumed tail | Median | Mean | P(loss) | P(ruin) |
| --- | --- | --- | --- | --- |
| Fat (α 0.8) | $88 | $245,859 | 88% | 85% |
| Medium (α 1.5) | $76 | $190 | 100% | 100% |
| Thin (α 3.0) | $75 | $76 | 100% | 100% |

That mean-versus-median gap *is* the strategy: it is positive expectancy that
loses almost every time, and only pays through a tail nobody here has measured.
The assumed α decides the entire answer, so measuring the realised distribution
of token returns is the work that would turn this from a structure into a
strategy.

Execution is not implemented — the example prints intended orders. Real fills
need a funded wallet, slippage-bounded routing and MEV protection, all of which
make outcomes worse than modelled.

### Measuring the tail: the answer is "this data cannot tell you"

`backtest.cohort` and `examples/measure_token_cohort.py` exist to replace the
assumed α with a measured one. I captured a **616-pool birth cohort** live
(25-minute window) and resolved it from candle history.

It cannot be done retrospectively. Solana creates ~1,600 pools an hour against
a 200-row discovery feed, so the feed spans about **seven minutes** — there is
no reaching back. And any cohort assembled later from what is still listed has
had its failures delisted already.

The first measurement looked spectacular: α = 0.59 (infinite mean), a 1,802×
best peak, 2.65% of tokens reaching 100×. **Almost all of it was artifact.**
An adversarial review of five independent checks found:

- **70.6% of pools have their maximum high inside candle zero** — the
  pool-initialisation sweep, where the first minute spans hundreds of × between
  low and high before real liquidity exists.
- The largest peak, 152,363×, was a single tick carrying **$0.00074 of volume**
  in a pool whose entire lifetime volume was $6.65.
- **My own code had a bug**: `pool_ohlcv` returned `[]` on a rate-limit error,
  so 164 of 300 pools were silently recorded as "never traded". Re-querying
  them politely returned full history for **every one**. It now returns `None`
  for a failed fetch and `[]` only for a genuinely empty pool, and
  `MeasurementReport` counts the three cases separately.
- **Volume thresholds cannot establish that trading was real.** Three separate
  mints named XAU/SOL, created within 18 seconds, had lifetime volumes of
  $6529.84 / $6529.86 / $6529.98 — agreeing to four significant figures. One
  bot, three clones, wash volume that clears any threshold. `detect_clone_groups`
  now flags these.

Peak distribution under each defence, same 136 pools:

| Rule | n | Median | p90 | p99 | Best | α |
| --- | --- | --- | --- | --- | --- | --- |
| A raw | 136 | 1.34× | 7.84× | 404× | 152,363× | 0.55 |
| B skip birth candle | 101 | 1.04× | 4.88× | 27.5× | 152,722× | 0.67 |
| C peak needs $100 volume | 136 | 1.13× | 5.37× | 20.6× | 41.6× | 1.06 |
| D peak confirmed by a close | 136 | 1.05× | 5.86× | 232× | 152,363× | 0.62 |
| E all three | 101 | 1.00× | 2.92× | 18.1× | 27.5× | 0.93 |

**α is not identified by this sample.** It moves from 0.55 to 1.06 across
defensible rules, straddling the 1.0 line that separates a finite mean from an
infinite one — and the whole breadth argument rests on which side it falls. No
mean should be quoted at all: deleting one dust tick moved it three orders of
magnitude.

**The tail is not even stable in time.** Two measurements of the *same cohort*
40 minutes apart reported best peaks of 1,802× and 152,363×. I first assumed
that was a candle-aggregation effect and was wrong — re-bucketing 1-minute bars
into 5-minute bars reproduces the peak exactly, because a maximum of highs is
invariant to aggregation. The real cause was that ZACK/SOL printed its
$0.00074 dust tick at 04:24 UTC, after the first measurement had already
finished at ~04:21. One garbage print, arriving between two runs, moved the
measured tail by 84×. An α estimated from this data is a snapshot that any
subsequent dust tick can overturn.

The honest verdict is not "breadth works" or "breadth fails" — it is that a
616-pool cohort observed for under an hour **cannot resolve the parameter the
strategy depends on**. What it did establish is how to measure it without
fooling yourself, and that every artifact found made the strategy look better
than it is, never worse.

The screen and the strategy also genuinely disagree, and the example says so:
`token_screen` defaults to a 24-hour minimum age because ~69% of launches never
trade past day one, but the only cohort with flow is *under* a day old. Running
breadth there means lowering that floor deliberately and managing the risk by
size instead of selection, keeping only the checks size cannot protect you
from — live mint authority, freeze authority, and top-10 concentration.

## Design notes

- **Testability first.** Clocks and sleep are injected into the order manager
  and circuit breaker, so time-dependent logic is deterministic under test.
  The `PaperBroker` implements the same `OMS` surface as a live adapter.
- **Friction is not optional.** Every signal is meant to be evaluated *after*
  fees, slippage, gas, and spread; `should_use_maker` suppresses taker orders
  when the spread would eat too much of the edge.
- **Fail-safe risk.** The circuit breaker latches once tripped — a recovering
  equity curve cannot silently re-enable trading; a human must `reset()`.
- **No look-ahead by construction.** The walk-forward loop derives its training
  window from the prediction date and cuts strictly at `date < d`, so a fit
  cannot see a result it is about to bet on. A test tampers with a future
  matchday's scores and asserts every earlier bet is unchanged.
- **Measure against the right null.** A metric that a random strategy also
  earns is not evidence. `backtest.clv` makes that baseline a first-class
  object rather than something a reader is trusted to remember.

## Scope & disclaimer

This is research/engineering scaffolding, not financial advice and not a
turnkey money printer. Live venue adapters (Interactive Brokers, CCXT,
Kalshi/Polymarket) are intentionally left as thin integrations behind the
`OMS` protocol. Trade at your own risk and comply with the terms of every
venue you connect to.

## Requirements

Python 3.11+, `numpy`, `duckdb`, `pandas`, `scipy`; `pytest` for the test suite.
