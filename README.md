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
| `data.earnings` | Earnings surprises (Alpha Vantage, key or cache) joined to keyless Yahoo prices; `trailing_sue` standardises by *prior* surprises only, and `first_tradeable_date` encodes that a post-market report cannot be traded that day. |
| `strategies.earnings_market` | Prices Polymarket's "Will X beat quarterly earnings?" contracts: trailing beat rate shrunk toward the **listed** universe's base rate, a Brier skill control against that base rate, and edge scored against bid/ask rather than the mid. |
| `backtest.event_study` | Market-adjusted drift measurement: entry at the first genuinely tradeable session, benchmark netted over identical sessions, and the announcement gap reported separately so it can never be booked as drift. |
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
| `paper.settlement` | Closes positions whose markets have resolved, crediting the payout. Accepts a resolution **only** when outcome prices form a real binary payout — `closed == true` is not sufficient, and settling on it would zero out a winning basket. |
| `paper` | Durable paper trading: DuckDB-backed account where fills are the only state (cash/positions/P&L are recomputed, never stored), idempotent on a caller-supplied ref, and an all-or-nothing basket runner wired to the circuit breaker. |
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

The honest verdict at 45 minutes was not "breadth works" or "breadth fails" —
it was that the cohort **could not resolve the parameter the strategy depends
on**. Every artifact found made the strategy look better than it is, never
worse.

### At 3.6 days: the tail index was the wrong question

The same cohort was re-resolved three and a half days later, **610 of 616
pools**, using hourly candles covering their whole lives. It did not identify
α — the estimate spread *wider*, from 0.20 to 1.03 across defensible rules.
It found something that matters more.

**These tokens do not go to zero. They stop trading.**

| Death measured by | Rate |
| --- | --- |
| Price (final < 1% of entry) | **4.3%** |
| Silence (no trade in 6h) | 99.0% |
| Silence (no trade in 24h) | **95.6%** |
| Silence (no trade in 48h) | 92.1% |

69.6% of pools produced **exactly one hourly candle** in their entire life, and
the median pool last traded 87 hours ago. Only **27 of 610 (4.4%)** were still
tradeable after 3.6 days, and of those only one was meaningfully up on real
volume.

This invalidates the survival statistic both readings had reported. "97.3%
survived" at 45 minutes and "98.3%" at 3.6 days are pure censoring artifacts: a
token whose market disappears stops producing candles, so its last close
freezes at whatever it printed on the way out and it looks alive forever. **A
price series cannot show you the absence of trading.** `Outcome.died` now says
so in its docstring and `Outcome.stopped_trading()` is the measure that means
something.

For the strategy the consequence is blunt and does not depend on α at all. A
breadth book buys many names intending to exit winners at a take-profit. Within
a day, **95.6% of those positions have no counterparty**. You cannot run an
exit rule on assets nobody will buy, so the tail index is irrelevant — the
position is a total loss regardless of what the last print says. That is a
harder constraint than any of the capacity ceilings found earlier, and it is
the one that ends this line of work.

The screen and the strategy also genuinely disagree, and the example says so:
`token_screen` defaults to a 24-hour minimum age because ~69% of launches never
trade past day one, but the only cohort with flow is *under* a day old. Running
breadth there means lowering that floor deliberately and managing the risk by
size instead of selection, keeping only the checks size cannot protect you
from — live mint authority, freeze authority, and top-10 concentration.

## Paper trading with real money on the line, minus the money

Everything above is a backtest or a snapshot. `sttbot.paper` is the first piece
that runs continuously against live prices and keeps a durable record — fake
money, real markets, real accounting.

`PaperBroker` in `execution.oms` is an in-memory simulator built for tests: it
forgets everything on exit. `sttbot.paper.PaperAccount` is the other thing a
paper account has to be. It is backed by DuckDB, and its one design rule is
that **fills are the only state** — cash, positions, average cost and realised
P&L are all recomputed from the fill ledger rather than stored and mutated
beside it. A long-running bot that keeps a running cash balance *and* a fill
log will eventually disagree with itself after a crash mid-write, and the
disagreement is invisible until the numbers are already wrong. With one source
of truth there is nothing to reconcile.

Every fill carries a caller-supplied `ref`. A runner that dies after trading
but before recording will retry on restart, and recording an existing ref is a
no-op — the retry path is the normal path, not an error path.

`sttbot.paper.PaperRunner` is the loop: check the circuit breaker, price each
strategy's proposed `Basket` against a fill model, record what clears. Baskets
are **all-or-nothing** on purpose. The one strategy in this repository with a
measured positive edge is multi-outcome arbitrage, and a partially filled
arbitrage is not a smaller arbitrage — it is an unhedged directional bet on
whichever legs happened to fill. A test pins this: three good legs plus one
leg with no depth must trade zero legs, not three. Tripping the circuit
breaker actually flattens open positions through the same account rather than
only setting a flag, because a kill switch that leaves positions open is not a
kill switch — also tested directly.

`python examples/paper_trade_polymarket.py` wires this to the Polymarket scan:
scan the venue, confirm each candidate against live CLOB depth, size it
atomically, record whatever clears. A live run traded a real four-leg
arbitrage basket:

```
4 fills, 0 baskets rejected, equity $981.25
  FILLED BUY  250 @ 0.0045  fee $0.0557  ...top-selling artist...
  FILLED BUY  250 @ 0.4400  fee $3.0800  ...
  FILLED BUY  250 @ 0.0406  fee $0.4867  ...
  FILLED BUY  250 @ 0.4700  fee $3.1138  ...
  cash $754.50   market value $226.75   unrealised P&L $-12.01
```

The unrealised loss is not a bug. Marked at the current bid, a position that
just crossed the spread to buy shows the spread as an immediate paper loss —
that is what a real arbitrage position looks like *before* the underlying
resolves and pays out the guaranteed $1 the basket was built to capture, not
after. Re-running the script immediately confirmed it does not pile into a
basket already held (0 new fills, same 4 on disk) and that restarting a
process picks the ledger up exactly where it left off.

Schedule it (cron, a systemd timer) to let it accumulate a real history rather
than running it once and calling that a track record.

### Running on a schedule

`scripts/run_paper_trade_polymarket.sh` wraps the example for cron. It exists
because a bare `python examples/paper_trade_polymarket.py` in a crontab line
gets two things wrong:

- **Overlap.** A tick takes roughly 2.5 minutes, almost all of it paginating
  the gamma API. DuckDB does not support concurrent writers to one file, so a
  run still in progress when the next one fires must be skipped, not started
  alongside it — two processes racing on the same database corrupt or error
  rather than merging. The wrapper holds a lock via `flock` on a dedicated file
  descriptor (not the `flock command...` form, so a held lock and a failed
  script stay distinguishable in the log) and skips cleanly if it can't get it.
- **Unbounded logs.** Left alone, appending forever eventually fills the disk.
  The wrapper truncates its own log to the last 20 MB before each run.

Installed with:

```bash
( crontab -l 2>/dev/null; echo "*/15 * * * * $(pwd)/scripts/run_paper_trade_polymarket.sh" ) | crontab -
```

Fifteen minutes gives roughly 6x margin over the run time. Logs land in
`logs/paper_trade_polymarket.log`; the account is `paper_polymarket.duckdb` in
the repo root. Both are gitignored — this is running state, not something to
commit. Inspect the book any time without disturbing the cron job:

```python
from sttbot.paper.account import PaperAccount
acc = PaperAccount("paper_polymarket.duckdb")
snap = acc.snapshot({})  # {} = unmarked; equity below excludes open positions
print(f"{len(acc.fills())} fills, cash ${snap.cash:,.2f}, "
      f"{len(acc.open_positions())} open positions")
```

Remove the schedule with `crontab -e` (delete the line) or `crontab -r` to
clear everything. On WSL, cron only runs while the WSL instance is up — it
does not survive a full Windows shutdown the way a real Linux host's cron
would.

## What statistical arbitrage actually works on Polymarket

A systematic sweep of every stat-arb class this venue could support, measured
against live depth rather than argued from theory. Datasets: a 20,503-event /
203,768-market open snapshot, **38,993 resolved markets with realized
outcomes**, and 863 of those with pre-resolution price history.

**Every pure-arbitrage class is exhausted.** Not "competitive" — exhausted,
by a structural mechanism described below.

| Class | Executable venue-wide | Verdict |
| --- | --- | --- |
| negRisk boundary baskets *(deployed)* | **$66** on $2,334 | Marginal, running |
| Threshold / scope / cross-event consistency | **$6.14** on $1,990 | <1% annualized |
| YES–NO complementarity | **$0** | Structurally impossible |
| Cross-venue vs Kalshi | $2,541 on **$810,916** | **−$86,227** vs T-bills |
| Calibration / favorite-longshot bias | none established | Prices efficient where liquid |

**YES–NO arbitrage cannot exist here.** Measured across liquid markets,
`NO_ask == 1 − YES_bid` and `NO_bid == 1 − YES_ask`, *exactly*, every time.
YES and NO are not two books to arbitrage between — the matching engine
mirrors one book. This is worth stating because treating them as independent
is a common starting assumption.

**Why arbitrage is structurally dead, not merely crowded.** Polymarket's fee
is quadratic and peaks at mid-price: a two-leg trade costs
`2 × 0.05 × 0.25 = 2.5¢/contract` at p=0.50. Any violation smaller than that
near the middle is unexploitable, so survivors are confined to the tails —
executed superset prices ran 0.010–0.100 — which is precisely where books are
thinnest. Fees consumed **63% of gross edge** on the trades that did execute.
The residual is not being competed away by faster traders; it is being eaten
by the fee schedule before anyone can reach it.

**Cross-venue is a rules problem wearing a pricing problem's clothes.** Of 30
randomly sampled "matches", only 14 were genuinely the same question. Every
large apparent edge was a criteria mismatch: an 84¢ iPhone "arb" was base
model vs product line; a 40¢ Trump impeachment edge was the single word
*removed* (impeached-and-convicted vs impeached) — the correct counterpart
market trades within a cent. Where the two venues disagree by more than 2¢,
they are describing different events. `venues.prediction`'s refusal to net
legs without an explicit equivalence declaration is doing real work.

**No calibration edge, and the apparent one was a trap.** Mid-range buckets
are near-perfectly calibrated (0.35–0.50: priced 0.438, realized 0.426, n=195;
0.50–0.65: priced 0.555, realized 0.542, n=190). Longshots *appeared*
underpriced by +13 to +26 points — the reverse of the textbook
favorite-longshot bias, and a headline result if true. It is not: the effect
strengthens monotonically with volume (+0.059 → +0.101 → +0.132 across volume
tiers), which is the signature of selection. A market only accumulates $100k
of volume when something dramatic happened, and "something happened"
correlates with YES resolving. At the moment you would place the bet, you do
not know the market will go on to trade $100k.

### The finding that reframes everything: capacity

Measured from 166 live order books, not from the venue's own metric.

- **`liquidityNum` overstates real 2¢ depth by ~48× venue-wide** ($622M claimed
  vs $13.0M measured). It ranks, it does not size — within the top 60 markets,
  where sizing decisions are made, its correlation with real depth collapses to
  Spearman +0.38 and the error spans four orders of magnitude.
- **24h volume does not predict depth at all** (Spearman **+0.06** within the
  top 60).
- **Median top-of-book notional in the venue's most active markets: $259.**
- **~99 markets venue-wide** can absorb a $10k buy within 2¢; **~11** can
  absorb $50k.
- The 0.40–0.60 price band holds 75,339 markets of which **3% traded**, median
  spread **94¢** — that is not a middle market, it is untouched books.
- Sports and Politics carry 80% of volume. Long-dated markets hold 39% of
  claimed liquidity on 9.5% of volume — that is where `liquidityNum` is parked
  and nothing trades.

**Realistic capacity for a strategy that takes liquidity and holds to
resolution: $250k–750k/day of executed notional, centred ~$400k**, across ~118
qualifying markets. Hard ceiling before you are the market: ~$3.3M/day.

That number is far larger than the arbitrage findings suggest, and it is the
actual answer to "what works here": **Polymarket has real capacity, but only
for strategies carrying a forecasting edge.** The free-lunch strategies are
measured out in tens to low-thousands of dollars. The venue will absorb
serious money — it just will not hand you any. And since prices are
well-calibrated wherever they are liquid, that edge has to come from
information or modelling, not from statistical patterns in the price series.

### Momentum and mean reversion: tested, and inside the costs

The remaining gap, now closed. Measured on **794 full hourly price paths** in a
strictly walk-forward panel: at each timestamp the signal uses only prior
prices and the payoff is the *next* window's move, so nothing about resolution
enters either side. Windows are stepped by the forward horizon so they never
overlap, and standard errors are clustered by market.

**Momentum does not exist here.** All eight horizons tested show *negative*
correlation between past and future moves (−0.027 to −0.109, t from −3.5 to
−8.9). Buying strength loses at every horizon. The tradeable direction, if
any, is to fade moves.

**Mean reversion exists, and is not tradeable.** At 24h/24h it grosses 3.5¢
against a 2.0¢ fee — apparently +1.6¢ of edge. It does not survive scrutiny:

| Test | Result |
| --- | --- |
| Delay entry by 1 hour | **36% of the edge vanishes** (3.5¢ → 2.2¢) |
| …net after fees | +0.25¢ |
| Add 1¢ round-trip spread | **−0.75¢** |
| Add 2¢ (typical tradeable book) | −1.75¢ |
| Horizon smoothness | peaks at 24h, insignificant by 48h (t=1.63), gone at 72h |
| Where it concentrates | 0.65–0.98, n≈300; the mid-range with the most data is insignificant (t=0.83) |

The one-hour-delay test is the decisive one. A third of the signal dying the
moment you stop trading at the same tick the signal ends is the signature of
**bid-ask bounce** — alternating trades at bid and ask manufacture negative
autocorrelation out of nothing, and no taker can harvest it. What survives is
+2.2¢ gross against a 2.0¢ fee, i.e. the entire effect sits inside the
transaction cost. Worse, the strategy is least profitable exactly where the
data is thickest: the quadratic fee peaks at mid-price (2.4¢ at 0.35–0.65,
where 1,124 of the observations are) and the only significant bands are the
thin ones at the extremes.

This is the same structural verdict as every other class here. The price
series genuinely contains reversion — the venue is not efficient in the strong
sense — but Polymarket's fee schedule is wider than the inefficiency.

## PEAD, finally wired to real earnings

`strategies.pead` has existed since the first commit and had never seen an
earnings surprise. It does now: `data.earnings` pulls Alpha Vantage earnings
(API key, or a cache written out-of-band) and joins them to keyless Yahoo
prices, and `backtest.event_study` measures the drift.

PEAD is among the most-published anomalies in finance and among the easiest to
fake, because every way of faking it is a timing error. Three are enforced in
code rather than left to the caller:

- **A post-market report is not tradeable that day.** Entry is the first
  session on or after `first_tradeable_date()`, which shifts post-market
  announcements to the next day and steps over weekends and holidays. Entering
  at the announcement close instead captures the overnight repricing and books
  it as drift — in this sample that gap averages **−1.03%**, comparable to the
  entire effect being measured.
- **SUE cannot see the future.** `trailing_sue` standardises by the previous
  eight surprises only. Using the full-sample dispersion tells a 2015 trade how
  volatile 2024 was.
- **Drift is net of the market.** A stock up 4% while the index rose 4% has
  drifted nowhere.

### The result: not enough data to answer

| SUE bucket | n | Mean excess (20d) | t | Entry gap |
| --- | --- | --- | --- | --- |
| < −1.5 | 6 | +1.85% | 0.39 | −0.81% |
| −1.5 … −0.5 | 12 | +2.91% | 1.19 | −4.75% |
| −0.5 … +0.5 | 46 | −0.73% | −0.76 | −1.58% |
| +0.5 … +1.5 | 26 | +1.05% | 0.90 | +0.30% |
| > +1.5 | 17 | +0.10% | 0.04 | +0.96% |

Long-short at the strategy's default 1.5 threshold: **−0.41%, t = −0.20**.

PEAD predicts that column rises monotonically. It does not — the *biggest
misses drifted up*, which is the opposite. But the honest reading is not "PEAD
is dead": it is that **107 events across 2 symbols cannot test this
hypothesis**, and the shape shown is what noise looks like. A 5-session horizon
prints +2.15% at t=2.35, which is exactly the kind of number to distrust: one
of four horizons tried, on 23 events, not persisting at 10, 20 or 40 sessions.

The binding limit is the data channel, not the code. Earnings require a
credential, and without `ALPHAVANTAGE_API_KEY` they can only arrive through a
channel that passes through a context window, which caps the universe at a
dozen symbols. With a key the same example runs on thousands:

```bash
python examples/pead_event_study.py --symbols AAPL,MSFT,CULP,SCVL,...
```

Worth noting for whoever runs it at scale: the thesis is specifically about
*micro-caps*, where institutional non-coverage lets drift persist. This sample
is half mega-cap (IBM), where PEAD should be arbitraged away, and half
micro-cap (CULP) — too small to split. A real test needs the universe skewed
small, and should expect the short leg to be expensive or impossible to borrow
in exactly the names where the effect is supposed to live.

## Polymarket earnings markets: the forecasting edge, priced

The research above concluded this venue pays for a view, not for arbitrage.
Polymarket runs ~41 live *"Will X beat quarterly earnings?"* contracts, which
is a forecasting question with a cheap answer: a company's own record of
beating consensus. `strategies.earnings_market` prices them and
`examples/polymarket_earnings.py` runs it against the live venue, optionally
feeding the paper account.

**The base rate is the whole strategy, and it is easy to measure on the wrong
population.** Measured on real histories:

| Universe | Beat rate |
| --- | --- |
| IBM (mega-cap) | 87.9% |
| HD (large-cap) | 84.5% |
| CULP (micro-cap) | **53.4%** |
| Polymarket's median ask on these markets | ~0.90 |

Polymarket lists only large, well-covered names — the ones that guide analysts
down and beat ~86% of the time. I built the pricer, pooled all three cached
companies into the prior, and it returned a confident **BUY NO at +3.5¢** on a
live Home Depot market. Under the correct large-cap prior the same market is
**−0.1¢: no trade.** One unlisted micro-cap in the average moved the prior 11
points and manufactured the entire signal.

That failure does not look like a bug. It looks like alpha. The prior is now
computed strictly from the tickers that actually have live markets, and a test
reproduces the contamination case directly.

Two further guards, both from lessons measured elsewhere in this repo:

- **Edge is scored against bid and ask, never the mid.** Both sides cross the
  spread, so with a 5¢ spread and the quadratic fee the model must disagree
  with the market by roughly 4–6¢ before either side clears. A 16¢-spread
  market like DKS is untradeable at any conviction.
- **Ties count as misses**, because the contract says *beat*, not *meet or
  beat* — and large caps hit consensus exactly quite often (HD did it twice in
  58 quarters, IBM three times). Whether Polymarket resolves an exact meet as a
  beat is worth **~4 percentage points**, more than any edge the model claims.
  Confirm it per market before trading; the cross-venue study already showed
  resolution wording dominating apparent pricing edges.

The model does carry information — Brier skill of **+0.09** against always
quoting the base rate, walk-forward — so trailing beat history genuinely
predicts the next beat. But that is 162 observations across 3 companies, and
on the one live market priceable today it produces no trade. The market is
approximately right, which is what the calibration study predicted for a venue
that is efficient wherever it is liquid.

Scaling this needs earnings history for the ~40 listed tickers, which needs
`ALPHAVANTAGE_API_KEY`. Without it the cache can only be filled a symbol at a
time.

## Settlement: without it the paper book cannot be measured

A hold-to-resolution strategy pays cash out to buy a basket and collects $1 a
set when it resolves. `sttbot.paper` did the first half and never the second,
so cash could only leave the account and the reported return drifted downward
regardless of whether the strategy worked. The live book showed **−11.7% while
being worth +0.8%** once its resolved legs were valued.

`paper.settlement` closes the loop. A settlement is recorded as an ordinary
fill — a sale of the whole position at the settled price — so cash, average
cost and realised P&L all fall out of the existing ledger with no
special-casing. Redemption is not a trade, so no fee is charged.

**The hard part is deciding what "resolved" means, and the venue does not say
it directly.** `closed == true` is *not* sufficient. Surveyed across 60 closed
Polymarket markets:

| `outcomePrices` | Count | What it actually is |
| --- | --- | --- |
| `["0", "0"]` | 36 | closed, no payout written |
| `["0.58", "0.42"]` | 1 | the last price someone traded at |
| `["0", "1"]` / `["0.000001", "0.999999"]` | rest | a real settlement |

Settling on the first would **zero out every leg of a basket including the
winner**, turning a $1 payout into nothing. Settling on the second books a mark
as if it were an outcome. So a resolution is accepted only when the prices form
an actual binary payout — summing to 1, one leg ≈1 and the rest ≈0. Anything
else leaves the position open and is reported, because an unsettled position is
a knowable state and a wrongly settled one is a silent, permanent error.

Two API details that are silent traps: gamma's `clob_token_ids` filter takes
**repeated parameters** (a comma-separated list is rejected as "invalid clob
token ids"), and **`closed=true` is required** or resolved markets are filtered
out of the default view and every lookup returns empty.

Run live against the account, it settled 12 positions across 3 baskets for
exactly their face value:

| Basket | Legs | Cost | Payout | Profit |
| --- | --- | --- | --- | --- |
| KAROL G album sales | 6 | $4.82 | $5.00 | **+$0.18** |
| LASK Linz 2nd half | 3 | $10.00 | $10.00 | +$0.00 |
| Fiorentina 2nd half | 3 | $5.00 | $5.00 | +$0.00 |

That is the arbitrage doing exactly what it claims — and the first time this
repo has booked a realised outcome rather than a mark.

## Faster profits: the horizon is where the edge lives

The paper book's capital sits idle because several baskets do not resolve until
January 2027. Two ways to speed that up were priced, and both fail — for
opposite reasons that turn out to be the same reason.

**Exiting early costs 6% against a 1% edge.** Priced against the live book, one
basket at a time:

| | |
| --- | --- |
| Sell the whole book at current bids | **−$81.08** |
| Hold every basket to resolution | **+$8.59** |
| Cost of scalping instead of holding | **−$89.67** |

Only 1 of 22 baskets could be exited at a profit. The cost scales with leg
count — 3-leg baskets lose $2–7 on exit, 7-leg baskets lose up to $27 — because
**a multi-leg basket crosses the spread on every leg, twice.** The edge is ~1%
of the basket; the round trip is ~6%.

**Filtering to short-dated markets removes the edge entirely.** The obvious
alternative is to stop entering long-dated baskets. Measured across 7,170
complete baskets in one venue-wide scan:

| Horizon | Baskets | With edge | Hit rate |
| --- | --- | --- | --- |
| past due | 70 | 0 | 0% |
| < 1 day | 417 | 0 | 0% |
| 1–7 days | 3,557 | **0** | 0% |
| 7–30 days | 2,866 | 0 | 0% |
| 30–180 days | 236 | 1 | 0.42% |
| 180 days+ | 22 | 1 | **4.55%** |

**Zero positive-edge baskets in 4,044 markets resolving inside a week.** The
hit rate only becomes non-zero past 30 days and is highest at 180 days+. Two
successes is a thin sample, but zero out of 4,044 is a real absence rather than
noise — at even a 0.1% hit rate you would expect four.

`--max-days` exists and is tested, but **defaults to off**, because setting it
to 7 does not make the strategy faster, it stops it trading.

The economics are consistent with everything else measured here: the
mispricing survives precisely where nobody is watching, and not being watched
is the same property that makes a market slow to resolve and expensive to
leave. Speed and edge are not independent dials on this venue — they are the
same dial, pointing opposite ways.

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
