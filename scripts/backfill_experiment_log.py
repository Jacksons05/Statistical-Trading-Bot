"""One-off backfill: log this repo's real historical strategy trials.

The experiment registry (`sttbot.research.experiment_log`, added
2026-08-20) postdates most of this repo's actual research. That work still
happened -- and is verifiable in `git log` -- so it belongs in the registry
retroactively rather than leaving the log starting empty as if no trials
existed before the tool did. Every number below is transcribed from the
commit message of the commit that produced it (cited by short SHA); nothing
here is estimated or invented. Where a commit reports a negative or
inconclusive result, it is logged as exactly that -- the registry's value is
recording failed hypotheses too, not just the wins.

Run once:

    python scripts/backfill_experiment_log.py

Idempotent in the sense that re-running appends the same records again
(the log is append-only by design, per experiment_log.py) -- so don't re-run
against a log that already has this history in it. It's a backfill script,
not a scheduled job.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sttbot.research.experiment_log import ExperimentRecord, append_experiment

LOG_PATH = Path(__file__).resolve().parent.parent / "experiments" / "sttbot_history.jsonl"


def _ts(iso: str) -> str:
    return dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc).isoformat()


RECORDS = [
    ExperimentRecord(
        strategy="dixon_coles",
        hypothesis="A weighted-MLE Dixon-Coles fit produces a profitable "
                   "1X2 betting strategy on real football odds after friction.",
        dataset_version="football_matches (bulk open dataset), 7 seasons x 7 divisions",
        train_interval="trailing window, refit per matchday (walk-forward)",
        validation_interval="n/a -- single walk-forward pass, no separate validation split",
        test_interval="7 real seasons, 7 divisions",
        parameters={"weighting": "exponential time decay", "optimizer": "L-BFGS-B",
                    "identifiability": "mean(attack)=0, n-1 free attack params"},
        search_method="single run, no hyperparameter search performed or logged",
        execution_assumptions="bets pre-match line, settles at that price; FrictionModel "
                               "charges the same cost to the entry threshold and the P&L",
        results={
            "profitable": False,
            "note": "every division loses money; CLV vs a single book's close is negative "
                    "except in the two thinnest markets tested (Scottish tiers, t=5.1 and "
                    "t=4.1 vs the random-selection baseline), and even there the skill is "
                    "worth less than the margin plus commission",
            "sc3_roi_pct": 3.12,
            "sc3_significance": "0.5 standard errors from zero; noise, not an edge",
        },
        costs={"friction_model": "economics.friction.FrictionModel"},
        influenced_later_decisions=True,
        notes="Motivated the CLV-baseline discipline (backtest/clv.py) and the "
              "thin-markets thesis tested next (673d7bc).",
        timestamp=_ts("2026-07-28T10:09:41-04:00"),
        code_version="edcca80",
    ),
    ExperimentRecord(
        strategy="dixon_coles / clv_skill",
        hypothesis="CLV skill increases with market thinness (proxied by mean "
                   "single-book overround), pre-specified before viewing the "
                   "out-of-sample result.",
        dataset_version="football-data.co.uk, 22 divisions with closing odds, ~53,000 matches",
        train_interval="7 divisions used to form the hypothesis (not part of this test)",
        validation_interval="n/a",
        test_interval="15 out-of-sample divisions",
        parameters={"test": "Spearman correlation, thinness proxy vs CLV skill",
                    "n_divisions_tested": 15},
        search_method="single pre-registered test (ONE Spearman correlation), no search",
        execution_assumptions="odds-matched CLV baseline (not just uniform-random) to "
                               "control for which prices the model favors",
        results={
            "spearman_rho": 0.539,
            "p_value": 0.038,
            "significant_divisions": "14 of 15, t>2; English National League strongest at t=10.2",
            "profitable_divisions": 0,
            "skill_magnitude_pct": "0.3 to 0.75 percentage points, vs a 5-9% overround",
            "confound_check": "longshot-only pseudo-strategy scores large skill against "
                               "uniform baseline, exactly zero against odds-matched baseline",
            "mechanism_caveat": "thin markets also have wider CLV dispersion (rho=0.661 with "
                                 "overround); normalizing keeps the point estimate but loses "
                                 "significance at n=15",
        },
        influenced_later_decisions=True,
        notes="Confirms genuine, statistically real selection skill in thin markets that "
              "still does not clear transaction costs -- the load-bearing null result "
              "behind this repo's 'capacity, not calibration' thesis.",
        timestamp=_ts("2026-07-29T21:53:23-04:00"),
        code_version="673d7bc",
    ),
    ExperimentRecord(
        strategy="prob_arbitrage (Polymarket whole-venue scan)",
        hypothesis="Complete-basket (negRisk) and cross-outcome boundary arbitrage exists "
                    "at meaningful scale across the live Polymarket venue.",
        dataset_version="live gamma API snapshot: 16,241 open events / 144,876 markets",
        train_interval="n/a -- live snapshot scan, not a fitted model",
        validation_interval="n/a",
        test_interval="single snapshot, re-priced against real CLOB depth (not top-of-book)",
        parameters={"basket_completeness": "requires every outcome leg quoted, not a subset"},
        search_method="exhaustive scan of the live venue",
        execution_assumptions="executable_arbitrage walks real order-book depth and stops at "
                               "the first size it cannot fill",
        results={
            "total_executable_arbitrage_usd": 66,
            "capital_locked_usd": 2334,
            "largest_opportunity_contracts": 950,
            "pct_markets_zero_24h_volume": 91.1,
            "median_market_liquidity_usd": 18,
            "scalping_screen_after_realistic_filters": "57 of 144,876 markets",
            "naive_subset_scoring_false_positive_count": 165,
        },
        influenced_later_decisions=True,
        notes="Established that gamma top-of-book prices carry no size and must be "
              "re-priced against real depth -- now a structural design constraint "
              "throughout venues/polymarket.py.",
        timestamp=_ts("2026-07-29T22:42:34-04:00"),
        code_version="28c0b40",
    ),
    ExperimentRecord(
        strategy="prob_arbitrage + venues.prediction (systematic stat-arb sweep)",
        hypothesis="At least one class of pure statistical arbitrage (negRisk, logical "
                    "consistency, cross-venue vs Kalshi) is exploitable on Polymarket "
                    "after fees.",
        dataset_version="20,503-event open snapshot; 38,993 resolved markets with realized "
                         "outcomes; 863 with pre-resolution price history; 166 live order books",
        train_interval="n/a -- direct measurement against live/historical depth and outcomes",
        validation_interval="n/a",
        test_interval="single systematic sweep across all classes",
        parameters={"classes_tested": ["negRisk baskets", "logical consistency (YES-NO)",
                                        "cross-venue vs Kalshi", "calibration by volume tier"]},
        search_method="exhaustive sweep, not a parameter search",
        execution_assumptions="depth-walked executable sizing; cross-venue matches manually "
                               "checked for resolution-criteria equivalence",
        results={
            "negrisk_arbitrage_usd": 66,
            "logical_consistency_arbitrage_usd": 6.14,
            "yes_no_arbitrage": "structurally impossible",
            "cross_venue_vs_kalshi_gross_usd": 2541,
            "cross_venue_capital_locked_usd": 810916,
            "cross_venue_locked_days": 775,
            "cross_venue_net_vs_tbills_usd": -86227,
            "cross_venue_matched_question_rate": "14 of 30 sampled pairs were the same question",
            "fee_share_of_gross_edge_pct": 63,
            "calibration_edge": "none in liquid mid-range buckets",
            "apparent_longshot_underpricing_pts": "13-26, found to be a volume-tier "
                                                    "selection artifact, not alpha",
            "realistic_daily_capacity_usd": "250,000-750,000",
            "liquidity_num_overstatement_factor": 48,
            "median_top_of_book_notional_top_markets_usd": 259,
        },
        influenced_later_decisions=True,
        notes="The headline finding that reframes the rest of this repo's Polymarket "
              "work: pure arbitrage is structurally capped by the quadratic fee near "
              "mid-price; any real edge has to come from forecasting, not price patterns.",
        timestamp=_ts("2026-08-14T09:54:20-04:00"),
        code_version="402eb97",
    ),
    ExperimentRecord(
        strategy="prob_arbitrage (momentum / mean-reversion)",
        hypothesis="Short-horizon price momentum or mean-reversion in Polymarket price "
                    "series is tradeable after transaction costs.",
        dataset_version="794 full hourly price paths, strictly walk-forward",
        train_interval="signal uses only prior prices",
        validation_interval="n/a",
        test_interval="8 horizons, non-overlapping windows stepped by the forward horizon",
        parameters={"horizons_tested": 8, "clustering": "standard errors clustered by market"},
        search_method="8 horizons tested, all reported (not cherry-picked)",
        execution_assumptions="fee = 2.0c reference; spread/delay stress tests applied "
                               "post-hoc to the one surviving horizon",
        results={
            "momentum_exists": False,
            "momentum_correlation_range": "-0.027 to -0.109, t from -3.5 to -8.9 (negative "
                                            "at every horizon)",
            "mean_reversion_exists": True,
            "mean_reversion_tradeable": False,
            "24h_gross_edge_c": 3.5,
            "24h_fee_c": 2.0,
            "24h_net_after_1h_entry_delay_c": 0.25,
            "24h_net_after_1c_roundtrip_spread_c": -0.75,
            "24h_net_at_realistic_2_4c_spread_c": -1.75,
            "significance_by_horizon": "peaks at 24h, insignificant by 48h (t=1.63), gone by 72h",
            "mechanism": "bid-ask bounce (alternating trades at bid/ask), not harvestable "
                         "by a taker",
        },
        influenced_later_decisions=True,
        notes="Closes the last untested pure-price stat-arb class with a clean negative "
              "result: the venue is not strong-form efficient (reversion is real) but the "
              "fee schedule is wider than the inefficiency.",
        timestamp=_ts("2026-08-14T10:10:36-04:00"),
        code_version="d2489a5",
    ),
    ExperimentRecord(
        strategy="pead",
        hypothesis="Post-Earnings-Announcement Drift, measured with correct entry timing "
                   "(first tradeable session, not announcement close), is present and "
                   "monotone in SUE bucket.",
        dataset_version="Alpha Vantage earnings + Yahoo prices, 107 events / 23 symbols",
        train_interval="n/a -- event study, not a fitted model",
        validation_interval="n/a",
        test_interval="107 events; drift also tested at 5/10/20/40-session horizons on 23 events",
        parameters={"sue_window": "trailing 8 prior surprises only", "strategy_threshold": 1.5},
        search_method="4 horizons tried, all reported",
        execution_assumptions="entry at first_tradeable_date(); benchmark-netted over "
                               "identical sessions; announcement-day gap (-1.03% avg) "
                               "reported separately, never booked as drift",
        results={
            "sue_buckets_monotone": False,
            "note": "biggest misses drifted up, opposite of the prediction",
            "long_short_at_threshold_pct": -0.41,
            "long_short_t_stat": -0.20,
            "5_session_horizon_pct": 2.15,
            "5_session_horizon_t_stat": 2.35,
            "persists_at_10_20_40_sessions": False,
            "conclusion": "inconclusive -- n too small (2 symbols at the strategy threshold) "
                          "to test the hypothesis; the one nominally-significant horizon is "
                          "exactly the kind of number this repo's own discipline says to distrust",
        },
        influenced_later_decisions=True,
        notes="Sample is half mega-cap (data-access constrained by ALPHAVANTAGE_API_KEY "
              "availability), where PEAD should already be arbitraged away; repo notes it "
              "should be re-run on micro-caps with a key.",
        timestamp=_ts("2026-08-14T11:33:09-04:00"),
        code_version="98d4b90",
    ),
    ExperimentRecord(
        strategy="earnings_market",
        hypothesis="A shrinkage-toward-base-rate model of trailing beat rate prices "
                   "Polymarket's 'will X beat earnings' contracts better than the market, "
                   "net of the correct bid/ask spread and the 'beat' (not meet-or-beat) "
                   "resolution rule.",
        dataset_version="real trailing earnings histories, 3 cached companies (IBM, HD, CULP) "
                        "+ Polymarket's live ~41 beat-earnings contracts",
        train_interval="trailing beat-rate history per company",
        validation_interval="walk-forward Brier-skill evaluation, 162 observations / 3 companies",
        test_interval="1 live, priceable Home Depot market at capture time",
        parameters={"prior": "shrinkage toward base rate of tickers with live Polymarket "
                              "markets only (not all cached tickers)"},
        search_method="single specified model, no search",
        execution_assumptions="edge scored against bid/ask, not mid; ties count as misses "
                               "per the contract's literal wording",
        results={
            "brier_skill_vs_base_rate": 0.09,
            "n_observations": 162,
            "n_companies": 3,
            "live_tradeable_signal": False,
            "note": "correct large-cap-only prior produces -0.1c edge on the one live HD "
                    "market (no trade); the wrong pooled-prior version had produced a "
                    "false +3.5c BUY NO, caused by one micro-cap ticker with no live "
                    "market contaminating the prior by 11 points",
            "resolution_rule_effect_pct": 4,
        },
        influenced_later_decisions=True,
        notes="Model carries genuine walk-forward information (Brier skill +0.09) but the "
              "one live market it can be tested against today produces no trade -- 'the "
              "market is approximately right', consistent with the calibration finding in "
              "402eb97 that Polymarket is well-priced wherever liquid.",
        timestamp=_ts("2026-08-14T15:06:13-04:00"),
        code_version="f4586d0",
    ),
    ExperimentRecord(
        strategy="token_screen",
        hypothesis="A rug/honeypot pre-trade screen combined with depth-capped exit sizing "
                   "identifies tradeable fresh Solana token launches.",
        dataset_version="live DexScreener discovery feed, 24 discovered tokens",
        train_interval="n/a -- rule-based screen, not fitted",
        validation_interval="n/a",
        test_interval="single live run",
        parameters={"screen_thresholds": "default ScreenThresholds", "impact_budget_pct": 2},
        search_method="none -- fixed screen thresholds",
        execution_assumptions="exit sized against real pool depth via a 2% price-impact budget",
        results={
            "n_discovered": 24,
            "n_passed": 0,
            "n_rejected": 24,
            "primary_rejection_reasons": "under 24h old or under $25k liquidity",
            "deepest_token_2pct_exit_usd": 323,
            "example_volume_vs_liquidity_mismatch": "RIKA: $1.3M/day volume against a pool "
                                                      "absorbing only $173 on exit",
        },
        influenced_later_decisions=True,
        notes="'The screen working as avoidance rather than selection' -- established that "
              "for this asset class, capacity (not selection) is the binding constraint, "
              "motivating a8d34f6 and 7aa1ca8.",
        timestamp=_ts("2026-07-29T22:48:28-04:00"),
        code_version="73ad40a",
    ),
    ExperimentRecord(
        strategy="token_screen (established-but-thin hypothesis)",
        hypothesis="Tokens that survive their first month occupy a sweet spot: liquid "
                    "enough to size an exit, still inefficient enough (thin) to trade.",
        dataset_version="GeckoTerminal volume-ranked pools, 96 established tokens (1-6 months old)",
        train_interval="n/a -- cross-sectional measurement",
        validation_interval="n/a",
        test_interval="single cross-sectional pass",
        parameters={"impact_budget_pct": 2, "turnover_metric": "24h volume / pool liquidity"},
        search_method="none",
        execution_assumptions="capacity measured via 2% price-impact exit sizing",
        results={
            "sweet_spot_exists": False,
            "median_exit_under_1_day_usd": 160,
            "median_exit_1_to_6_months_usd": 26892,
            "capacity_improvement_factor": 150,
            "n_clearing_5k_exit_and_0.5x_turnover": 1,
            "that_one_token": "SOL, $26M liquidity -- the opposite of thin",
            "pct_established_tokens_turnover_under_0.1x_per_day": 90,
            "pct_established_tokens_in_25k_to_1M_thin_band": 90,
            "mechanism": "thin + active is not a stable state -- it either attracts "
                         "liquidity and stops being thin, or activity dies and depth "
                         "strands",
        },
        influenced_later_decisions=True,
        notes="Directly echoes the same structural finding as 402eb97 (91% of Polymarket "
              "markets traded $0): inefficiency is easy to find precisely where nobody "
              "trades, which is why it survives unexploited.",
        timestamp=_ts("2026-07-29T23:31:07-04:00"),
        code_version="a8d34f6",
    ),
    ExperimentRecord(
        strategy="breadth",
        hypothesis="A many-names/small-size portfolio can convert per-name capacity "
                   "limits into a viable strategy through width rather than depth.",
        dataset_version="pool reserves math (exact) + assumed payoff tail (Monte Carlo, "
                        "explicitly not fitted to realized returns)",
        train_interval="n/a",
        validation_interval="n/a",
        test_interval="12-round Monte Carlo from $10,000 starting capital; 1 live pipeline run",
        parameters={"assumed_survival_rate_pct": 3, "roundtrip_cost_pct": "2-4",
                     "names_for_95pct_confidence": 99},
        search_method="none -- parameters stated explicitly as assumptions, not fit",
        execution_assumptions="round-trip cost simulated exactly through pool reserves "
                               "(both legs priced against the same unmoved pool, "
                               "regression-tested against the reversed-impact bug)",
        results={
            "breakeven_multiple_at_3pct_survival": 33.3,
            "live_pipeline_book_size": 2,
            "miss_probability_20_name_book_pct": 54,
            "miss_probability_2_name_book_pct": 94,
            "monte_carlo_median_outcome_usd": 88,
            "monte_carlo_mean_outcome_usd": 245859,
            "monte_carlo_ruin_probability_pct": 85,
            "conclusion": "positive expectancy that loses almost every time and pays only "
                          "through a tail this repo could not yet measure; execution is "
                          "not implemented, example prints intended orders only",
        },
        influenced_later_decisions=True,
        notes="The unmeasured tail parameter this trial depends on is exactly what "
              "ed13f2e / d7597cf / 5386072 then tried, and failed, to pin down.",
        timestamp=_ts("2026-07-29T23:45:00-04:00"),
        code_version="7aa1ca8",
    ),
    ExperimentRecord(
        strategy="cohort (token payoff tail)",
        hypothesis="The token payoff tail (Hill alpha) that breadth's Monte Carlo assumed "
                   "can be measured from a live-captured birth cohort.",
        dataset_version="616 pools captured live over a 25-minute window, 2026-07-30",
        train_interval="n/a",
        validation_interval="n/a",
        test_interval="single cohort, first-pass measurement then adversarial re-review",
        parameters={"defenses_added": ["skip_first_candle", "min_peak_volume",
                                         "confirm_peak_fraction"]},
        search_method="single measurement, five independent adversarial checks applied "
                       "after the first result looked too good",
        execution_assumptions="candle-history resolution rather than a later re-check, "
                               "specifically to survive delisted failures",
        results={
            "first_pass_hill_alpha": 0.59,
            "first_pass_implied": "infinite mean",
            "first_pass_best_peak_multiple": 1802,
            "first_pass_pct_reaching_100x": 2.65,
            "verdict_after_adversarial_review": "mostly artifact, biased toward flattering "
                                                  "the strategy",
            "pct_pools_peak_in_birth_candle": 70.6,
            "largest_peak_was": "one $0.00074-volume tick in a pool with $6.65 lifetime volume",
            "own_bug_found": "pool_ohlcv returned [] on rate-limit failure, misclassifying "
                              "164/300 pools as never-traded and silently deleting a third "
                              "of the cohort; fixed to return None for a failed fetch",
            "wash_trading_found": "3 clone mints (XAU/SOL), created 18s apart, volumes "
                                   "within 4 significant figures of each other",
            "hill_alpha_after_defenses": "0.93 to 1.06, straddling the finite/infinite-mean line",
            "conclusion": "this cohort (616 pools, <1hr observed) cannot resolve the "
                          "parameter breadth depends on; no mean should be quoted when "
                          "deleting one dust tick moves it 3 orders of magnitude",
        },
        influenced_later_decisions=True,
        notes="A negative methodological result about the data, not just the strategy: "
              "establishes how to measure a fat-tail parameter without fooling yourself, "
              "which the follow-up (d7597cf, 5386072) then builds on.",
        timestamp=_ts("2026-07-30T01:39:28-04:00"),
        code_version="ed13f2e",
    ),
    ExperimentRecord(
        strategy="cohort (survival resolution)",
        hypothesis="The same 2026-07-30 birth cohort, re-resolved after 3.6 days with "
                   "hourly candles covering full pool lifetimes, will pin down the tail "
                   "index and validate breadth's exit-on-take-profit assumption.",
        dataset_version="same 616-pool cohort, re-resolved at 3.6 days, 610/616 pools fetched",
        train_interval="n/a",
        validation_interval="n/a",
        test_interval="3.6-day resolution window",
        parameters={"outcome_taxonomy": ["died (price)", "stopped_trading (silence)"]},
        search_method="none",
        execution_assumptions="hourly candles spanning each pool's whole life",
        results={
            "tail_index_pinned_down": False,
            "hill_alpha_range_across_defensible_rules": "0.20 to 1.03",
            "death_by_price_pct": 4.3,
            "death_by_silence_24h_pct": 95.6,
            "pct_producing_exactly_one_candle_ever": 69.6,
            "median_hours_since_last_trade": 87,
            "still_tradeable_at_3.6_days": "27 of 610",
            "prior_survival_stats_were_censoring_artifacts": True,
            "prior_reported_survival_pct": "97.3% at 45min, 98.3% at 3.6 days -- both wrong: "
                                            "a token whose market stops producing candles "
                                            "looks 'alive' at its frozen last price forever",
            "conclusion": "kills breadth independent of the tail-index question: 95.6% of "
                          "positions have no counterparty to exit into within a day, so a "
                          "take-profit exit rule cannot run regardless of the alpha estimate",
        },
        influenced_later_decisions=True,
        notes="This is the trial that actually ends the breadth/meme-coin line of work -- "
              "not an inconclusive tail estimate, but a structural finding (no exit "
              "liquidity) that makes the tail-index question moot.",
        timestamp=_ts("2026-08-02T16:47:46-04:00"),
        code_version="5386072",
    ),
    ExperimentRecord(
        strategy="market_making (prediction-market scalping)",
        hypothesis="Fee-aware, inventory-skewed two-sided quoting on binary prediction "
                   "contracts is profitable once adverse selection (markout) is measured "
                   "rather than assumed away.",
        dataset_version="simulated price paths (offline), not live-captured order flow",
        train_interval="n/a -- mechanism/parameter test, not a fitted model",
        validation_interval="n/a",
        test_interval="simulated paths under two regimes: quoting around last mid, "
                      "slow drift",
        parameters={"risk_aversion_corrected": 0.05, "risk_aversion_original_bug": 1.0},
        search_method="none -- deliberate mechanism tests, not a parameter sweep",
        execution_assumptions="mark-out P&L (adverse selection measured, not assumed away); "
                               "Kalshi quadratic fee paid on both legs",
        results={
            "risk_aversion_bug_found": "default was mis-scaled at 1.0, shifting quotes 25c "
                                        "at full inventory against a ~1c spread; corrected "
                                        "to 0.05",
            "pnl_at_buggy_default": -5.05,
            "pnl_at_corrected_default": 2.01,
            "kalshi_breakeven_spread_at_50c_c": 3.50,
            "kalshi_breakeven_spread_at_5c_c": 0.67,
            "pct_book_untradeable_at_1c_spread": 84,
            "properties_confirmed": [
                "quoting around last mid loses whenever price swings wider than the spread",
                "slow drift never fills a maker that re-quotes every tick",
            ],
        },
        influenced_later_decisions=True,
        notes="Not a profitability trial by itself -- a mechanism/parameter-correctness "
              "trial that found and fixed a real sign/scale bug before any capital-facing "
              "conclusion could be drawn from this module.",
        timestamp="2026-07-29T22:53:44+00:00",
        code_version="422094b",
    ),
]


def main() -> None:
    for record in RECORDS:
        append_experiment(record, LOG_PATH)
    print(f"Wrote {len(RECORDS)} historical records to {LOG_PATH}")


if __name__ == "__main__":
    main()
