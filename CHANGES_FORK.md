# Changelog – Learning engine integration

This fork adds two changes so the bot can learn from past trades and improve signal weights over time.

## 1. Pass `signal_sources` in trade metadata

**File:** `bot.py` (in `_record_paper_trade`)

When a paper trade is recorded, the list of signal sources that contributed to the fused signal is now stored in the trade metadata:

- **Before:** `metadata` only had `simulated`, `num_signals`, and `fusion_score`.
- **After:** `metadata` also includes `signal_sources`: a list of processor names (e.g. `["SpikeDetection", "OrderBookImbalance", "TickVelocity"]`).

The learning engine uses this to attribute wins and losses to each signal source and to adjust fusion weights. Without it, weight optimization had no per-source data and could not run meaningfully.

## 2. Run `optimize_weights()` on a schedule

**File:** `bot.py`

- **New state:** `_last_weight_optimize` is set in the strategy `__init__` and used to remember the last time weights were optimized.
- **Timer loop:** In `_timer_loop` (which runs every 10 seconds), the bot now checks whether at least 24 hours have passed since the last optimization. If so, it calls `await self.learning_engine.optimize_weights()` and updates `_last_weight_optimize`.

So the learning engine runs automatically once per day. It does not place or cancel orders; it only updates the fusion engine weights used for future trading decisions.

## Summary

- **Tracking:** Each recorded paper trade now includes which signals were involved (`signal_sources`).
- **Learning:** Once every 24 hours the bot runs the existing learning logic to recompute and apply better weights.
- **Safety:** No change to order placement, risk checks, or execution path; only extra metadata and a periodic, read-only optimization step.
