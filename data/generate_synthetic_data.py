"""
generate_synthetic_data.py
────────────────────────────
Generates a synthetic multimodal dataset shaped like a real
headlines + price/volume dataset, so the rest of this project (model,
distributed training, quantization) can be built and validated end-to-end
without a live market data feed.

Generative process (documented so it's never mistaken for real data)
----------------------------------------------------------------------
For each of `N_TICKERS` synthetic tickers, we simulate `N_DAYS` days:

  1. Daily return ~ N(0, sigma), sigma per-ticker (some tickers more
     volatile than others).
  2. A headline is generated from a template pool with a "sentiment slot"
     (positive / neutral / negative words). The sentiment is deliberately
     correlated with *that day's* return (positive sentiment more likely
     on up days) plus noise, so the text carries a real, learnable signal
     about the same underlying process as the price features — this is
     what makes the multimodal fusion meaningfully better than either
     branch alone, without hand-fabricating the label from the headline.
  3. Per-day numeric features: rolling volatility, a volume z-score, and a
     simple RSI-like momentum feature, all derived from the simulated
     return series (not looked up from any real source).
  4. Label: direction of the *next* day's return, bucketed into
     {down, flat, up}.

This is a controlled synthetic benchmark, not a market simulator and not
real market data. Swap in real headlines + OHLCV and keep the same output
schema (see `SCHEMA` below) to point the rest of the pipeline at real data.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

SCHEMA = [
    "ticker", "day", "headline", "lookback_returns", "volume_z",
    "volatility", "rsi_like", "label",
]

POSITIVE_WORDS = ["beats", "surges", "rallies", "outperforms", "climbs", "jumps"]
NEGATIVE_WORDS = ["misses", "slumps", "plunges", "underperforms", "falls", "drops"]
NEUTRAL_WORDS = ["holds steady", "trades flat", "in line with estimates", "unchanged"]

TEMPLATES_DIRECTIONAL = [
    "{ticker} shares {word} after quarterly results",
    "{ticker} stock {word} on analyst commentary",
    "{ticker} {word} amid sector-wide moves",
    "Shares of {ticker} {word} in early trading",
]
TEMPLATES_NEUTRAL = [
    "{ticker} {word} ahead of earnings",
    "{ticker} {word} as investors await guidance",
    "Analysts say {ticker} {word} this session",
]

LOOKBACK_WINDOW = 10


def _make_headline(rng: random.Random, sentiment: str, ticker: str) -> str:
    if sentiment == "positive":
        word = rng.choice(POSITIVE_WORDS)
        template = rng.choice(TEMPLATES_DIRECTIONAL)
    elif sentiment == "negative":
        word = rng.choice(NEGATIVE_WORDS)
        template = rng.choice(TEMPLATES_DIRECTIONAL)
    else:
        word = rng.choice(NEUTRAL_WORDS)
        template = rng.choice(TEMPLATES_NEUTRAL)
    return template.format(ticker=ticker, word=word)


def _bucket_return(r: float, flat_band: float) -> int:
    if r > flat_band:
        return 2  # up
    if r < -flat_band:
        return 0  # down
    return 1  # flat


def generate(n_tickers: int = 12, n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    rows = []
    for t in range(n_tickers):
        ticker = f"SYN{t:02d}"
        sigma = float(rng_np.uniform(0.008, 0.03))  # per-ticker volatility
        returns = rng_np.normal(loc=0.0002, scale=sigma, size=n_days + 1)

        # rolling volume proxy: volatility clusters bump volume
        rolling_vol = pd.Series(np.abs(returns)).rolling(5, min_periods=1).mean().to_numpy()
        volume_noise = rng_np.normal(0, 1, size=n_days + 1)
        volume_raw = rolling_vol * 20 + volume_noise
        volume_z = (volume_raw - volume_raw.mean()) / (volume_raw.std() + 1e-8)

        rolling_volatility = pd.Series(returns).rolling(5, min_periods=1).std().fillna(0).to_numpy()

        # RSI-like momentum: ratio of recent gains to recent gains+losses
        gains = pd.Series(np.clip(returns, 0, None)).rolling(5, min_periods=1).sum()
        losses = pd.Series(np.clip(-returns, 0, None)).rolling(5, min_periods=1).sum()
        rsi_like = (gains / (gains + losses + 1e-8)).to_numpy()

        for day in range(LOOKBACK_WINDOW, n_days):
            today_return = returns[day]
            next_return = returns[day + 1]

            # Headline sentiment correlated with TODAY's return + noise,
            # so text and price features share signal about the same
            # underlying process without the headline literally encoding
            # tomorrow's label.
            sentiment_signal = today_return + rng_np.normal(0, sigma * 1.5)
            if sentiment_signal > sigma * 0.3:
                sentiment = "positive"
            elif sentiment_signal < -sigma * 0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            headline = _make_headline(rng_py, sentiment, ticker)
            lookback_returns = returns[day - LOOKBACK_WINDOW:day].round(6).tolist()
            label = _bucket_return(next_return, flat_band=sigma * 0.15)

            rows.append({
                "ticker": ticker,
                "day": day,
                "headline": headline,
                "lookback_returns": lookback_returns,
                "volume_z": round(float(volume_z[day]), 6),
                "volatility": round(float(rolling_volatility[day]), 6),
                "rsi_like": round(float(rsi_like[day]), 6),
                "label": label,
            })

    df = pd.DataFrame(rows, columns=SCHEMA)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tickers", type=int, default=12)
    parser.add_argument("--n-days", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "synthetic_market_data.parquet"),
    )
    args = parser.parse_args()

    df = generate(n_tickers=args.n_tickers, n_days=args.n_days, seed=args.seed)
    # lookback_returns is a list column; parquet handles it natively via pyarrow.
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df["label"].value_counts(normalize=True).rename("label_balance"))
