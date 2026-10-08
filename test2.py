import ccxt
import pandas as pd
import numpy as np
import time


# ============================================================
# POT CRYPTO FUTURES SCANNER
# Binance USDT-Margined Perpetuals
# ============================================================

# -----------------------------
# SETTINGS
# -----------------------------

MIN_24H_VOLUME = 100_000_000       # $100M minimum 24h futures volume
MIN_OPEN_INTEREST = 20_000_000     # $20M minimum OI
MIN_VOL_OI = 1.5                   # Volume/OI minimum
MAX_VOL_OI = 5.0                   # Volume/OI maximum
MIN_ATR_PCT = 2.5                  # Minimum 14-day ATR %
MAX_SYMBOLS_STAGE_2 = 150          # Limit expensive ATR requests
REQUEST_DELAY = 0.05               # Small delay between requests


# -----------------------------
# BINANCE CONNECTION
# -----------------------------

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})


# ============================================================
# ATR CALCULATION
# ============================================================

def calculate_atr_pct(symbol):

    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe='1d',
            limit=15
        )

        if len(ohlcv) < 15:
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=[
                'timestamp',
                'open',
                'high',
                'low',
                'close',
                'volume'
            ]
        )

        # Previous candle close
        previous_close = df['close'].shift(1)

        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - previous_close)
        tr3 = abs(df['low'] - previous_close)

        df['TR'] = np.maximum(
            tr1,
            np.maximum(tr2, tr3)
        )

        # 14-day average true range
        atr = df['TR'].rolling(14).mean().iloc[-1]

        current_close = df['close'].iloc[-1]

        if current_close <= 0:
            return None

        atr_pct = (atr / current_close) * 100

        return atr_pct

    except Exception:
        return None


# ============================================================
# MAIN SCANNER
# ============================================================

def run_custom_futures_scanner():

    print("\n" + "=" * 70)
    print("POT CRYPTO FUTURES SCANNER")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MARKETS
    # --------------------------------------------------------

    print("\nLoading Binance futures markets...")

    markets = exchange.load_markets()

    symbols = []

    for symbol, market in markets.items():

        try:

            # USDT linear perpetual contracts only
            if (
                market.get('linear') is True
                and market.get('quote') == 'USDT'
                and market.get('swap') is True
                and market.get('active') is True
            ):
                symbols.append(symbol)

        except Exception:
            continue

    print(f"USDT perpetual markets found: {len(symbols)}")

    # --------------------------------------------------------
    # STAGE 1
    # VOLUME + OI + VOLUME/OI
    # --------------------------------------------------------

    print("\nStage 1: Checking volume, OI and Volume/OI...")

    stage1 = []

    for i, symbol in enumerate(symbols, start=1):

        try:

            ticker = exchange.fetch_ticker(symbol)

            last_price = ticker.get('last')
            quote_volume = ticker.get('quoteVolume')

            if not last_price or not quote_volume:
                continue

            # 24h volume filter
            if quote_volume < MIN_24H_VOLUME:
                continue

            # Open interest
            oi_data = exchange.fetch_open_interest(symbol)

            oi_amount = oi_data.get('openInterestAmount')

            if not oi_amount:
                continue

            # Convert OI from contracts to USD
            open_interest_usd = oi_amount * last_price

            # OI filter
            if open_interest_usd < MIN_OPEN_INTEREST:
                continue

            # Volume / OI
            vol_oi_ratio = quote_volume / open_interest_usd

            # Volume/OI filter
            if not (
                MIN_VOL_OI
                <= vol_oi_ratio
                <= MAX_VOL_OI
            ):
                continue

            stage1.append({
                'Symbol': symbol,
                'Price': last_price,
                'Volume': quote_volume,
                'OI': open_interest_usd,
                'Vol_OI': vol_oi_ratio
            })

        except Exception:
            continue

        # Progress
        if i % 25 == 0:
            print(
                f"Processed {i}/{len(symbols)} "
                f"| Candidates: {len(stage1)}"
            )

        time.sleep(REQUEST_DELAY)

    print(
        f"\nStage 1 complete: "
        f"{len(stage1)} candidates"
    )

    if not stage1:
        print("No coins passed Stage 1.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # SORT STAGE 1
    # --------------------------------------------------------

    stage1_df = pd.DataFrame(stage1)

    # Prioritize meaningful OI
    stage1_df = stage1_df.sort_values(
        by='OI',
        ascending=False
    )

    # Limit expensive ATR calculations
    stage1_df = stage1_df.head(
        MAX_SYMBOLS_STAGE_2
    )

    # --------------------------------------------------------
    # STAGE 2
    # ATR
    # --------------------------------------------------------

    print(
        f"\nStage 2: Calculating 14-day ATR "
        f"for {len(stage1_df)} candidates..."
    )

    final_results = []

    for i, row in enumerate(
        stage1_df.iterrows(),
        start=1
    ):

        _, data = row

        symbol = data['Symbol']

        atr_pct = calculate_atr_pct(symbol)

        if atr_pct is None:
            continue

        # ATR filter
        if atr_pct < MIN_ATR_PCT:
            continue

        final_results.append({
            'Symbol': symbol,
            'Price': data['Price'],
            '24h Vol ($M)': data['Volume'] / 1_000_000,
            'OI ($M)': data['OI'] / 1_000_000,
            'Vol/OI': data['Vol_OI'],
            'ATR %': atr_pct
        })

        if i % 10 == 0:
            print(
                f"ATR processed: "
                f"{i}/{len(stage1_df)}"
            )

        time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # FINAL DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(final_results)

    if df.empty:
        print("\nNo coins passed all filters.")
        return df

    # ========================================================
    # POT RANKING
    # ========================================================

    # Normalize metrics between 0 and 1
    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                [1.0] * len(series),
                index=series.index
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    df['OI Score'] = normalize(df['OI ($M)'])

    df['Volume Score'] = normalize(
        np.log1p(df['24h Vol ($M)'])
    )

    df['ATR Score'] = normalize(
        df['ATR %']
    )

    # Volume/OI is best around the middle of the range,
    # rather than simply "the higher the better".
    ideal_vol_oi = 3.0

    df['Vol/OI Score'] = (
        1 -
        abs(df['Vol/OI'] - ideal_vol_oi)
        / max(
            ideal_vol_oi - MIN_VOL_OI,
            MAX_VOL_OI - ideal_vol_oi
        )
    ).clip(lower=0)

    # --------------------------------------------------------
    # POT SCORE
    # --------------------------------------------------------

    df['POT Score'] = (
        df['OI Score'] * 0.30 +
        df['Volume Score'] * 0.25 +
        df['Vol/OI Score'] * 0.20 +
        df['ATR Score'] * 0.25
    ) * 100

    # --------------------------------------------------------
    # ROUND VALUES
    # --------------------------------------------------------

    df['Price'] = df['Price'].round(8)
    df['24h Vol ($M)'] = df['24h Vol ($M)'].round(1)
    df['OI ($M)'] = df['OI ($M)'].round(1)
    df['Vol/OI'] = df['Vol/OI'].round(2)
    df['ATR %'] = df['ATR %'].round(2)
    df['POT Score'] = df['POT Score'].round(1)

    # --------------------------------------------------------
    # FINAL SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by='POT Score',
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("POT FUTURES SHORTLIST")
    print("=" * 70)

    print(
        df[
            [
                'Symbol',
                'Price',
                '24h Vol ($M)',
                'OI ($M)',
                'Vol/OI',
                'ATR %',
                'POT Score'
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)

    print(
        f"Final candidates: {len(df)}"
    )

    print(
        f"Filters: "
        f"Volume >= ${MIN_24H_VOLUME/1e6:.0f}M | "
        f"OI >= ${MIN_OPEN_INTEREST/1e6:.0f}M | "
        f"Vol/OI {MIN_VOL_OI}-{MAX_VOL_OI} | "
        f"ATR >= {MIN_ATR_PCT}%"
    )

    return df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    results = run_custom_futures_scanner()