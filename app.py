import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Monthly Trading Scheme Demo", layout="wide")

st.title("Monthly Trading Scheme (Public Demo)")
st.caption("Educational demo only — not investment advice. Data from Yahoo Finance via yfinance; may be delayed or incomplete.")

# ----------------------------
# Strategy code (adapted from your notebook)
# ----------------------------

# --- Strategy Parameters (defaults match your notebook cell) ---
ASSET_TICKERS = ["GLD", "SPY", "QQQ", "IEMG", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SHY", "BIL"]

CASH_PROXY_TICKER = "BIL"  # used only if cash filter enabled

START_DATE_DEFAULT = "2007-01-01"


def get_padded_selection(selection_list, num_assets_in_portfolio):
    """Pads a selection list with CASH (or empty) to a fixed size for display consistency."""
    padded_list = list(selection_list)
    while len(padded_list) < num_assets_in_portfolio:
        padded_list.append("CASH")
    return padded_list[:num_assets_in_portfolio]


@st.cache_data(show_spinner=False)
def get_price_data(tickers, start, end):
    """
    Fetch adjusted close prices from yfinance.
    Cached for public demo performance and rate-limit safety.
    """
    adj_close_prices = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not data.empty and "Close" in data:
                adj_close_prices[ticker] = data["Close"]
        except Exception:
            # swallow individual ticker failures
            pass

    # Drop columns entirely missing
    adj_close_prices = adj_close_prices.dropna(axis=1, how="all")

    # Drop rows with all NaN
    adj_close_prices = adj_close_prices.dropna(how="all")

    # Forward-fill gaps (typical with ETFs)
    adj_close_prices = adj_close_prices.ffill()

    # Drop any rows that still have NaN anywhere (forces common start)
    adj_close_prices = adj_close_prices.dropna()

    return adj_close_prices


def calculate_monthly_momentum(daily_prices, lookback_months=8, target_date=None):
    """
    Momentum based on month-end resampled prices:
    momentum = pct_change(lookback_months) on month-end prices.
    If target_date is provided, returns the last momentum values available up to target_date.
    """
    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # If peeking as-of target_date, truncate daily prices
    prices_for_momentum = daily_prices.copy()
    if target_date is not None:
        prices_for_momentum = prices_for_momentum[prices_for_momentum.index <= pd.to_datetime(target_date)]

    monthly_prices = prices_for_momentum.resample("ME").last().dropna(how="all")
    if len(monthly_prices) < lookback_months + 1:
        return pd.DataFrame(), pd.Series(dtype=float)

    momentum = monthly_prices.pct_change(lookback_months)

    if target_date is not None:
        # last available momentum row up to target_date
        relevant_dates = momentum.index[momentum.index <= pd.to_datetime(target_date)]
        if len(relevant_dates) == 0:
            return momentum, pd.Series(dtype=float)
        return momentum, momentum.loc[relevant_dates[-1]].dropna()

    return momentum, pd.Series(dtype=float)


def get_lowest_avg_correlation_triplet(prices_df, tickers_triplets, lookback_days=20, as_of_date=None):
    """
    For each triplet, compute average pairwise correlation of daily returns over lookback_days.
    Pick the triplet with lowest average correlation.
    """
    if prices_df is None or prices_df.empty:
        return None, None

    df = prices_df.copy()
    if as_of_date is not None:
        df = df[df.index <= pd.to_datetime(as_of_date)]

    returns = df.pct_change().dropna()
    if len(returns) < lookback_days + 5:
        return None, None

    returns_window = returns.tail(lookback_days)

    best_triplet = None
    best_score = None

    for trip in tickers_triplets:
        sub = returns_window[list(trip)]
        corr = sub.corr()

        # average of off-diagonal correlations
        vals = []
        for i in range(len(trip)):
            for j in range(i + 1, len(trip)):
                vals.append(corr.iloc[i, j])
        avg_corr = float(np.mean(vals)) if vals else np.nan

        if np.isnan(avg_corr):
            continue

        if best_score is None or avg_corr < best_score:
            best_score = avg_corr
            best_triplet = trip

    return best_triplet, best_score


def run_strategy_selection_logic(
    daily_prices: pd.DataFrame,
    momentum_lookback_months: int,
    correlation_lookback_days: int,
    num_top_candidates: int,
    num_assets_in_final_portfolio: int,
    use_cash_filter: bool,
    cash_proxy_ticker: str,
):
    """
    Core monthly process:
    - compute momentum at month-end
    - select top N momentum candidates
    - from their combinations, choose lowest average-correlation triplet
    Returns:
      final_portfolios_over_time: dict(month_end -> list of tickers)
      momentum_table: DataFrame of momentum values at each month-end
    """
    # monthly momentum over full history
    momentum_df, _ = calculate_monthly_momentum(daily_prices, lookback_months=momentum_lookback_months)

    if momentum_df is None or momentum_df.empty:
        return {}, momentum_df

    final_portfolios_over_time = {}

    for dt in momentum_df.index:
        mom_row = momentum_df.loc[dt].dropna().sort_values(ascending=False)

        if mom_row.empty:
            continue

        # optional cash filter (your notebook default is False)
        if use_cash_filter and cash_proxy_ticker in mom_row.index:
            cash_mom = mom_row.loc[cash_proxy_ticker]
            mom_row = mom_row[mom_row > cash_mom]
            if mom_row.empty:
                final_portfolios_over_time[dt] = ["CASH"]
                continue

        top_candidates = list(mom_row.head(num_top_candidates).index)

        # If fewer candidates than final portfolio size, just take what we have
        if len(top_candidates) <= num_assets_in_final_portfolio:
            final_portfolios_over_time[dt] = top_candidates
            continue

        trips = list(combinations(top_candidates, num_assets_in_final_portfolio))
        best_triplet, _ = get_lowest_avg_correlation_triplet(
            prices_df=daily_prices,
            tickers_triplets=trips,
            lookback_days=correlation_lookback_days,
            as_of_date=dt,
        )

        if best_triplet is None:
            # fallback: top momentum assets
            final_portfolios_over_time[dt] = top_candidates[:num_assets_in_final_portfolio]
        else:
            final_portfolios_over_time[dt] = list(best_triplet)

    return final_portfolios_over_time, momentum_df


def calculate_performance(daily_prices: pd.DataFrame, final_portfolios_over_time: dict):
    """
    Build a daily equity curve for equal-weight portfolio held from each month-end selection
    until next month-end selection.
    Returns:
      perf_df (daily): columns: portfolio_return, equity_curve
      holdings_df (monthly): columns: holdings list at month-end
    """
    if daily_prices is None or daily_prices.empty or not final_portfolios_over_time:
        return pd.DataFrame(), pd.DataFrame()

    # create a month-end series aligned to actual trading days
    month_ends = sorted(final_portfolios_over_time.keys())
    if len(month_ends) < 2:
        return pd.DataFrame(), pd.DataFrame()

    px = daily_prices.copy()
    px = px.loc[px.index >= month_ends[0]].copy()
    rets = px.pct_change().fillna(0.0)

    # Build portfolio returns day-by-day
    port_ret = pd.Series(index=rets.index, dtype=float)

    holdings_records = []

    for i in range(len(month_ends) - 1):
        start_dt = month_ends[i]
        end_dt = month_ends[i + 1]
        holdings = final_portfolios_over_time[start_dt]

        # holding period is (start_dt, end_dt] in daily index terms
        period_mask = (rets.index > start_dt) & (rets.index <= end_dt)
        period_rets = rets.loc[period_mask]

        # equal weight among holdings that exist in returns
        valid = [t for t in holdings if t in period_rets.columns]
        if len(valid) == 0:
            port_ret.loc[period_mask] = 0.0
        else:
            port_ret.loc[period_mask] = period_rets[valid].mean(axis=1)

        holdings_records.append({"month_end": start_dt, "holdings": holdings})

    perf_df = pd.DataFrame({"portfolio_return": port_ret}).dropna()
    perf_df["equity_curve"] = (1 + perf_df["portfolio_return"]).cumprod()

    holdings_df = pd.DataFrame(holdings_records).set_index("month_end")
    return perf_df, holdings_df


def daily_peek_signal(
    peek_date_str: str,
    momentum_lookback: int,
    corr_lookback_days: int,
    num_top_candidates: int,
    num_final_portfolio_assets: int,
    all_daily_prices_df: pd.DataFrame,
    cash_proxy_ticker: str,
    use_cash_filter: bool
):
    """
    "Peek ahead" signal as-of a given date:
    - compute momentum using data up to peek_date
    - pick top candidates
    - pick lowest correlation triplet
    Returns dict for display.
    """
    _, mom_series = calculate_monthly_momentum(
        all_daily_prices_df, lookback_months=momentum_lookback, target_date=peek_date_str
    )
    mom_series = mom_series.dropna().sort_values(ascending=False)

    if mom_series.empty:
        return {"error": "Not enough data to compute momentum for peek date."}

    if use_cash_filter and cash_proxy_ticker in mom_series.index:
        cash_mom = mom_series.loc[cash_proxy_ticker]
        mom_series = mom_series[mom_series > cash_mom]
        if mom_series.empty:
            return {"peek_date": peek_date_str, "signal": ["CASH"], "reason": "Cash filter triggered."}

    top_candidates = list(mom_series.head(num_top_candidates).index)

    if len(top_candidates) <= num_final_portfolio_assets:
        return {
            "peek_date": peek_date_str,
            "top_candidates": top_candidates,
            "suggested_portfolio": top_candidates,
            "note": "Fewer candidates than portfolio size; using all."
        }

    trips = list(combinations(top_candidates, num_final_portfolio_assets))
    best_triplet, best_score = get_lowest_avg_correlation_triplet(
        prices_df=all_daily_prices_df,
        tickers_triplets=trips,
        lookback_days=corr_lookback_days,
        as_of_date=peek_date_str,
    )

    suggested = list(best_triplet) if best_triplet else top_candidates[:num_final_portfolio_assets]
    return {
        "peek_date": peek_date_str,
        "top_candidates": top_candidates,
        "suggested_portfolio": suggested,
        "avg_corr_score": None if best_score is None else float(best_score)
    }


def perf_summary(perf_df: pd.DataFrame):
    if perf_df is None or perf_df.empty:
        return pd.DataFrame()

    equity = perf_df["equity_curve"]
    daily_ret = perf_df["portfolio_return"]

    total_return = equity.iloc[-1] - 1
    days = len(equity)
    years = days / 252.0
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() != 0 else np.nan

    # max drawdown
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return pd.DataFrame(
        {
            "Metric": ["Total Return", "CAGR", "Annual Volatility", "Sharpe (rf=0)", "Max Drawdown"],
            "Value": [total_return, cagr, vol, sharpe, max_dd],
        }
    )


# ----------------------------
# UI controls
# ----------------------------
with st.sidebar:
    st.header("Controls")

    start_date = st.text_input("Start date (YYYY-MM-DD)", value=START_DATE_DEFAULT)
    end_date = st.text_input("End date (YYYY-MM-DD)", value=datetime.today().strftime("%Y-%m-%d"))

    MOMENTUM_LOOKBACK_MONTHS = st.number_input("Momentum lookback (months)", min_value=1, max_value=24, value=8, step=1)
    CORRELATION_LOOKBACK_DAYS = st.number_input("Correlation lookback (days)", min_value=5, max_value=120, value=20, step=1)

    NUM_ASSETS_IN_FINAL_PORTFOLIO = st.number_input("Assets in final portfolio", min_value=1, max_value=6, value=3, step=1)

    NUM_TOP_MOMENTUM_CANDIDATES = st.slider(
        "NUM_TOP_MOMENTUM_CANDIDATES",
        min_value=2,
        max_value=20,
        value=6,
        step=1
    )

    USE_CASH_FILTER = st.checkbox("Use cash filter", value=False)
    run_peek = st.checkbox("Show daily peek-ahead", value=True)

    st.sidebar.write("Asset universe:", ASSET_TICKERS)

    st.divider()
    run_btn = st.button("Run backtest", type="primary")


# ----------------------------
# Run the strategy
# ----------------------------
if run_btn:
    with st.spinner("Downloading price data (cached) and running strategy..."):
        daily_prices = get_price_data(ASSET_TICKERS, start_date, end_date)

    if daily_prices is None or daily_prices.empty:
        st.error("No price data downloaded. Try a different date range or check ticker availability.")
        st.stop()

    st.success(f"Downloaded prices for {daily_prices.shape[1]} tickers from {daily_prices.index.min().date()} to {daily_prices.index.max().date()}")

    # Strategy selection
    final_portfolios_over_time, momentum_df = run_strategy_selection_logic(
        daily_prices=daily_prices,
        momentum_lookback_months=int(MOMENTUM_LOOKBACK_MONTHS),
        correlation_lookback_days=int(CORRELATION_LOOKBACK_DAYS),
        num_top_candidates=int(NUM_TOP_MOMENTUM_CANDIDATES),
        num_assets_in_final_portfolio=int(NUM_ASSETS_IN_FINAL_PORTFOLIO),
        use_cash_filter=bool(USE_CASH_FILTER),
        cash_proxy_ticker=CASH_PROXY_TICKER,
    )

    if not final_portfolios_over_time:
        st.error("Strategy produced no month-end portfolios (likely not enough history).")
        st.stop()

    # Performance
    perf_df, holdings_df = calculate_performance(daily_prices, final_portfolios_over_time)

    # Layout: top metrics + plot
    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        st.subheader("Performance summary")
        summary = perf_summary(perf_df)
        if summary.empty:
            st.write("No performance output.")
        else:
            # format as percentages where appropriate
            def fmt_row(m, v):
                if m in ["Total Return", "CAGR", "Annual Volatility", "Max Drawdown"]:
                    return f"{v:.2%}"
                if m.startswith("Sharpe"):
                    return f"{v:.2f}"
                return str(v)

            summary_display = summary.copy()
            summary_display["Value"] = [fmt_row(m, v) for m, v in zip(summary["Metric"], summary["Value"])]
            st.table(summary_display)

        st.subheader("Latest official month-end holdings")
        last_dt = max(final_portfolios_over_time.keys())
        last_holdings = final_portfolios_over_time[last_dt]
        st.write(f"**{last_dt.date()}** → {get_padded_selection(last_holdings, int(NUM_ASSETS_IN_FINAL_PORTFOLIO))}")

    with colB:
        st.subheader("Equity curve")
        if perf_df is None or perf_df.empty:
            st.write("No equity curve to plot.")
        else:
            fig = plt.figure()
            plt.plot(perf_df.index, perf_df["equity_curve"])
            plt.xlabel("Date")
            plt.ylabel("Equity (start=1.0)")
            st.pyplot(fig, clear_figure=True)

    st.divider()

    # Holdings table
    st.subheader("Holdings over time (month-end decisions)")
    holdings_table = holdings_df.copy()
    holdings_table["holdings"] = holdings_table["holdings"].apply(lambda x: ", ".join(x))
    st.dataframe(holdings_table.tail(60), width='stretch')

    # Momentum table (recent)
    st.subheader("Momentum (recent month-ends)")
    if momentum_df is not None and not momentum_df.empty:
        st.dataframe(momentum_df.tail(24), width='stretch')

    # Daily peek
    if run_peek:
        st.subheader("Daily peek-ahead (as of end date)")
        peek = daily_peek_signal(
            peek_date_str=end_date,
            momentum_lookback=int(MOMENTUM_LOOKBACK_MONTHS),
            corr_lookback_days=int(CORRELATION_LOOKBACK_DAYS),
            num_top_candidates=int(NUM_TOP_MOMENTUM_CANDIDATES),
            num_final_portfolio_assets=int(NUM_ASSETS_IN_FINAL_PORTFOLIO),
            all_daily_prices_df=daily_prices,
            cash_proxy_ticker=CASH_PROXY_TICKER,
            use_cash_filter=bool(USE_CASH_FILTER),
        )
        st.json(peek)

else:
    st.info("Set parameters in the sidebar, then click **Run backtest**.")
