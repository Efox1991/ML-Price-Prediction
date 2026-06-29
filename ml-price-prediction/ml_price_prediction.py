# =============================================================================
# price_prediction.py
#
# Binary classification model for next-day return direction prediction.
# Uses RandomForestClassifier vs Logistic Regression (L2) as a linear baseline.
# Evaluated via walk-forward validation with purge and embargo to prevent
# data leakage. Macro features (SPY, VIX) provide market context.
#
# Target: 1 if next-day log return > 0, else 0.
# Features: lagged returns, ATR, RSI, OBV diff, SPY return, VIX level/change,
#           200-day MA regime flag, volatility regime flag, beta-residual return.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Asset and data settings ---
TICKER        = "GOOGL"     # Primary asset to model
SPY_TICKER    = "SPY"       # S&P 500 ETF used as market proxy for macro feature
VIX_TICKER    = "^VIX"      # CBOE Volatility Index: market-implied 30-day vol
PERIOD        = "5y"        # History window passed to yfinance
INTERVAL      = "1d"        # Daily bars

# --- Feature construction ---
N_LAGS        = 5           # Number of lagged log return features
ATR_WINDOW    = 14          # Rolling window for Average True Range (standard: 14)
RSI_WINDOW    = 14          # Rolling window for RSI (standard: 14)
OBV_DIFF_LAG  = 1           # Period for OBV first difference
MA_WINDOW     = 200         # Moving average window for regime detection
VOL_REGIME_W  = 20          # Window for rolling volatility regime flag
BETA_WINDOW   = 60          # Rolling window for estimating GOOGL beta vs SPY

# --- Walk-forward validation ---
N_FOLDS           = 5       # Number of walk-forward folds
MIN_TRAIN_ROWS    = 200     # Minimum rows required in the first training window
PURGE_ROWS        = 14      # Rows dropped from the END of each training set.
                            # Equals the longest rolling feature window, so that
                            # no training feature overlaps with test features.
EMBARGO_ROWS      = 5       # Additional rows dropped from the START of each test
                            # set. Prevents the model exploiting autocorrelation
                            # in features on the split boundary.

# --- Models ---
RF_N_TREES    = 500         # Number of trees in the random forest
RF_MAX_DEPTH  = 5           # Max depth per tree; shallow trees reduce overfitting
                            # on low-signal financial data
LR_C          = 1.0         # Inverse regularisation strength for Logistic
                            # Regression. Smaller C = more regularisation.
                            # Equivalent to 1/alpha in Ridge regression.

# --- Reproducibility ---
RANDOM_SEED   = 42


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_raw_data(df: pd.DataFrame, label: str) -> None:
    """
    Confirm the downloaded DataFrame has the expected OHLCV columns and
    sufficient history to support feature construction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from yfinance.
    label : str
        Ticker label used in error messages.

    Raises
    ------
    ValueError
        If required columns are missing or row count is below threshold.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    if len(df) < MIN_TRAIN_ROWS + PURGE_ROWS + EMBARGO_ROWS + 50:
        raise ValueError(
            f"{label}: only {len(df)} rows downloaded. Increase PERIOD."
        )


def validate_vix_data(df: pd.DataFrame) -> None:
    """
    VIX data from yfinance only has a Close column reliably.
    Confirm it is present.

    Parameters
    ----------
    df : pd.DataFrame
        Raw VIX data.

    Raises
    ------
    ValueError
        If Close column is absent.
    """
    if "Close" not in df.columns:
        raise ValueError("VIX data missing Close column.")


def validate_features(X: pd.DataFrame, label: str = "") -> None:
    """
    Confirm no NaN values remain in the feature matrix after construction
    and cleaning.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    label : str
        Optional label for the error message.

    Raises
    ------
    ValueError
        If NaNs are present.
    """
    if X.isnull().any().any():
        raise ValueError(
            f"Feature matrix{' (' + label + ')' if label else ''} contains "
            "NaN values after cleaning."
        )


def validate_fold_sizes(train_size: int, test_size: int, fold: int) -> None:
    """
    Confirm both train and test sets have enough rows to fit and evaluate
    a model.

    Parameters
    ----------
    train_size : int
        Number of training rows after purge.
    test_size : int
        Number of test rows after embargo.
    fold : int
        Fold index, used in the error message.

    Raises
    ------
    ValueError
        If either set is too small.
    """
    if train_size < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Fold {fold}: training set has only {train_size} rows after purge. "
            f"Increase MIN_TRAIN_ROWS or reduce N_FOLDS."
        )
    if test_size < 10:
        raise ValueError(
            f"Fold {fold}: test set has only {test_size} rows after embargo. "
            "Reduce N_FOLDS or increase PERIOD."
        )


# =============================================================================
# DATA ACQUISITION
# =============================================================================

def download_asset(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker via yfinance.

    auto_adjust=True applies the split and dividend adjustment to all price
    columns, ensuring historical prices are comparable across corporate actions.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    period : str
        Lookback period (e.g. "5y").
    interval : str
        Bar size (e.g. "1d").

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by date.
    """
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.dropna(inplace=True)

    # yfinance occasionally returns a MultiIndex column structure when
    # downloading a single ticker. Flatten to a simple column index.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def download_all_data(
    ticker: str, spy_ticker: str, vix_ticker: str,
    period: str, interval: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download price data for the primary asset, SPY, and VIX, then align
    all three to a common date index.

    SPY is used to compute market returns and estimate the asset's beta.
    VIX is the CBOE Volatility Index: it represents the market's 30-day
    implied volatility derived from S&P 500 option prices. It is the
    standard measure of market fear or risk.

    Parameters
    ----------
    ticker : str
        Primary asset ticker.
    spy_ticker : str
        S&P 500 ETF ticker.
    vix_ticker : str
        VIX ticker (^VIX on Yahoo Finance).
    period : str
        Lookback period.
    interval : str
        Bar size.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (asset_df, spy_df, vix_df) aligned to their common date index.
    """
    print(f"Downloading {ticker}...")
    asset_df = download_asset(ticker, period, interval)
    validate_raw_data(asset_df, ticker)

    print(f"Downloading {spy_ticker}...")
    spy_df = download_asset(spy_ticker, period, interval)
    validate_raw_data(spy_df, spy_ticker)

    # VIX has no Volume and sometimes no Open/High/Low from yfinance.
    # Downloaded as a full OHLCV call only using the Close column.
    print(f"Downloading {vix_ticker}...")
    vix_df = download_asset(vix_ticker, period, interval)
    validate_vix_data(vix_df)

    # Align all three DataFrames to dates present in all three series.
    # Resolves the issues of potential NaNs from missing trading days in any single series.
    common_index = asset_df.index.intersection(spy_df.index).intersection(vix_df.index)
    asset_df = asset_df.loc[common_index]
    spy_df   = spy_df.loc[common_index]
    vix_df   = vix_df.loc[common_index]

    print(
        f"\n{len(asset_df)} aligned rows: "
        f"{asset_df.index[0].date()} to {asset_df.index[-1].date()}.\n"
    )
    return asset_df, spy_df, vix_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# JUSTIFICATION OF LOG RETURNS OVER RAW PRICES
#
# Raw prices are non-stationary: the mean and variance drift over time, which
# violates assumptions made by most ML models. A model trained on 2019 price
# levels has no basis to generalise to 2024 price levels.
#
# Log returns r_t = ln(P_t / P_{t-1}) are approximately stationary and additive
# across time. They are the standard unit of analysis in quantitative finance.
#
# JUSTIFICATION OF BINARY CLASSIFICATION OVER REGRESSION
#
# Predicting the exact magnitude of tomorrow's log return is difficuly:
# financial returns are close to white noise, so regression models produce
# predictions near zero with large errors. The question to ask 
# for a trading strategy is which direction will the asset move day to day.
#  
# Reformulating as binary classification.
# Makes the target cleaner and the evaluation metrics more interpretable.

def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Compute daily log returns from a closing price series.

    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    close : pd.Series
        Closing price series.

    Returns
    -------
    pd.Series
        Log return series. First value is NaN by construction.
    """
    return np.log(close / close.shift(1))


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute Average True Range (ATR): the standard measure of intraday
    volatility used in technical analysis.

    True Range on each bar is the maximum of:
      - High - Low (intraday range)
      - |High - Previous Close| (gap up scenario)
      - |Low  - Previous Close| (gap down scenario)

    ATR is the rolling mean of True Range. It captures how much the asset
    typically moves per day. It is used in quant work for position sizing
    (volatility targeting) and regime detection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with High, Low, Close columns.
    window : int
        Rolling window length.

    Returns
    -------
    pd.Series
        ATR series.
    """
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI): a bounded momentum oscillator
    in [0, 100].

    High RSI (>70) is interpreted as overbought; low RSI (<30)
    as oversold. It is one of the most widely used signals in
    trading strategies.

    Formula:
        RS  = EMA(gains, window) / EMA(losses, window)
        RSI = 100 - 100 / (1 + RS)

    Parameters
    ----------
    close : pd.Series
        Closing price series.
    window : int
        Lookback window.

    Returns
    -------
    pd.Series
        RSI series, values in [0, 100].
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=window, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, min_periods=window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Compute On-Balance Volume (OBV): cumulative volume indicator.

    On each bar, the full volume is added to a running total if the close
    is up, subtracted if the close is down. Volume potentially
    precedes price: rising OBV suggests accumulation; falling OBV suggests
    distribution.

    We use the first difference of OBV as a feature rather than the raw
    cumulative total, because the cumulative total is non-stationary.

    Parameters
    ----------
    close : pd.Series
    volume : pd.Series

    Returns
    -------
    pd.Series
        Cumulative OBV series (to be differenced in build_features).
    """
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def compute_beta(asset_returns: pd.Series, market_returns: pd.Series,
                 window: int) -> pd.Series:
    """
    Compute rolling beta of the asset relative to the market.

    Beta measures the asset's sensitivity to market moves:
        beta_t = Cov(r_asset, r_market) / Var(r_market)

    over a rolling window of length `window`. A beta of 1.5 means the asset
    historically moves 1.5% for every 1% market move. Beta is central to
    factor models (CAPM, Fama-French) and to computing the market-adjusted
    (idiosyncratic) component of returns.

    Parameters
    ----------
    asset_returns : pd.Series
        Log returns of the primary asset.
    market_returns : pd.Series
        Log returns of the market proxy (SPY).
    window : int
        Rolling window for covariance and variance estimation.

    Returns
    -------
    pd.Series
        Rolling beta series.
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def build_features(
    asset_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    n_lags: int,
    atr_window: int,
    rsi_window: int,
    obv_diff_lag: int,
    ma_window: int,
    vol_regime_w: int,
    beta_window: int
) -> pd.DataFrame:
    """
    Construct the full feature matrix and binary classification target
    from aligned OHLCV data for the asset, SPY, and VIX.

    Features
    --------
    Lagged log returns (return_lag_1 ... return_lag_N):
        Short-term autocorrelation / momentum. Lag 1 = yesterday's
        return, lag N = N days ago.

    ATR normalised (atr_normalised):
        ATR divided by Close price to give a dimensionless volatility measure
        comparable across price levels and time.

    RSI (rsi):
        Momentum oscillator. Captures overbought/oversold conditions.

    OBV first difference (obv_diff):
        Change in on-balance volume per bar. Captures volume flow direction.

    SPY log return (spy_return):
        Market return on the same day. Captures whether the asset is moving
        with or against the broad market. This is the most important macro
        feature for a stock like GOOGL.

    VIX level (vix_level):
        Current level of the CBOE Volatility Index. High VIX indicates
        elevated market fear and is associated with larger
        price moves. Useful as a regime indicator.

    VIX daily change (vix_change):
        Day-over-day change in VIX. A spike in VIX (sudden fear increase)
        often precedes a sharp equity sell-off.

    200-day MA regime flag (ma_regime):
        Binary: 1 if Close > 200-day moving average, 0 otherwise.
        A price above its 200-day MA is the standard definition of a
        long-term uptrend. Many systematic strategies condition signal
        direction on this flag because mean-reversion and momentum signals
        behave differently in trending vs contracting regimes.

    Volatility regime flag (vol_regime):
        Binary: 1 if current ATR is above its own rolling median, 0 otherwise.
        High-volatility regimes often coincide with trending markets where
        momentum signals work better; low-volatility regimes are more
        mean-reverting.

    Beta-adjusted residual return (beta_residual):
        r_asset - beta * r_market. This is the idiosyncratic (stock-specific)
        return after removing the market component. It is a purer signal of
        stock-specific momentum or mean-reversion, uncontaminated by broad
        market moves. Used in statistical arbitrage and factor-neutral
        long/short strategies.

    Target
    ------
    Binary label: 1 if next-day log return > 0, else 0.
    The target is shifted back by one day so that the label on row t is
    the return realised on day t+1, which is what the model must predict
    using features observed up to and including day t.

    Parameters
    ----------
    asset_df : pd.DataFrame
        OHLCV data for the primary asset.
    spy_df : pd.DataFrame
        OHLCV data for SPY.
    vix_df : pd.DataFrame
        Data for VIX (Close column used).
    n_lags : int
        Number of lagged return features.
    atr_window : int
        ATR rolling window.
    rsi_window : int
        RSI rolling window.
    obv_diff_lag : int
        Period for OBV differencing.
    ma_window : int
        Moving average window for regime flag.
    vol_regime_w : int
        Window for volatility regime rolling median.
    beta_window : int
        Window for rolling beta estimation.

    Returns
    -------
    pd.DataFrame
        Feature matrix with binary "target" column.
    """
    features = pd.DataFrame(index=asset_df.index)

    # --- Log returns for asset and market ---
    asset_ret  = compute_log_returns(asset_df["Close"])
    market_ret = compute_log_returns(spy_df["Close"])

    # --- Lagged asset log returns ---
    for lag in range(1, n_lags + 1):
        features[f"return_lag_{lag}"] = asset_ret.shift(lag)

    # --- ATR normalised ---
    atr = compute_atr(asset_df, window=atr_window)
    features["atr_normalised"] = atr / asset_df["Close"]

    # --- RSI ---
    features["rsi"] = compute_rsi(asset_df["Close"], window=rsi_window)

    # --- OBV first difference ---
    obv = compute_obv(asset_df["Close"], asset_df["Volume"])
    features["obv_diff"] = obv.diff(obv_diff_lag)

    # --- SPY return (macro: market direction) ---
    # Using the lagged SPY return (shift 1) so that today's model input
    # uses yesterday's SPY return, which is known at market open.
    features["spy_return"] = market_ret.shift(1)

    # --- VIX level and daily change ---
    vix_close = vix_df["Close"]
    features["vix_level"]  = vix_close.shift(1)         # Yesterday's VIX level
    features["vix_change"] = vix_close.diff(1).shift(1) # Yesterday's VIX change

    # --- 200-day MA regime flag ---
    ma200 = asset_df["Close"].rolling(window=ma_window).mean()
    features["ma_regime"] = (asset_df["Close"] > ma200).astype(int)

    # --- Volatility regime flag ---
    atr_median = atr.rolling(window=vol_regime_w).median()
    features["vol_regime"] = (atr > atr_median).astype(int)

    # --- Beta-adjusted residual return ---
    rolling_beta = compute_beta(asset_ret, market_ret, window=beta_window)
    # Residual = asset return - beta * market return, lagged by 1.
    residual = asset_ret - rolling_beta * market_ret
    features["beta_residual"] = residual.shift(1)

    # --- Binary target ---
    # 1 if next-day return is positive, 0 if negative or zero.
    # Shifted back by 1 so the label on row t is the outcome on day t+1.
    features["target"] = (asset_ret.shift(-1) > 0).astype(int)

    # Drop rows with NaN from rolling windows, lags, and target shift.
    features.dropna(inplace=True)

    return features


# =============================================================================
# PURGE AND EMBARGO
# =============================================================================

# DEFINITION OF PURGE AND EMBARGO
#
# Standard k-fold cross-validation assumes observations are independent.
# Financial time series are not: consecutive rows share rolling window
# computations (ATR, RSI, beta all look back 14-60 days). A training
# row immediately before the test boundary and the first test row are
# computed from overlapping raw data. Training on one and testing on the
# other inflates performance.
#
# PURGE: remove the last PURGE_ROWS rows from the training set at each
# fold boundary. This ensures no training feature uses raw data that also
# contributed to test features.
#
# EMBARGO: remove the first EMBARGO_ROWS rows from the test set at each
# fold boundary. This guards against autocorrelation in the target
# (if tomorrow's return correlates with today's, the model could
# exploit this without learning anything useful).
#
# This is the standard "purged cross-validation" approach
# described in Marcos Lopez de Prado's "Advances in Financial Machine
# Learning" (2018).

def apply_purge_embargo(
    features: pd.DataFrame,
    train_end_idx: int,
    test_start_idx: int,
    purge_rows: int,
    embargo_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply purge and embargo to a single fold's train and test slices.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix.
    train_end_idx : int
        Integer iloc index of the last training row (before purge).
    test_start_idx : int
        Integer iloc index of the first test row (before embargo).
    purge_rows : int
        Rows to remove from the end of the training set.
    embargo_rows : int
        Rows to remove from the start of the test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test) after purge and embargo.
    """
    # Purge: move the effective training end back by purge_rows.
    purged_train_end = train_end_idx - purge_rows
    train = features.iloc[:purged_train_end]

    # Embargo: move the effective test start forward by embargo_rows.
    embargoed_test_start = test_start_idx + embargo_rows
    test = features.iloc[embargoed_test_start:]

    return train, test


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

# DEFINITION OF WALK-FORWARD:
#
# The full feature set is divided into N_FOLDS + 1 segments. In each fold:
#   - The training set is all data up to the current fold boundary (expanding).
#   - The test set is the next segment.
#   - Purge and embargo are applied at the boundary.
#   - Both models are fitted on the training set and predictions collected
#     on the test set.
#
# After all folds predictions are concatenated into a single out-of-sample
# series covering the full test period. This gives a distribution of
# performance across time rather than a single number from one split.
#
# EXPANDING OVER ROLLING:
# We use an expanding window: each fold adds more training data. This is
# standard for equities where more history generally improves estimation.
# A rolling window (fixed-length training set) would be preferable if
# the data-generating process is believed to be non-stationary over long
# horizons (e.g. structural regime changes). For GOOGL over 5 years
# we choose expanding.

def run_walk_forward(
    features: pd.DataFrame,
    n_folds: int,
    min_train_rows: int,
    purge_rows: int,
    embargo_rows: int,
    rf_n_trees: int,
    rf_max_depth: int,
    lr_c: float,
    seed: int
) -> tuple[pd.DataFrame, list[RandomForestClassifier]]:
    """
    Run expanding-window walk-forward validation with purge and embargo.

    At each fold:
      1. Slice train and test sets at the fold boundary.
      2. Apply purge to the end of the training set.
      3. Apply embargo to the start of the test set.
      4. Fit StandardScaler on training features only.
      5. Fit RandomForestClassifier and LogisticRegression on scaled training data.
      6. Predict class probabilities and labels on scaled test data.
      7. Store predictions, actuals, and dates.

    Fold boundaries are spaced evenly across the data after reserving
    MIN_TRAIN_ROWS for the first training set.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix with "target" column.
    n_folds : int
        Number of walk-forward folds.
    min_train_rows : int
        Minimum training rows before the first fold.
    purge_rows : int
        Rows to purge from each training end.
    embargo_rows : int
        Rows to embargo from each test start.
    rf_n_trees : int
        Random forest tree count.
    rf_max_depth : int
        Random forest max depth.
    lr_c : float
        Logistic regression regularisation (inverse strength).
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, list[RandomForestClassifier]]
        - results_df: concatenated out-of-sample predictions across all folds,
          with columns: actual, rf_pred, lr_pred, rf_prob, lr_prob, fold.
        - rf_models: list of fitted RandomForestClassifier objects, one per fold.
          The last entry corresponds to the most recent training window and is
          used for feature importance reporting.
    """
    feature_cols = [c for c in features.columns if c != "target"]
    n_rows       = len(features)

    # Compute fold boundaries as evenly spaced integer indices.
    # The first boundary is at min_train_rows; the last is at n_rows.
    # Each fold's test set is the segment between consecutive boundaries.
    
    usable_rows    = n_rows - min_train_rows
    fold_size      = usable_rows // (n_folds + 1)
    fold_boundaries = [
        min_train_rows + i * fold_size for i in range(n_folds + 1)
    ]

    print(fold_boundaries)

    all_results = []
    rf_models   = []
    
    for fold in range(n_folds):
        train_end   = fold_boundaries[fold + 1]   # End of training slice
        test_start  = fold_boundaries[fold + 1]   # Start of test slice
        test_end    = fold_boundaries[fold + 2] if fold + 2 <= n_folds else n_rows # End of test slice

        # Guard: if this is the last fold, test_end is already n_rows.
        # Clamp to valid range.
        test_end = min(test_end, n_rows)
        
        # Apply purge and embargo at the fold boundary.
        train, test = apply_purge_embargo(
            features,
            train_end_idx=train_end,
            test_start_idx=test_start,
            purge_rows=purge_rows,
            embargo_rows=embargo_rows
        )

        validate_fold_sizes(len(train), len(test), fold + 1)

        X_train = train[feature_cols]
        y_train = train["target"].values
        X_test  = test[feature_cols]
        y_test  = test["target"].values

        validate_features(X_train, label=f"fold {fold+1} train")
        validate_features(X_test,  label=f"fold {fold+1} test")

        # Fit scaler on training data only.
        # Fitting on the full dataset would leak test distribution information
        # into the scaling parameters (a form of data leakage).
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # --- Random Forest ---
        # n_jobs=-1 uses all CPU cores for parallel tree building.
        rf = RandomForestClassifier(
            n_estimators=rf_n_trees,
            max_depth=rf_max_depth,
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        # predict_proba returns [P(class=0), P(class=1)] per row.
        # We keep P(class=1) as the model's confidence in an "up" prediction.
        rf_prob = rf.predict_proba(X_test_s)[:, 1]

        # --- Logistic Regression (L2 baseline) ---
        # LogisticRegression with penalty="l2" is the classification equivalent
        # of Ridge regression. It applies an L2 penalty to the coefficient vector,
        # shrinking coefficients toward zero and reducing overfitting when features
        # are collinear. C is the inverse of regularisation strength (1/alpha).
        # max_iter=1000 avoids convergence warnings on larger datasets.
        lr = LogisticRegression(C=lr_c, penalty="l2", max_iter=1000,
                                random_state=seed)
        lr.fit(X_train_s, y_train)
        lr_pred = lr.predict(X_test_s)
        lr_prob = lr.predict_proba(X_test_s)[:, 1]

        # Collect results for this fold into a DataFrame aligned to test dates.
        fold_df = pd.DataFrame(
            {
                "actual":  y_test,
                "rf_pred": rf_pred,
                "lr_pred": lr_pred,
                "rf_prob": rf_prob,
                "lr_prob": lr_prob,
                "fold":    fold + 1,
            },
            index=test.index
        )
        all_results.append(fold_df)
        rf_models.append(rf)

        print(
            f"Fold {fold+1}/{n_folds} | "
            f"Train: {len(train)} rows | "
            f"Test: {len(test)} rows | "
            f"RF acc: {accuracy_score(y_test, rf_pred):.3f} | "
            f"LR acc: {accuracy_score(y_test, lr_pred):.3f}"
        )

    results_df = pd.concat(all_results)
    return results_df, rf_models


# =============================================================================
# EVALUATION METRICS
# =============================================================================

# DEFINITION OF METRICS FOR BINARY CLASSIFICATION:
#
# Accuracy: fraction of correct predictions. Baseline is max(class frequency).
#   If 55% of days are "up", a model that always predicts "up" gets 55% accuracy.
#   Accuracy above this naive baseline indicates some value in predictions.
#
# Precision: percentage of correctly predicted days as "up".
#   High precision = fewer false positives. Relevant for a long-only strategy
#   where false positives are costly (buying on days the stock falls).
#
# Recall: percentage of days that actually went "up".
#   High recall = fewer false negatives. Relevant when missing an up day is costly.
#
# F1: harmonic mean of precision and recall.
#   Standard summary metric for binary classification.
#
# Directional accuracy = accuracy as the classification target is
# direction.

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics for a single fold or the full out-of-sample
    period.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    dict
        Dictionary of metric name to value.
    """
    return {
        "Accuracy":   accuracy_score(y_true, y_pred),
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "F1":         f1_score(y_true, y_pred, zero_division=0),
    }


# =============================================================================
# OUTPUT
# =============================================================================

def print_per_fold_table(results_df: pd.DataFrame) -> None:
    """
    Print a per-fold metrics table for both models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Concatenated walk-forward results with columns: actual, rf_pred,
        lr_pred, fold.
    """
    rows = []
    for fold_num in sorted(results_df["fold"].unique()):
        fold_data = results_df[results_df["fold"] == fold_num]
        y_true    = fold_data["actual"].values
        rf_m      = compute_fold_metrics(y_true, fold_data["rf_pred"].values)
        lr_m      = compute_fold_metrics(y_true, fold_data["lr_pred"].values)

        rows.append({
            "Fold":          fold_num,
            "N Test":        len(fold_data),
            "RF Accuracy":   rf_m["Accuracy"],
            "RF Precision":  rf_m["Precision"],
            "RF Recall":     rf_m["Recall"],
            "RF F1":         rf_m["F1"],
            "LR Accuracy":   lr_m["Accuracy"],
            "LR Precision":  lr_m["Precision"],
            "LR Recall":     lr_m["Recall"],
            "LR F1":         lr_m["F1"],
        })

    df_table = pd.DataFrame(rows).set_index("Fold")

    print("=" * 90)
    print("PER-FOLD METRICS")
    print("=" * 90)
    print(df_table.round(3).to_string())
    print()


def print_aggregate_table(results_df: pd.DataFrame) -> None:
    """
    Print aggregate metrics across the full out-of-sample period for both models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Concatenated walk-forward results.
    """
    y_true = results_df["actual"].values
    rf_m   = compute_fold_metrics(y_true, results_df["rf_pred"].values)
    lr_m   = compute_fold_metrics(y_true, results_df["lr_pred"].values)

    # Naive baseline: always predict the majority class.
    majority_class  = int(results_df["actual"].mean() >= 0.5)
    naive_pred      = np.full(len(y_true), majority_class)
    naive_acc       = accuracy_score(y_true, naive_pred)

    df_agg = pd.DataFrame(
        {"Random Forest": rf_m, "Logistic Regression": lr_m}
    )
    df_agg.index.name = "Metric"

    print("=" * 55)
    print("AGGREGATE METRICS (FULL OUT-OF-SAMPLE PERIOD)")
    print("=" * 55)
    print(df_agg.round(4).to_string())
    print(f"\nNaive majority-class baseline accuracy: {naive_acc:.4f}")
    print("=" * 55)
    print()


def print_feature_importances(
    rf_model: RandomForestClassifier, feature_names: list
) -> None:
    """
    Print ranked feature importances from a fitted RandomForestClassifier.

    Feature importance is measured by mean decrease in Gini impurity across
    all splits in all trees. Higher values indicate features the model relied
    on more heavily when partitioning the data.

    In context it tells you which signals the model is using.
    A well-specified model should weight meaningful features
    (macro regime, volatility state) over noise (arbitrary lags).

    Parameters
    ----------
    rf_model : RandomForestClassifier
        Fitted model from the final walk-forward fold.
    feature_names : list
        Feature column names in the same order as the training matrix.
    """
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("RANDOM FOREST FEATURE IMPORTANCES")
    print("(mean decrease in Gini impurity, final fold model)")
    print("-" * 50)
    print(importances.round(4).to_string())
    print()


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(
    results_df: pd.DataFrame,
    asset_df: pd.DataFrame
) -> None:
    # Compute actual log returns over the full asset history.
    actual_returns = np.log(
        asset_df["Close"] / asset_df["Close"].shift(1)
    )

    # Define the test period as the full span from the first prediction
    # date to the last, including any purge/embargo gaps between folds.
    test_start = results_df.index[0]
    test_end   = results_df.index[-1]

    # Slice buy-and-hold to the test period using the full asset index.
    # This gives a continuous daily return series with no gaps.
    bh_returns_test = actual_returns.loc[test_start:test_end]
    signal_index    = bh_returns_test.index

    # Build fold_start_dates without groupby().apply() to avoid the pandas
    # duplicate-index issue introduced in newer versions. Instead, take the
    # first index value per fold directly from the grouped object.
    fold_start_dates = {
        fold_num: group.index[0]
        for fold_num, group in results_df.groupby("fold")
    }

    # Reindex predictions onto the continuous signal index.
    # results_df may have a non-unique index if any dates fall in both a
    # fold's test window and an adjacent fold's embargo window. Drop
    # duplicates keeping the first occurrence before reindexing.
    rf_pred_series = results_df["rf_pred"][~results_df.index.duplicated(keep="first")]
    lr_pred_series = results_df["lr_pred"][~results_df.index.duplicated(keep="first")]

    # Days where no prediction exists (purge/embargo gaps) are filled with
    # zero: the strategy is flat on those days.
    rf_daily = actual_returns.reindex(signal_index) * \
               rf_pred_series.reindex(signal_index).fillna(0)
    lr_daily = actual_returns.reindex(signal_index) * \
               lr_pred_series.reindex(signal_index).fillna(0)

    bh_cumulative = bh_returns_test.cumsum()
    rf_cumulative = rf_daily.cumsum()
    lr_cumulative = lr_daily.cumsum()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Panel 1: per-fold accuracy ---
    folds   = sorted(results_df["fold"].unique())
    rf_accs = []
    lr_accs = []
    for fold_num in folds:
        fold_data = results_df[results_df["fold"] == fold_num]
        rf_accs.append(accuracy_score(fold_data["actual"], fold_data["rf_pred"]))
        lr_accs.append(accuracy_score(fold_data["actual"], fold_data["lr_pred"]))

    x     = np.arange(len(folds))
    width = 0.35
    axes[0].bar(x - width / 2, rf_accs, width, label="Random Forest",
                color="steelblue")
    axes[0].bar(x + width / 2, lr_accs, width, label="Logistic Regression",
                color="darkorange")
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--",
                    label="50% baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {f}" for f in folds])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-Fold Directional Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0.3, 0.75)

    # --- Panel 2: continuous cumulative signal return vs buy-and-hold ---
    axes[1].plot(signal_index, bh_cumulative,
                 label="Buy and hold", color="black", linewidth=1.2)
    axes[1].plot(signal_index, rf_cumulative,
                 label="RF signal return", color="steelblue",
                 linewidth=1.0, alpha=0.8)
    axes[1].plot(signal_index, lr_cumulative,
                 label="LR signal return", color="darkorange",
                 linewidth=1.0, alpha=0.8)

    for fold_num, date in fold_start_dates.items():
        if fold_num == 1:
            continue
        axes[1].axvline(date, color="grey", linewidth=0.7,
                        linestyle=":", alpha=0.7)
        axes[1].text(date, bh_cumulative.min(), f" F{fold_num}",
                     fontsize=7, color="grey", va="bottom")

    axes[1].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Cumulative Log Return")
    axes[1].set_title("Cumulative Signal Return vs Buy and Hold (Full Test Period)")
    axes[1].legend()

    # --- Panel 3: RF probability distribution ---
    axes[2].hist(results_df["rf_prob"], bins=30, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[2].axvline(0.5, color="black", linewidth=0.8, linestyle="--",
                    label="Decision boundary")
    axes[2].set_xlabel("Predicted P(up)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(
        "Distribution of RF Predicted Probabilities (All Test Folds)"
    )
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    End-to-end pipeline:
      1. Download asset, SPY, and VIX data.
      2. Build feature matrix with macro and technical features.
      3. Run walk-forward validation with purge and embargo.
      4. Print per-fold and aggregate metrics.
      5. Print feature importances from the final fold model.
      6. Plot results.
    """

    # --- Data ---
    asset_df, spy_df, vix_df = download_all_data(
        TICKER, SPY_TICKER, VIX_TICKER, PERIOD, INTERVAL
    )

    # --- Features ---
    print("Building features...")
    features = build_features(
        asset_df, spy_df, vix_df,
        n_lags=N_LAGS,
        atr_window=ATR_WINDOW,
        rsi_window=RSI_WINDOW,
        obv_diff_lag=OBV_DIFF_LAG,
        ma_window=MA_WINDOW,
        vol_regime_w=VOL_REGIME_W,
        beta_window=BETA_WINDOW
    )
    print(f"Feature matrix: {features.shape[0]} rows, "
          f"{features.shape[1] - 1} features.\n")

    # --- Walk-forward validation ---
    print("Running walk-forward validation...\n")
    results_df, rf_models = run_walk_forward(
        features,
        n_folds=N_FOLDS,
        min_train_rows=MIN_TRAIN_ROWS,
        purge_rows=PURGE_ROWS,
        embargo_rows=EMBARGO_ROWS,
        rf_n_trees=RF_N_TREES,
        rf_max_depth=RF_MAX_DEPTH,
        lr_c=LR_C,
        seed=RANDOM_SEED
    )

    print()

    # --- Output ---
    print_per_fold_table(results_df)
    print_aggregate_table(results_df)

    # Feature importances from the final fold model, which was trained on
    # the most data and is the most representative of the full training history.
    feature_cols = [c for c in features.columns if c != "target"]
    print_feature_importances(rf_models[-1], feature_cols)

    # --- Plot ---
    plot_results(results_df, asset_df)

if __name__ == "__main__":
    main()