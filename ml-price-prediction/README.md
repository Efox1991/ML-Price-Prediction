# Price Prediction

Binary classification model for next-day return direction prediction using
walk-forward validation with purge and embargo.


## Overview

This script downloads OHLCV data for a configurable ticker via yfinance,
alongside SPY (market proxy) and VIX (volatility index) for macro context.
It engineers a feature set of lagged log returns, standard technical
indicators, and macro regime features, then trains two models: a
RandomForestClassifier and a Logistic Regression baseline. Both are evaluated
using expanding-window walk-forward validation with purge and embargo applied
at each fold boundary to prevent data leakage.


## Features

- Binary classification target: 1 if next-day log return is positive, else 0
- Technical features: lagged returns, ATR, RSI, OBV first difference
- Macro features: SPY return, VIX level, VIX daily change
- Regime features: 200-day MA flag, volatility regime flag
- Beta-adjusted residual return isolating the stock-specific component
- Expanding-window walk-forward validation across configurable fold count
- Purge and embargo at each fold boundary following Lopez de Prado (2018)
- Per-fold and aggregate metrics: accuracy, precision, recall, F1
- Naive majority-class baseline for comparison
- Feature importances from the final fold Random Forest
- Three-panel plot: per-fold accuracy, cumulative signal return vs
  buy-and-hold, RF predicted probability distribution


## Installation

Clone the repository and install the required dependencies:

```bash

git clone https://github.com/Efox1991/Markowitz-Portfolio-Optimization-Model.git

cd Markowitz-Portfolio-Optimization-Model

pip install -r requirements.txt

```


## Usage

Edit the configuration block at the top of `price_prediction.py`:

```python
TICKER        = "GOOGL"
SPY_TICKER    = "SPY"
VIX_TICKER    = "^VIX"
PERIOD        = "5y"
INTERVAL      = "1d"
N_FOLDS       = 5
MIN_TRAIN_ROWS = 200
PURGE_ROWS    = 14
EMBARGO_ROWS  = 5
RF_N_TREES    = 500
RF_MAX_DEPTH  = 5
LR_C          = 1.0
RANDOM_SEED   = 42
```

Then run:

```bash
python price_prediction.py
```


## Output

- Fold-by-fold progress with training size, test size, and accuracy per model
- Per-fold metrics table: accuracy, precision, recall, F1 for both models
- Aggregate metrics table across the full out-of-sample period
- Naive majority-class baseline accuracy for reference
- Random Forest feature importances ranked by mean decrease in Gini impurity
- Three-panel plot: per-fold accuracy bars, cumulative signal vs buy-and-hold,
  RF predicted probability distribution


## Limitations

- Daily equity returns are close to unpredictable. Accuracy meaningfully above
  the majority-class baseline over many folds would be notable. Results near
  50-55% are expected and do not indicate a broken implementation.
- No transaction costs, slippage, or position sizing are modelled. Signal
  return in the plot is not a realistic backtest.
- Features use only price, volume, and two macro series. Order book data,
  earnings calendars, sentiment, and alternative data are excluded.
- The model is retrained at each fold boundary but not updated within a fold.
  A production system would retrain more frequently.
- Purge and embargo reduce the effective training size at each boundary.
  With short PERIOD settings or many folds this can produce very small
  training sets, which the validate_fold_sizes function will flag.

## Interpretation of Results

**Analysis of the Results:**

-Accuracy oscillates around 50% across all five folds with no consistent 
 direction. Folds 2 and 4 are marginally above 50%, folds 1, 3, and 5 are below it. 
-This pattern is indistinguishable from noise. The signal return lines sit close to 
 zero for most of the test period and only pick up slightly in the final fold.
-Not enough to claim a signal.
-Buy and hold compounds roughly 140% over the same period which would be a better 
 strategy.

**Why this is expected:**

-GOOGL is a large-cap, heavily traded, and extensively analysed stock. Price and 
 volume data is available to every participant in the market. Any pattern that 
 exists in lagged returns or standard technical indicators gets arbitraged away quickly 
 because thousands of quantitative strategies are already looking for these patterns. 
-This is the efficient market hypothesis operating in practice.
-The probability distribution in panel 3 confirms this: the histogram is roughly 
 bell-shaped and centred near 0.5, meaning the model has no confident predictions 
 in either direction. 
-A model with a signal would show more entries at 0 and 1 in the histogram in panel 3.

## Reference

Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
The purge and embargo methodology follows Chapter 7.


## Dependencies

- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

## License

MIT
