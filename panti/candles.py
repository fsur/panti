import numpy as np
import pandas as pd

from panti.moving_averages import SMA, EMA, RMA


def TR(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range is the maximum of three price ranges.
    Most recent period's high minus the most recent period's low.
    Absolute value of the most recent period's high minus the previous close.
    Absolute value of the most recent period's low minus the previous close."""

    tr1 = (high - low).abs()  # True Range = High less Low

    prev_close = close.shift()
    tr2 = (high - prev_close).abs()  # True Range = High less Previous Close

    tr3 = (prev_close - low).abs()  # True Range = Previous Close less Low

    tr = pd.concat([tr1, tr2, tr3], axis=1)

    return tr.max(axis=1)


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, avg_method="sma") -> pd.Series:
    """Average True Range is moving average of True Range.
    The ATR can be calculated differently depending on its definition.
    (see https://www.macroption.com/atr-calculation/)
    """
    tr = TR(high, low, close)

    if avg_method.lower() == "sma":
        return SMA(tr, period=period)
    elif avg_method.lower() == "ema":
        return EMA(tr, period=period)
    else:  # Wilder's moving average
        return RMA(tr, period=period)


def WILLIAMS(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
     of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
     Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
     of its recent trading range.
     The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = high.rolling(center=False, window=period).max()
    lowest_low = low.rolling(center=False, window=period).min()

    wr = (highest_high - close) / (highest_high - lowest_low)

    return -100. * wr


def MFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """The money flow index (MFI) is a momentum indicator that measures
    the inflow and outflow of money into a security over a specific period of time.
    MFI can be understood as RSI adjusted for volume.
    The money flow indicator is one of the more reliable indicators of overbought and oversold conditions, perhaps partly because
    it uses the higher readings of 80 and 20 as compared to the RSI's overbought/oversold readings of 70 and 30"""

    tp = TP(high, low, close)
    rmf = tp * volume  ## Real Money Flow

    tp_diff = tp.diff()

    delta_pos = pd.Series(np.where(tp_diff > 0, rmf, 0), index=high.index)
    delta_neg = pd.Series(np.where(tp_diff < 0, rmf, 0), index=high.index)

    mfratio = delta_pos.rolling(window=period).sum() / delta_neg.rolling(window=period).sum()

    return 100 - (100 / (1 + mfratio))


def TP(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Typical Price refers to the arithmetic average of the high, low, and closing prices for a given period."""

    return (high + low + close) / 3
