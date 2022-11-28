from itertools import accumulate

import numpy as np
import pandas as pd

from panti.base import ER_


def EMA_init_(ser: pd.Series, period: int = 9) -> pd.Series:
    ser_init = ser.copy()

    first_valid_index = ser.index.get_loc(ser.first_valid_index())
    sma_init = ser.iloc[first_valid_index:first_valid_index + period].mean()

    first_valid_index += period - 1
    ser_init.iloc[:first_valid_index] = np.nan
    ser_init.iloc[first_valid_index] = sma_init
    return ser_init


def SMA(ser: pd.Series, period: int = 41, **kwds) -> pd.Series:
    """
    Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
    The simple moving average (SMA) is the most basic of the moving averages used for trading.
    """
    return ser.rolling(window=period, **kwds).mean()


def EMA(ser: pd.Series, period: int = 9, **kwds) -> pd.Series:
    """
    Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for
    trending markets.
    When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and
    vice-versa for a down trend.
    EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge
    their validity.
    """
    ser_init = EMA_init_(ser, period)
    return ser_init.ewm(span=period, adjust=False, **kwds).mean()


def DEMA(ser: pd.Series, period: int = 9) -> pd.Series:
    """
    Double Exponential Moving Average - attempts to remove the inherent lag associated to Moving Averages
     by placing more weight on recent values. The name suggests this is achieved by applying a double exponential
    smoothing which is not the case. The name double comes from the fact that the value of an EMA
    (Exponential Moving Average) is doubled.
    To keep it in line with the actual data and to remove the lag the value 'EMA of EMA'
    is subtracted from the previously doubled EMA.
    Because EMA(EMA) is used in the calculation, DEMA needs 2 * period -1 samples to start producing values
    in contrast to the period samples needed by a regular EMA
    """

    ema = EMA(ser, period)
    return 2 * ema - EMA(ema, period)


def TEMA(ser: pd.Series, period: int = 9) -> pd.Series:
    """
    Triple exponential moving average - attempts to remove the inherent lag associated to Moving Averages
    by placing more weight on recent values.
    The name suggests this is achieved by applying a triple exponential smoothing which is not the case.
    The name triple comes from the fact that the
    value of an EMA (Exponential Moving Average) is triple.
    To keep it in line with the actual data and to remove the lag the value 'EMA of EMA' is subtracted 3 times
    from the previously tripled EMA.

    Finally, 'EMA of EMA of EMA' is added.
    Because EMA(EMA(EMA)) is used in the calculation, TEMA needs 3 * period - 2 samples to start producing values
    in contrast to the period samples needed by a regular EMA.
    """

    ema = EMA(ser, period)
    ema_ema = EMA(ema, period)
    ema_ema_ema = EMA(ema_ema, period)

    return 3. * ema - 3. * ema_ema + ema_ema_ema


def TRIMA(ser: pd.Series, period: int = 18) -> pd.Series:
    """
    The Triangular Moving Average (TRIMA) [also known as TMA] represents an average of prices,
    but places weight on the middle prices of the time period.
    The calculations double-smooth the data using a window width that is one-half the length of the series.
    source: https://www.thebalance.com/triangular-moving-average-tma-description-and-uses-1031203
    """
    half_period = period // 2
    return SMA(SMA(ser, half_period + 1), period - half_period)


def WMA(ser: pd.Series, period: int = 9) -> pd.Series:
    """
    WMA stands for weighted moving average. It helps to smooth the price curve for better trend identification.
    It places even greater importance on recent data than the EMA does.

    :period: Specifies the number of Periods used for WMA calculation
    """

    d = (period * (period + 1)) / 2  # denominator
    weights = np.arange(1, period + 1)

    close = ser.rolling(period, min_periods=period)
    wma = close.apply(lambda x: (weights * x).sum() / d, raw=True)

    return wma


def RMA(ser: pd.Series, period=9, **kwargs) -> pd.Series:
    """Indicator: WildeR's Moving Average (RMA)
    The WildeR's Moving Average is simply an Exponential Moving Average (EMA) with
    a modified alpha = 1 / length.

    Sources:
        https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
        https://www.incrediblecharts.com/indicators/wilder_moving_average.php

    :param ser: data, a pd.Series
    :param period: range
    :return: result pd.Series
    """
    ser_init = EMA_init_(ser, period)
    return ser_init.ewm(alpha=1.0 / period, adjust=False, **kwargs).mean()


def SMMA(ser: pd.Series, period: int = 42, **kwargs) -> pd.Series:
    """The SMMA (Smoothed Moving Average) gives recent prices an equal weighting to historic prices."""

    return RMA(ser, period, **kwargs)


def KAMA(ser: pd.Series, ema_fast: int = 2, ema_slow: int = 30, period: int = 20) -> pd.Series:
    """Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account
    for market noise or volatility.
    Its main advantage is that it takes into consideration not just the direction, but the market volatility as well."""

    er = ER_(ser, period)
    fast_alpha = 2 / (ema_fast + 1)
    slow_alpha = 2 / (ema_slow + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2  # smoothing constant

    kama = [ser[period - 1]]
    kama = list(
        accumulate(kama + list(zip(sc.iloc[period:], ser.iloc[period:])),
                   func=lambda a, b: a + b[0] * (b[1] - a)))

    return pd.Series([np.nan] * period + kama[1:], index=ser.index)
