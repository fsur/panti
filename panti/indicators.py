import numpy as np
import pandas as pd

from panti.base import ER_
from panti.moving_averages import EMA, SMA, SMMA
from panti.candles import TR


def MACD(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, period_signal: int = 9) -> pd.Series:
    """
    MACD, MACD Signal and MACD difference.
    The MACD Line oscillates above and below the zero line, which is also known as the centerline.
    These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course,
    depends on the direction of the moving average cross.
    Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as
    the shorter EMA diverges further from the longer EMA.
    This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA
    is below the 26-day EMA.
    Negative values increase as the shorter EMA diverges further below the longer EMA.
    This means downside momentum is increasing.

    Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
    As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
    A bullish crossover occurs when the MACD turns up and crosses above the signal line.
    A bearish crossover occurs when the MACD turns down and crosses below the signal line.
    """

    ema_fast = EMA(ser, period=period_fast)
    ema_slow = EMA(ser, period=period_slow)

    return ema_fast - ema_slow


def MACD_signal(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.Series:
    macd = MACD(ser, period_fast, period_slow)
    return EMA(macd, period=signal)


def MACD_hist(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = EMA(ser, period=period_fast)
    ema_slow = EMA(ser, period=period_slow)

    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, period=signal)
    return macd - macd_signal


def MACD_df(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = EMA(ser, period=period_fast)

    ema_slow = EMA(ser, period=period_slow)

    macd = ema_fast - ema_slow
    macd.name = "MACD"

    macd_signal = EMA(macd, period=signal)
    macd_signal.name = "MACD_signal"

    macd_hist = macd - macd_signal
    macd_hist.name = "MACD_hist"
    return pd.concat([macd, macd_signal, macd_hist], axis=1)


def MOM(ser: pd.Series, period: int = 10) -> pd.Series:
    """Market momentum is measured by continually taking price differences for a fixed time interval.
    To construct a 10-day momentum line, simply subtract the closing price 10 days ago from the last closing price.
    This positive or negative value is then plotted around a zero line."""

    return ser.diff(period)


def ROC(ser: pd.Series, period: int = 12) -> pd.Series:
    """The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum,
    is a pure momentum oscillator that measures the percent change in price from one period to the next.
    The ROC calculation compares the current price with the price ???n??? periods ago."""

    return 100. * ser.diff(period) / ser.shift(period)


def RSI(ser: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above
    70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
    RSI can also be used to identify the general trend."""

    # get the price diff
    up = ser.diff()
    down = up.copy()

    # positive gains (up) and negative gains (down) Series
    up[up < 0] = 0
    down[down > 0] = 0

    # EMAs of ups and downs
    _gain = SMMA(up, period=period)
    _loss = SMMA(down.abs(), period=period)

    rs = _gain / _loss
    return 100. - 100. / (1. + rs)


def BBANDS_df(ser: pd.Series, period: int = 20, std_multiplier: float = 2, ma: pd.Series = None) -> pd.DataFrame:
    """
     Developed by John Bollinger, Bollinger Bands?? are volatility bands placed above and below a moving average.
     Volatility is based on the standard deviation, which changes as volatility increases and decreases.
     The bands automatically widen when volatility increases and narrow when volatility decreases.

     This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
     Pass desired moving average as <MA> argument. For example BBANDS(ma=KAMA(20)).
     """

    middle_band = ma

    if middle_band is None:
        middle_band = SMA(ser, period)
    middle_band.name = "BB_middle"

    std = ser.rolling(window=period).std(ddof=0)
    deviations = std_multiplier * std

    upper_bb = middle_band + deviations
    upper_bb.name = "BB_upper"

    lower_bb = middle_band - deviations
    lower_bb.name = "BB_lower"

    return pd.concat([lower_bb, middle_band, upper_bb], axis=1)


def BBANDS_middle(ser: pd.Series, period: int = 20, ma: pd.Series = None) -> pd.Series:
    """
     Developed by John Bollinger, Bollinger Bands?? are volatility bands placed above and below a moving average.
     Volatility is based on the standard deviation, which changes as volatility increases and decreases.
     The bands automatically widen when volatility increases and narrow when volatility decreases.

     This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
     Pass desired moving average as <MA> argument. For example BBANDS(MA=KAMA(20)).
     """

    middle_band = ma
    if middle_band is None:
        middle_band = SMA(ser, period)
    return middle_band


def BBANDS_lower(ser: pd.Series, period: int = 20, std_multiplier: float = 2, ma: pd.Series = None) -> pd.Series:
    """
     Developed by John Bollinger, Bollinger Bands?? are volatility bands placed above and below a moving average.
     Volatility is based on the standard deviation, which changes as volatility increases and decreases.
     The bands automatically widen when volatility increases and narrow when volatility decreases.

     This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
     Pass desired moving average as <MA> argument. For example BBANDS(MA=KAMA(20)).
     """

    middle_band = ma
    if middle_band is None:
        middle_band = SMA(ser, period)

    std = ser.rolling(window=period).std(ddof=0)
    return middle_band - (std_multiplier * std)


def BBANDS_upper(ser: pd.Series, period: int = 20, std_multiplier: float = 2, ma: pd.Series = None) -> pd.Series:
    """
     Developed by John Bollinger, Bollinger Bands?? are volatility bands placed above and below a moving average.
     Volatility is based on the standard deviation, which changes as volatility increases and decreases.
     The bands automatically widen when volatility increases and narrow when volatility decreases.

     This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
     Pass desired moving average as <MA> argument. For example BBANDS(MA=KAMA(20)).
     """

    middle_band = ma
    if middle_band is None:
        middle_band = SMA(ser, period)

    std = ser.rolling(window=period).std(ddof=0)
    return middle_band + (std_multiplier * std)


def UO(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different time frames.
    The multiple time frame objective seeks to avoid the pitfalls of other oscillators.
    Many momentum oscillators surge at the beginning of a strong advance and then form bearish divergence as the
    advance continues.
    This is because they are stuck with one time frame. The Ultimate Oscillator attempts to correct this fault by
    incorporating longer
    time frames into the basic formula."""

    close_shifted = close.shift(1)
    k = np.where(low < close_shifted, low, close_shifted)  # current low or past close
    bp = ser - k  # Buying pressure

    average7 = bp.rolling(window=7).sum() / TR(high, low, close).rolling(window=7).sum()
    average14 = bp.rolling(window=14).sum() / TR(high, low, close).rolling(window=14).sum()
    average28 = bp.rolling(window=28).sum() / TR(high, low, close).rolling(window=28).sum()

    return (100 * ((4 * average7) + (2 * average14) + average28)) / (4 + 2 + 1)


def ER(ser: pd.Series, period: int = 10) -> pd.Series:
    """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100,
    where zero is the center point.
     +100 is upward forex trending market and -100 is downwards trending markets."""

    return ER_(ser, period)


def TRIX(ser: pd.Series, period: int = 20) -> pd.Series:
    """
    The TRIX indicator calculates the rate of change of a triple exponential moving average.
    The values oscillate around zero. Buy/sell signals are generated when the TRIX crosses above/below zero.
    A (typically) 9 period exponential moving average of the TRIX can be used as a signal line.
    A buy/sell signals are generated when the TRIX crosses above/below the signal line and is also above/below zero.

    The TRIX was developed by Jack K. Hutson, publisher of Technical Analysis of Stocks & Commodities magazine,
    and was introduced in Volume 1, Number 5 of that magazine.
    """

    m = EMA(EMA(EMA(ser, period), period), period)
    return 100 * m.diff() / m.shift()
