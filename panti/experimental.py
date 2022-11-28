from itertools import accumulate

import pandas as pd
import numpy as np

from panti.indicators import MACD, ROC, RSI, BBANDS_df, BBANDS_middle, BBANDS_lower, BBANDS_upper
from panti.candles import ATR, TP, TR
from panti.moving_averages import SMA, EMA, DEMA, WMA, RMA


# --------------------------   MOVING AVERAGES   ------------------------------------------


def VAMA(ser: pd.Series, weights: pd.Series, period: int = 8) -> pd.Series:
    """
    Volume Adjusted Moving Average
    """

    vp = ser * weights
    volsum = weights.rolling(window=period).mean()
    vol_ratio = vp / volsum
    cum_sum = (vol_ratio * ser).rolling(window=period).sum()
    cum_div = vol_ratio.rolling(window=period).sum()

    return cum_sum / cum_div


def ZLEMA(ser: pd.Series, period: int = 26) -> pd.Series:
    """ZLEMA is an abbreviation of Zero Lag Exponential Moving Average. It was developed by John Ehlers and Rick Way.
    ZLEMA is a kind of Exponential moving average but its main idea is to eliminate the lag arising from
    the very nature of the moving averages and other trend following indicators.
    As it follows price closer, it also provides better price averaging and responds better to price swings."""

    lag = (period - 1) // 2

    ema = ser + (ser.diff(periods=lag))

    return EMA(ema, period)


def HMA(ser: pd.Series, period: int = 16) -> pd.Series:
    """
    HMA indicator is a common abbreviation of Hull Moving Average.
    The average was developed by Allan Hull and is used mainly to identify the current market trend.
    Unlike SMA (simple moving average) the curve of Hull moving average is considerably smoother.
    Moreover, because its aim is to minimize the lag between HMA and price it does follow
    the price activity much closer.
    It is used especially for middle-term and long-term trading.
    :period: Specifies the number of Periods used for WMA calculation
    """

    import math

    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wmaf = WMA(ser, period=half_length)
    wmas = WMA(ser, period=period)
    deltawma = 2 * wmaf - wmas
    return WMA(deltawma, period=sqrt_length)


def EVWMA(ser: pd.Series, weights: pd.Series, period: int = 20) -> pd.Series:
    """
    The eVWMA can be looked at as an approximation to the
    average price paid per share in the last n periods.

    :period: Specifies the number of Periods used for eVWMA calculation
    """

    vol_sum = weights.rolling(window=period).sum()  # floating shares in last N periods

    x = (vol_sum - weights) / vol_sum
    y = (weights * ser) / vol_sum

    evwma = [0]
    evwma.extend(list(zip(x.fillna(0.).values, y.fillna(0.).values)))

    #  evwma = (evma[-1] * (vol_sum - volume)/vol_sum) + (volume * price / vol_sum)
    evwma = list(accumulate(evwma, func=lambda a, b: a * b[0] + b[1]))
    evwma = pd.Series(evwma[1:], index=ser.index)
    evwma.loc[vol_sum.isna()] = np.nan
    return evwma


def FRAMA(ser: pd.Series, period: int = 16, batch: int = 10) -> pd.Series:
    """Fractal Adaptive Moving Average
    Adopted from: https://www.quantopian.com/posts/frama-fractal-adaptive-moving-average-in-python

    :period: Specifies the number of periods used for FRANA calculation
    :batch: Specifies the size of batches used for FRAMA calculation
    """

    assert period % 2 == 0, print("FRAMA period must be even")

    c = ser.copy()
    window = batch * 2

    hh = c.rolling(batch).max()
    ll = c.rolling(batch).min()

    n1 = (hh - ll) / batch
    n2 = n1.shift(batch)

    hh2 = c.rolling(window).max()
    ll2 = c.rolling(window).min()
    n3 = (hh2 - ll2) / window

    # calculate fractal dimension
    d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    alp = np.exp(-4.6 * (d - 1))
    alp = np.clip(alp, .01, 1)

    filt = c.values
    for i, x in enumerate(alp):
        cl = c.values[i]
        if i < window:
            continue
        filt[i] = cl * x + (1 - x) * filt[i - 1]

    return filt


# --------------------------   INDICATORS   ------------------------------------------


def PPO(ser: pd.Series, period_fast: int = 12, period_slow: int = 26) -> pd.Series:
    """
    Percentage Price Oscillator
    PPO, PPO Signal and PPO difference.
    As with MACD, the PPO reflects the convergence and divergence of two moving averages.
    While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing
    the difference by the slower moving average
    """

    ema_fast = EMA(ser, period=period_fast)
    ema_slow = EMA(ser, period=period_slow)

    return ((ema_fast - ema_slow) / ema_slow) * 100


def PPO_signal(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Percentage Price Oscillator
    PPO, PPO Signal and PPO difference.
    As with MACD, the PPO reflects the convergence and divergence of two moving averages.
    While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing
    the difference by the slower moving average
    """

    ppo = PPO(ser, period_fast, period_slow)

    return EMA(ppo, period=signal)


def PPO_histo(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Percentage Price Oscillator
    PPO, PPO Signal and PPO difference.
    As with MACD, the PPO reflects the convergence and divergence of two moving averages.
    While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing
    the difference by the slower moving average
    """

    ppo = PPO(ser, period_fast, period_slow)

    ppo_signal = PPO_signal(ser, period_fast, period_slow, signal)

    return ppo - ppo_signal


def PPO_df(ser: pd.Series, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Percentage Price Oscillator
    PPO, PPO Signal and PPO difference.
    As with MACD, the PPO reflects the convergence and divergence of two moving averages.
    While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing
    the difference by the slower moving average
    """

    ema_fast = EMA(ser, period=period_fast)

    ema_slow = EMA(ser, period=period_slow)

    ppo = ((ema_fast - ema_slow) / ema_slow) * 100
    ppo.name = "PPO"

    ppo_signal = EMA(ppo, period=signal)
    ppo_signal.name = "PPO_signal"

    ppo_histo = ppo - ppo_signal
    ppo_histo.name = "PPO_histo"

    return pd.concat([ppo, ppo_signal, ppo_histo], axis=1)


def VW_MACD_df(ser: pd.Series, weights: pd.Series, period_fast: int = 12, period_slow: int = 26,
               signal: int = 9) -> pd.DataFrame:
    """"Volume-Weighted MACD" is an indicator that shows how a volume-weighted moving average can be used to calculate
    moving average convergence/divergence (MACD).
    This technique was first used by Buff Dormeier, CMT, and has been written about since at least 2002."""

    macd = VW_MACD(ser, weights, period_fast, period_slow)
    macd.name = "MACD"

    macd_signal = EMA(macd, period=signal)
    macd_signal.name = "MACD_signal"

    return pd.concat([macd, macd_signal], axis=1)


def VW_MACD(ser: pd.Series, weights: pd.Series, period_fast: int = 12, period_slow: int = 26) -> pd.Series:
    """"Volume-Weighted MACD" is an indicator that shows how a volume-weighted moving average can be used to calculate
    moving average convergence/divergence (MACD).
    This technique was first used by Buff Dormeier, CMT, and has been written about since at least 2002."""

    vp = weights * ser

    vp_fast = EMA(vp, period=period_fast)
    weights_fast = EMA(weights, period=period_fast)

    vp_slow = EMA(vp, period=period_slow)
    weights_slow = EMA(weights, period=period_slow)

    return vp_fast / weights_fast - vp_slow / weights_slow


def VW_MACD_signal(ser: pd.Series, weights: pd.Series, period_fast: int = 12, period_slow: int = 26,
                   signal: int = 9) -> pd.Series:
    """"Volume-Weighted MACD" is an indicator that shows how a volume-weighted moving average can be used to calculate
    moving average convergence/divergence (MACD).
    This technique was first used by Buff Dormeier, CMT, and has been written about since at least 2002."""

    macd = VW_MACD(ser, weights, period_fast, period_slow)

    return EMA(macd, period=signal)


def EV_MACD_df(ser: pd.Series, weights: pd.Series, period_fast: int = 20, period_slow: int = 40,
               signal: int = 9) -> pd.DataFrame:
    """
    Elastic Volume Weighted MACD is a variation of standard MACD,
    calculated using two EVWMA's.

    :period_slow: Specifies the number of Periods used for the slow EVWMA calculation
    :period_fast: Specifies the number of Periods used for the fast EVWMA calculation
    :signal: Specifies the number of Periods used for the signal calculation
    """

    macd = EV_MACD(ser, weights, period_fast, period_slow)
    macd.name = "MACD"
    macd_signal = EMA(macd, period=signal)
    macd_signal.name = "MACD_signal"

    return pd.concat([macd, macd_signal], axis=1)


def EV_MACD(ser: pd.Series, weights: pd.Series, period_fast: int = 20, period_slow: int = 40) -> pd.Series:
    """
    Elastic Volume Weighted MACD is a variation of standard MACD,
    calculated using two EVWMA's.

    :period_slow: Specifies the number of Periods used for the slow EVWMA calculation
    :period_fast: Specifies the number of Periods used for the fast EVWMA calculation
    :signal: Specifies the number of Periods used for the signal calculation
    """

    evwma_slow = EVWMA(ser, weights, period=period_slow)

    evwma_fast = EVWMA(ser, weights, period=period_fast)

    return evwma_fast - evwma_slow


def EV_MACD_signal(ser: pd.Series, weights: pd.Series, period_fast: int = 20, period_slow: int = 40,
                   signal: int = 9) -> pd.Series:
    """
    Elastic Volume Weighted MACD is a variation of standard MACD,
    calculated using two EVWMA's.

    :period_slow: Specifies the number of Periods used for the slow EVWMA calculation
    :period_fast: Specifies the number of Periods used for the fast EVWMA calculation
    :signal: Specifies the number of Periods used for the signal calculation
    """

    macd = EV_MACD(ser, weights, period_fast, period_slow)
    return EMA(macd, period=signal)


def VBM(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, roc_period: int = 12,
        atr_period: int = 26) -> pd.Series:
    """The Volatility-Based-Momentum (VBM) indicator, The calculation for a volatility based momentum (VBM)
    indicator is very similar to ROC, but divides by the security’s historical volatility instead.
    The average true range indicator (ATR) is used to compute historical volatility.
    VBM(n,v) = (Close — Close n periods ago) / ATR(v periods)
    """

    return (ser.diff(roc_period) - ser.shift(roc_period)) / ATR(high, low, close, atr_period)


def IFT_RSI(ser: pd.Series, rsi_period: int = 5, wma_period: int = 9) -> pd.Series:
    """Modified Inverse Fisher Transform applied on RSI.
    Suggested method to use any IFT indicator is to buy when the indicator crosses over –0.5 or crosses over +0.5
    if it has not previously crossed over –0.5 and to sell short when the indicators crosses under +0.5
    or crossesunder –0.5
    if it has not previously crossed under +0.5."""

    # v1 = .1 * (rsi - 50)
    v1 = 0.1 * (RSI(ser, rsi_period) - 50)

    # v2 = WMA(wma_period) of v1
    v2 = WMA(v1, wma_period)

    return (v2 ** 2 - 1) / (v2 ** 2 + 1)


def STOCHRSI(ser: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """StochRSI is an oscillator that measures the level of RSI relative to its high-low range over a set time period.
    StochRSI applies the Stochastics formula to RSI values, instead of price values. This makes it an indicator of
    an indicator.
    The result is an oscillator that fluctuates between 0 and 1."""

    rsi = RSI(ser, rsi_period)
    return ((rsi - rsi.min()) / (rsi.max() - rsi.min())).rolling(window=stoch_period).mean()


def DYMI(ser: pd.Series, period=14) -> pd.Series:
    """
    The Dynamic Momentum Index is a variable term RSI. The RSI term varies from 3 to 30. The variable
    time period makes the RSI more responsive to short-term moves. The more volatile the price is,
    the shorter the time period is. It is interpreted in the same way as the RSI, but provides signals earlier.
    Readings below 30 are considered oversold, and levels over 70 are considered overbought. The indicator
    oscillates between 0 and 100.
    https://www.investopedia.com/terms/d/dynamicmomentumindex.asp
    """

    # Value available from nth period
    sd = ser.rolling(5).std(ddof=0)
    asd = sd.rolling(10).mean()
    v = sd / asd
    t = (period / v).fillna(0).astype(int)
    t = t.map(lambda x: int(min(max(x, 5), 30)))

    dymi = map(lambda t_, i_: RSI(ser.iloc[max(0, i_ - t_):i_ + 1],
                                  period=period).values[-1],
               t,
               range(len(ser)))
    return pd.Series(list(dymi), index=ser.index)


def MOBO_df(ser: pd.Series) -> pd.DataFrame:
    """
    "MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back"
    If the price breaks out of the MOBO band it can signify a trend move or price spike
    Contains 42% of price movements(noise) within bands.
    """

    return BBANDS_df(ser, period=10, std_multiplier=0.8).rename(columns={"BB_middle": "MOBO_middle",
                                                                         "BB_upper": "MOBO_upper",
                                                                         "BB_lower": "MOBO_lower"})


def MOBO_middle(ser: pd.Series) -> pd.Series:
    """
    "MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back"
    If the price breaks out of the MOBO band it can signify a trend move or price spike
    Contains 42% of price movements(noise) within bands.
    """

    return BBANDS_middle(ser, period=10)


def MOBO_lower(ser: pd.Series) -> pd.Series:
    """
    "MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back"
    If the price breaks out of the MOBO band it can signify a trend move or price spike
    Contains 42% of price movements(noise) within bands.
    """

    return BBANDS_lower(ser, period=10, std_multiplier=0.8)


def MOBO_upper(ser: pd.Series) -> pd.Series:
    """
    "MOBO bands are based on a zone of 0.80 standard deviation with a 10 period look-back"
    If the price breaks out of the MOBO band it can signify a trend move or price spike
    Contains 42% of price movements(noise) within bands.
    """

    return BBANDS_upper(ser, period=10, std_multiplier=0.8)


def BBWIDTH(ser: pd.Series, period: int = 20, std_multiplier: float = 2, ma: pd.Series = None) -> pd.Series:
    """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""

    bb = BBANDS_df(ser, period, std_multiplier, ma)
    return (bb["BB_upper"] - bb["BB_lower"]) / bb["BB_middle"]


def PERCENT_B(ser: pd.Series, period: int = 20, std_multiplier: float = 2, ma: pd.Series = None) -> pd.Series:
    """
    %b (pronounced 'percent b') is derived from the formula for Stochastics and shows where price is in relation to
    the bands.
    %b equals 1 at the upper band and 0 at the lower band.
    """

    bb = BBANDS_df(ser, period, std_multiplier, ma)

    return (ser - bb["BB_lower"]) / (bb["BB_upper"] - bb["BB_lower"])


def KC_df(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, atr_period: int = 10,
          kc_mult: float = 2, ma: pd.Series = None) -> pd.DataFrame:
    """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
    This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
    Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
    The channels are typically set two Average True Range values above and below the 20-day EMA.
    The exponential moving average dictates direction and the Average True Range sets channel width.
    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel
    direction.
    Channels can also be used to identify overbought and oversold levels when the trend is flat."""

    middle = ma
    if middle is None:
        middle = EMA(ser, period)
    middle.name = "KC_middle"

    upper = middle + (kc_mult * ATR(high, low, close, atr_period))
    upper.name = "KC_upper"

    lower = middle - (kc_mult * ATR(high, low, close, atr_period))
    lower.name = "KC_down"

    return pd.concat([lower, middle, upper], axis=1)


def KC_middle(ser: pd.Series, period: int = 20, ma: pd.Series = None) -> pd.Series:
    """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
    This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
    Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
    The channels are typically set two Average True Range values above and below the 20-day EMA.
    The exponential moving average dictates direction and the Average True Range sets channel width.
    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel
    direction.
    Channels can also be used to identify overbought and oversold levels when the trend is flat."""

    middle = ma
    if middle is None:
        middle = EMA(ser, period)
    return middle


def KC_upper(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20,
             atr_period: int = 10, kc_mult: float = 2, ma: pd.Series = None) -> pd.Series:
    """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
    This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
    Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
    The channels are typically set two Average True Range values above and below the 20-day EMA.
    The exponential moving average dictates direction and the Average True Range sets channel width.
    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel
    direction.
    Channels can also be used to identify overbought and oversold levels when the trend is flat."""

    middle = ma
    if middle is None:
        middle = EMA(ser, period)

    return middle + (kc_mult * ATR(high, low, close, atr_period))


def KC_lower(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20,
             atr_period: int = 10, kc_mult: float = 2, ma: pd.Series = None) -> pd.DataFrame:
    """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
    This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
    Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
    The channels are typically set two Average True Range values above and below the 20-day EMA.
    The exponential moving average dictates direction and the Average True Range sets channel width.
    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel
    direction.
    Channels can also be used to identify overbought and oversold levels when the trend is flat."""

    middle = ma
    if middle is None:
        middle = EMA(ser, period)

    return middle - (kc_mult * ATR(high, low, close, atr_period))


def KST_df(ser: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30) -> pd.DataFrame:
    """Know Sure Thing (KST) is a momentum oscillator based on the smoothed rate-of-change for four different
    time frames.
    KST measures price momentum for four different price cycles. It can be used just like any momentum oscillator.
    Chartists can look for divergences, overbought/oversold readings, signal line crossovers and centerline crossovers.
    """

    k = KST_k(ser, r1, r2, r3, r4)
    k.name = "KST"
    signal = k.rolling(window=10).mean()
    signal.name = "signal"

    return pd.concat([k, signal], axis=1)


def KST_k(ser: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30) -> pd.DataFrame:
    """Know Sure Thing (KST) is a momentum oscillator based on the smoothed rate-of-change for four different
    time frames.
    KST measures price momentum for four different price cycles. It can be used just like any momentum oscillator.
    Chartists can look for divergences, overbought/oversold readings, signal line crossovers and centerline crossovers.
    """

    r1 = ROC(ser, r1).rolling(window=10).mean()
    r2 = ROC(ser, r2).rolling(window=10).mean()
    r3 = ROC(ser, r3).rolling(window=10).mean()
    r4 = ROC(ser, r4).rolling(window=15).mean()

    return (r1 * 1) + (r2 * 2) + (r3 * 3) + (r4 * 4)


def KST_signal(ser: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30) -> pd.DataFrame:
    """Know Sure Thing (KST) is a momentum oscillator based on the smoothed rate-of-change for four different
    time frames.
    KST measures price momentum for four different price cycles. It can be used just like any momentum oscillator.
    Chartists can look for divergences, overbought/oversold readings, signal line crossovers and centerline crossovers.
    """

    k = KST_k(ser, r1, r2, r3, r4)
    return k.rolling(window=10).mean()


def TSI_df(ser: pd.Series, long: int = 25, short: int = 13, signal: int = 13) -> pd.DataFrame:
    """True Strength Index (TSI) is a momentum oscillator based on a double smoothing of price changes."""

    tsi = TSI_indicator(ser, long, short)
    tsi.name = "tsi"
    tsi_signal = EMA(tsi, period=signal)
    tsi_signal.name = "TSI_signal"

    return pd.concat([tsi, tsi_signal], axis=1)


def TSI_indicator(ser: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index (TSI) is a momentum oscillator based on a double smoothing of price changes."""

    # Double smoother price change
    momentum = ser.diff()  # 1 period momentum
    _EMA25 = EMA(momentum, period=long)
    _DEMA13 = DEMA(_EMA25, period=short)

    # Double smoothed absolute price change
    absmomentum = ser.diff().abs()
    _aEMA25 = EMA(absmomentum, period=long)
    _aDEMA13 = EMA(_aEMA25, period=short)

    return 100. * _DEMA13 / _aDEMA13


def TSI_signal(ser: pd.Series, long: int = 25, short: int = 13, signal: int = 13) -> pd.Series:
    """True Strength Index (TSI) is a momentum oscillator based on a double smoothing of price changes."""

    tsi = TSI_indicator(ser, long, short)
    return EMA(tsi, period=signal)


def OBV(ser: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV) measures buying and selling pressure as a cumulative indicator that adds volume on up days
    and subtracts volume on down days.
    OBV was developed by Joe Granville and introduced in his 1963 book, Granville's New Key to Stock Market Profits.
    It was one of the first indicators to measure positive and negative volume flow.
    Chartists can look for divergences between OBV and price to predict price movements or use OBV to confirm
    price trends.

    source: https://en.wikipedia.org/wiki/On-balance_volume#The_formula
    """

    return pd.Series(np.where(ser < ser.shift(1), -volume,
                              np.where(ser > ser.shift(1), volume, 0)
                              ).cumsum(), index=ser.index)


def WOBV(ser: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Weighted OBV
    Can also be seen as an OBV indicator that takes the price differences into account.
    In a regular OBV, a high volume bar can make a huge difference,
    even if the price went up only 0.01, and it it goes down 0.01
    instead, that huge volume makes the OBV go down, even though
    hardly anything really happened.
    """

    return (volume * ser.diff()).cumsum()


def VZO(ser: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """VZO uses price, previous price and moving averages to compute its oscillating value.
    It is a leading indicator that calculates buy and sell signals based on oversold / overbought conditions.
    Oscillations between the 5% and 40% levels mark a bullish trend zone, while oscillations between -40% and 5% mark a
    bearish trend zone.
    Meanwhile, readings above 40% signal an overbought condition, while readings above 60% signal an extremely
    overbought condition.
    Alternatively, readings below -40% indicate an oversold condition, which becomes extremely oversold below -60%."""

    r = ser.diff().apply(lambda a: (a > 0) - (a < 0)) * volume
    dvma = EMA(r, period=period)
    vma = EMA(volume, period=period)

    return 100 * (dvma / vma)


def PZO(ser: pd.Series, period: int = 14) -> pd.Series:
    """
    The formula for PZO depends on only one condition: if today's closing price is higher than yesterday's
    closing price,
    then the closing price will have a positive value (bullish); otherwise it will have a negative value (bearish).
    source: https://traders.com/Documentation/FEEDbk_docs/2011/06/Khalil.html

    :period: Specifies the number of Periods used for PZO calculation
    """

    r = np.sign(ser.diff()) * ser
    cp = EMA(r, period)
    tc = EMA(ser, period)

    return 100 * (cp / tc)


def EFI(ser: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Elder's Force Index is an indicator that uses price and volume to assess the power
     behind a move or identify possible turning points."""

    # https://tradingsim.com/blog/elders-force-index/
    fi = ser.diff() * volume
    return EMA(fi, period)


def CFI(ser: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Cummulative Force Index.
    Adopted from  Elder's Force Index.
    """

    fi1 = volume * ser.diff()
    cfi = EMA(fi1, period=10)
    return cfi.cumsum()


def EBBP_bull(ser: pd.Series, high: pd.Series) -> pd.Series:
    """Bull power and bear power by Dr. Alexander Elder show where today’s high and low lie relative to thday EMApd."""

    return high - EMA(ser, 13)


def EBBP_bear(ser: pd.Series, low: pd.Series) -> pd.Series:
    """Bull power and bear power by Dr. Alexander Elder show where today’s high and low lie relative to thday EMApd."""

    return low - EMA(ser, 13)


def EBBP_df(ser: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Bull power and bear power by Dr. Alexander Elder show where today’s high and low lie relative to thday EMApd."""

    bull_power = EBBP_bull(ser, high)
    bear_power = EBBP_bear(ser, low)
    return pd.concat([bull_power, bear_power], axis=1)


def COPP(ser: pd.Series) -> pd.Series:
    """The Coppock Curve is a momentum indicator, it signals buying opportunities when the indicator moved
    from negative territory to positive territory."""

    roc1 = ROC(ser, 14)
    roc2 = ROC(ser, 11)

    return EMA(roc1 + roc2, period=10)


def CMO(ser: pd.Series, period: int = 9, factor: int = 100) -> pd.Series:
    """
    Chande Momentum Oscillator (CMO) - technical momentum indicator invented by the technical analyst Tushar Chande.
    It is created by calculating the difference between the sum of all recent gains and the sum of all recent losses
    and then dividing the result by the sum of all price movement over the period.
    This oscillator is similar to other momentum indicators such as the Relative Strength Index and the
    Stochastic Oscillator because it is range bounded (+100 and -100)."""

    # get the price diff
    up = ser.diff()
    down = up.copy()

    # positive gains (up) and negative gains (down) Series
    up[up < 0] = 0
    down[down > 0] = 0

    # EMAs of ups and downs
    _gain = EMA(up, period=period)
    _loss = EMA(down, period=period)

    return factor * ((_gain - _loss) / (_gain + _loss))


def APZ(ser: pd.Series, high: pd.Series, low: pd.Series, period: int = 21, dev_factor: int = 2,
        ma: pd.Series = None) -> pd.DataFrame:
    """
    The adaptive price zone (APZ) is a technical indicator developed by Lee Leibfarth.

    The APZ is a volatility based indicator that appears as a set of bands placed over a price chart.

    Especially useful in non-trending, choppy markets,

    the APZ was created to help traders find potential turning points in the markets.
    """

    if ma is None:
        ma = DEMA(ser, period)
    price_range = EMA(high - low, period=period)
    volatility_value = EMA(price_range, period=period)

    # upper_band = dev_factor * volatility_value + dema
    upper_band = ma + (volatility_value * dev_factor)
    upper_band.name = "UPPER"
    lower_band = ma - (volatility_value * dev_factor)
    lower_band.name = "LOWER"

    return pd.concat([upper_band, lower_band], axis=1)


def SQZMI(ser: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20,
          std_multiplier: float = 2, atr_period: int = 10, ma: pd.Series = None) -> pd.Series:
    """
    Squeeze Momentum Indicator

    The Squeeze indicator attempts to identify periods of consolidation in a market.
    In general the market is either in a period of quiet consolidation or vertical price discovery.
    By identifying these calm periods, we have a better opportunity of getting into trades with the potential for
    larger moves.
    Once a market enters into a “squeeze”, we watch the overall market momentum to help forecast the market direction
    and await a release of market energy.

    :param ser: pd.Series the time serie to compute on
    :param high: pd.Series high value
    :param low: pd.Series low values
    :param close: pd.Series close values
    :param period: int - number of periods to take into consideration
    :param std_multiplier: float
    :param atr_period: int
    :param ma : pd.Series, override internal calculation which uses SMA with moving average of your choice
    :return pd.Series: indicator calcs as pandas Series. SQZMI is bool True/False, if True squeeze is on.
    If false,squeeeze has fired.
    """

    if ma is None:
        ma = SMA(ser, period)

    bb_lower = BBANDS_lower(ser, period=period, std_multiplier=std_multiplier, ma=ma)
    bb_upper = BBANDS_upper(ser, period=period, std_multiplier=std_multiplier, ma=ma)
    kc_lower = KC_lower(ser, high, low, close, period=period, atr_period=atr_period, kc_mult=1.5, ma=ma)
    kc_upper = KC_upper(ser, high, low, close, period=period, atr_period=atr_period, kc_mult=1.5, ma=ma)

    return pd.Series(np.where((bb_lower > kc_lower) and (bb_upper < kc_upper), True, False),
                     name="SQZMI",
                     index=ser.index)


def MSD(ser: pd.Series, period: int = 21) -> pd.Series:
    """
    Standard deviation is a statistical term that measures the amount of variability or dispersion around an average.
    Standard deviation is also a measure of volatility. Generally speaking, dispersion is the difference between the
    actual value and the average value.
    The larger this dispersion or variability is, the higher the standard deviation.
    Standard Deviation values rise significantly when the analyzed contract of indicator change in value dramatically.
    When markets are stable, low Standard Deviation readings are normal.
    Low Standard Deviation readings typically tend to come before significant upward changes in price.
    Analysts generally agree that high volatility is part of major tops, while low volatility accompanies major bottoms.

    :period: Specifies the number of Periods used for MSD calculation
    """

    return ser.rolling(period).std(ddof=0)


def STC(ser: pd.Series, period_fast: int = 23, period_slow: int = 50, k_period: int = 10,
        d_period: int = 3) -> pd.Series:
    """
    The Schaff Trend Cycle (Oscillator) can be viewed as Double Smoothed
    Stochastic of the MACD.

    Schaff Trend Cycle - Three input values are used with the STC:
    – Sh: shorter-term Exponential Moving Average with a default period of 23
    – Lg: longer-term Exponential Moving Average with a default period of 50
    – Cycle, set at half the cycle length with a default value of 10. (Stoch K-period)
    - Smooth, set at smoothing at 3 (Stoch D-period)

    The STC is calculated in the following order:
    EMA1 = EMA (Close, fast_period);
    EMA2 = EMA (Close, slow_period);
    MACD = EMA1 – EMA2.
    Second, the 10-period Stochastic from the MACD values is calculated:
    STOCH_K, STOCH_D  = StochasticFull(MACD, k_period, d_period)  // Stoch of MACD
    STC =  average(STOCH_D, d_period) // second smoothed

    In case the STC indicator is decreasing, this indicates that the trend cycle
    is falling, while the price tends to stabilize or follow the cycle to the downside.
    In case the STC indicator is increasing, this indicates that the trend cycle
    is up, while the price tends to stabilize or follow the cycle to the upside.
    """

    macd = MACD(ser, period_fast=period_fast, period_slow=period_slow)

    stok = ((macd - macd.rolling(window=k_period).min()) / (
            macd.rolling(window=k_period).max() - macd.rolling(window=k_period).min())) * 100.

    stod = stok.rolling(window=d_period).mean()
    return stod.rolling(window=d_period).mean()  # "double smoothed"


def EVSTC(
        ser: pd.Series,
        weights: pd.Series,
        period_fast: int = 12,
        period_slow: int = 30,
        k_period: int = 10,
        d_period: int = 3
) -> pd.Series:
    """Modification of Schaff Trend Cycle using EVWMA MACD for calculation"""

    ev_macd = EV_MACD(ser, weights, period_fast=period_fast, period_slow=period_slow)

    stok = ((ev_macd - ev_macd.rolling(window=k_period).min()) / (
            ev_macd.rolling(window=k_period).max() - ev_macd.rolling(window=k_period).min())) * 100

    stod = stok.rolling(window=d_period).mean()
    return stod.rolling(window=d_period).mean()


def WAVEPM(ser: pd.Series, period: int = 14, lookback_period: int = 100) -> pd.Series:
    """
    The Wave PM (Whistler Active Volatility Energy Price Mass) indicator is an oscillator described in the Mark
    Whistler’s book “Volatility Illuminated”.

    :param pd.Series ser: data
    :param int period: period for moving average
    :param int lookback_period: period for oscillator lookback
    :return Series: WAVE PM
    """

    ma = ser.rolling(window=period).mean()
    std = ser.rolling(window=period).std(ddof=0)

    dev = 3.2 * std
    power = np.power(dev / ma, 2)

    variance = power.rolling(window=lookback_period).sum() / lookback_period
    calc_dev = np.sqrt(variance) * ma
    y = (dev / calc_dev)
    return np.tanh(y)


# --------------------------------- CANDLES -------------------------------------

def VWAP(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    The volume weighted average price (VWAP) is a trading benchmark used especially in pension plans.
    VWAP is calculated by adding up the dollars traded for every transaction (price multiplied by number of shares traded) and then dividing
    by the total shares traded for the day.
    """

    return ((volume * TP(high, low, close)).cumsum()) / volume.cumsum()


def SAR(high: pd.Series, low: pd.Series, af: int = 0.02, amax: int = 0.2) -> pd.Series:
    """SAR stands for “stop and reverse,” which is the actual indicator used in the system.
    SAR trails price as the trend extends over time. The indicator is below prices when prices are rising and above prices when prices are falling.
    In this regard, the indicator stops and reverses when the price trend reverses and breaks above or below the indicator."""

    # Starting values
    sig0, xpt0, af0 = True, high[0], af
    _sar = [low[0] - (high - low).std()]

    for i in range(1, len(high)):
        sig1, xpt1, af1 = sig0, xpt0, af0

        lmin = min(low[i - 1], low[i])
        lmax = max(high[i - 1], high[i])

        if sig1:
            sig0 = low[i] > _sar[-1]
            xpt0 = max(lmax, xpt1)
        else:
            sig0 = high[i] >= _sar[-1]
            xpt0 = min(lmin, xpt1)

        if sig0 == sig1:
            sari = _sar[-1] + (xpt1 - _sar[-1]) * af1
            af0 = min(amax, af1 + af)

            if sig0:
                af0 = af0 if xpt0 > xpt1 else af1
                sari = min(sari, lmin)
            else:
                af0 = af0 if xpt0 < xpt1 else af1
                sari = max(sari, lmax)
        else:
            af0 = af
            sari = xpt0

        _sar.append(sari)

    return pd.Series(_sar, index=high.index)


def PSAR(high: pd.Series, low: pd.Series, close: pd.Series, iaf: int = 0.02, maxaf: int = 0.2) -> pd.DataFrame:
    """
    The parabolic SAR indicator, developed by J. Wells Wilder, is used by traders to determine trend direction and potential reversals in price.
    The indicator uses a trailing stop and reverse method called "SAR," or stop and reverse, to identify suitable exit and entry points.
    Traders also refer to the indicator as the parabolic stop and reverse, parabolic SAR, or PSAR.
    https://www.investopedia.com/terms/p/parabolicindicator.asp
    https://virtualizedfrog.wordpress.com/2014/12/09/parabolic-sar-implementation-in-python/
    """

    length = len(high)
    psar = close[0: len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    hp = high[0]
    lp = low[0]

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    psar = pd.Series(psar, name="psar", index=high.index)
    psarbear = pd.Series(psarbull, name="psarbear", index=high.index)
    psarbull = pd.Series(psarbear, name="psarbull", index=high.index)

    return pd.concat([psar, psarbull, psarbear], axis=1)


def DO_df(high: pd.Series, low: pd.Series, upper_period: int = 20, lower_period: int = 5) -> pd.DataFrame:
    """Donchian Channel, a moving average indicator developed by Richard Donchian.
    It plots the highest high and lowest low over the last period time intervals."""

    upper = high.rolling(center=False, window=upper_period).max()
    upper.name = "DO_upper"
    lower = low.rolling(center=False, window=lower_period).min()
    lower.name = "DO_lower"
    middle = (upper + lower) / 2
    middle.name = "DO_middle"

    return pd.concat([lower, middle, upper], axis=1)


def DO_middle(high: pd.Series, low: pd.Series, upper_period: int = 20, lower_period: int = 5) -> pd.Series:
    """Donchian Channel, a moving average indicator developed by Richard Donchian.
    It plots the highest high and lowest low over the last period time intervals."""

    upper = high.rolling(center=False, window=upper_period).max()
    lower = low.rolling(center=False, window=lower_period).min()
    middle = (upper + lower) / 2

    return middle


def DO_upper(high: pd.Series, period: int = 20) -> pd.Series:
    """Donchian Channel, a moving average indicator developed by Richard Donchian.
    It plots the highest high and lowest low over the last period time intervals."""

    return high.rolling(center=False, window=period).max()


def DO_lower(low: pd.Series, period: int = 5) -> pd.Series:
    """Donchian Channel, a moving average indicator developed by Richard Donchian.
    It plots the highest high and lowest low over the last period time intervals."""

    return low.rolling(center=False, window=period).min()


def DM_minus(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    return EMA(pd.Series(minus), period)


def DM_plus(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)

    return EMA(pd.Series(plus), period)


def DM_df(high: pd.Series, low: pd.Series, period: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    dmplus = SMA(pd.Series(plus), period)
    dmplus.name = "DM_plus"
    dmminus = SMA(pd.Series(minus), period)
    dmminus.name = "DM_minus"
    return pd.concat([dmminus, dmplus], axis=1)


def DMI_df(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
     for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
     relative strength index. DMI tells you when to be long or short.
     It is especially useful for trend trading strategies because it differentiates between strong and weak trends,
     allowing the trader to enter only the strongest trends.
    source: https://www.tradingview.com/wiki/Directional_Movement_(DMI)#CALCULATION

    :period: Specifies the number of Periods used for DMI calculation
    """

    up_move = high.diff()
    down_move = -low.diff()

    plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = ATR(high, low, close, period)
    diplus = 100. * RMA(plus / atr, period=period)
    diplus.name = "DI_plus"
    diminus = 100. * RMA(minus / atr, period=period)
    diminus.name = "DI_minus"

    return pd.concat([diminus, diplus], axis=1)


def DMI_minus(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
     for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
     relative strength index. DMI tells you when to be long or short.
     It is especially useful for trend trading strategies because it differentiates between strong and weak trends,
     allowing the trader to enter only the strongest trends.
    source: https://www.tradingview.com/wiki/Directional_Movement_(DMI)#CALCULATION

    :period: Specifies the number of Periods used for DMI calculation
    """

    up_move = high.diff()
    down_move = -low.diff()

    minus = np.where((up_move < down_move) & (down_move > 0), down_move, 0)

    return 100. * (minus / ATR(high, low, close, period)).ewm(alpha=1 / period).mean()


def DMI_plus(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
     for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
     relative strength index. DMI tells you when to be long or short.
     It is especially useful for trend trading strategies because it differentiates between strong and weak trends,
     allowing the trader to enter only the strongest trends.
    source: https://www.tradingview.com/wiki/Directional_Movement_(DMI)#CALCULATION

    :period: Specifies the number of Periods used for DMI calculation
    """

    up_move = high.diff()
    down_move = -low.diff()

    plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)

    return 100. * (plus / ATR(high, low, close, period)).ewm(alpha=1 / period).mean()


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """The A.D.X. is 100 * smoothed moving average of absolute value (DMI +/-) divided by (DMI+ + DMI-). ADX does not indicate trend direction or momentum,
    only trend strength. Generally, A.D.X. readings below 20 indicate trend weakness,
    and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50"""

    dmi = DMI_df(high, low, close, period)
    return 100. * (abs(dmi["DI_plus"] - dmi["DI_minus"]) / (dmi["DI_plus"] + dmi["DI_minus"])).ewm(
        alpha=1 / period).mean()


def PIVOT(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Pivot Points are significant support and resistance levels that can be used to determine potential trades.
    The pivot points come as a technical analysis indicator calculated using a financial instrument’s high, low, and close value.
    The pivot point’s parameters are usually taken from the previous day’s trading range.
    This means you’ll have to use the previous day’s range for today’s pivot points.
    Or, last week’s range if you want to calculate weekly pivot points or, last month’s range for monthly pivot points and so on.
    """
    # pivot is calculated of the previous trading session
    high_shifted = high.shift()
    low_shifted = low.shift()
    close_shifted = close.shift()

    pivot = TP(high_shifted, low_shifted, close_shifted)  # pivot is basically a lagging TP
    pivot.name = "pivot"

    s1 = (pivot * 2) - high_shifted
    s1.name = "s1"
    s2 = pivot - (high_shifted - low_shifted)
    s2.name = "s2"
    s3 = low_shifted - (2 * (high_shifted - pivot))
    s3.name = "s3"
    s4 = low_shifted - (3 * (high_shifted - pivot))
    s4.name = "s4"

    r1 = (pivot * 2) - low_shifted
    r1.name = "r1"
    r2 = pivot + (high_shifted - low_shifted)
    r2.name = "r2"
    r3 = high_shifted + (2 * (pivot - low_shifted))
    r3.name = "r3"
    r4 = high_shifted + (3 * (pivot - low_shifted))
    r4.name = "r4"

    return pd.concat([pivot, s1, s2, s3, s4, r1, r2, r3, r4], axis=1)


def PIVOT_FIB(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Fibonacci pivot point levels are determined by first calculating the classic pivot point,
    then multiply the previous day’s range with its corresponding Fibonacci level.
    Most traders use the 38.2%, 61.8% and 100% retracements in their calculations.
    """

    high_shifted = high.shift()
    low_shifted = low.shift()
    close_shifted = close.shift()

    pivot = TP(high_shifted, low_shifted, close_shifted)  # pivot is basically a lagging TP
    pivot.name = "pivot"

    r4 = pivot + ((high_shifted - low_shifted) * 1.382)
    r4.name = "r4"
    r3 = pivot + ((high_shifted - low_shifted) * 1)
    r4.name = "r3"
    r2 = pivot + ((high_shifted - low_shifted) * 0.618)
    r4.name = "r2"
    r1 = pivot + ((high_shifted - low_shifted) * 0.382)
    r4.name = "r1"
    s1 = pivot - ((high_shifted - low_shifted) * 0.382)
    r4.name = "s1"
    s2 = pivot - ((high_shifted - low_shifted) * 0.618)
    r4.name = "s2"
    s3 = pivot - ((high_shifted - low_shifted) * 1)
    r4.name = "s3"
    s4 = pivot - ((high_shifted - low_shifted) * 1.382)
    r4.name = "s4"

    return pd.concat([pivot, s1, s2, s3, s4, r1, r2, r3, r4], axis=1)


def STOCH(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic oscillator %K
     The stochastic oscillator is a momentum indicator comparing the closing price of a security
     to the range of its prices over a certain period of time.
     The sensitivity of the oscillator to market movements is reducible by adjusting that time
     period or by taking a moving average of the result.
    """

    highest_high = high.rolling(center=False, window=period).max()
    lowest_low = low.rolling(center=False, window=period).min()

    return (close - lowest_low) / (highest_high - lowest_low) * 100


def STOCHD(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 3, stoch_period: int = 14) -> pd.Series:
    """Stochastic oscillator %D
    STOCH%D is a 3 period simple moving average of %K.
    """

    return STOCH(high, low, close, stoch_period).rolling(center=False, window=period).mean()


def AO(high: pd.Series, low: pd.Series, slow_period: int = 34, fast_period: int = 5) -> pd.Series:
    """'EMA',
    Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages.
    The Simple Moving Averages that are used are not calculated using closing price but rather each bar's midpoints.
    AO is generally used to affirm trends or to anticipate possible reversals. """

    slow = ((high + low) / 2).rolling(window=slow_period).mean()
    fast = ((high + low) / 2).rolling(window=fast_period).mean()

    return fast - slow


def MI(high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    """Developed by Donald Dorsey, the Mass Index uses the high-low range to identify trend reversals based on range expansions.
    In this sense, the Mass Index is a volatility indicator that does not have a directional bias.
    Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend."""

    _range = high - low
    EMA9 = _range.ewm(span=period, ignore_na=False).mean()
    DEMA9 = EMA9.ewm(span=period, ignore_na=False).mean()
    mass = EMA9 / DEMA9

    return mass.rolling(window=25).sum()


def BOP(high: pd.Series, low: pd.Series, price_open: pd.Series, close: pd.Series) -> pd.Series:
    """Balance Of Power indicator"""

    return (close - price_open) / (high - low)


def VORTEX_df(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """The Vortex indicator plots two oscillating lines, one to identify positive trend movement and the other
     to identify negative price movement.
     Indicator construction revolves around the highs and lows of the last two days or periods.
     The distance from the current high to the prior low designates positive trend movement while the
     distance between the current low and the prior high designates negative trend movement.
     Strongly positive or negative trend movements will show a longer length between the two numbers while
     weaker positive or negative trend movement will show a shorter length."""

    VMP = (high - low.shift()).abs()
    VMM = (low - high.shift()).abs()

    VMPx = VMP.rolling(window=period).sum()
    VMMx = VMM.rolling(window=period).sum()
    tr = TR(high, low, close).rolling(window=period).sum()

    VIp = (VMPx / tr).interpolate(method="index")
    VIp.name = "VI_positive"
    VIm = (VMMx / tr).interpolate(method="index")
    VIp.name = "VI_negative"

    return pd.concat([VIm, VIp], axis=1)


def VORTEX_negative(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """The Vortex indicator plots two oscillating lines, one to identify positive trend movement and the other
     to identify negative price movement.
     Indicator construction revolves around the highs and lows of the last two days or periods.
     The distance from the current high to the prior low designates positive trend movement while the
     distance between the current low and the prior high designates negative trend movement.
     Strongly positive or negative trend movements will show a longer length between the two numbers while
     weaker positive or negative trend movement will show a shorter length."""

    VMM = (low - high.shift()).abs()

    VMMx = VMM.rolling(window=period).sum()
    tr = TR(high, low, close).rolling(window=period).sum()

    return (VMMx / tr).interpolate(method="index")


def VORTEX_positive(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """The Vortex indicator plots two oscillating lines, one to identify positive trend movement and the other
     to identify negative price movement.
     Indicator construction revolves around the highs and lows of the last two days or periods.
     The distance from the current high to the prior low designates positive trend movement while the
     distance between the current low and the prior high designates negative trend movement.
     Strongly positive or negative trend movements will show a longer length between the two numbers while
     weaker positive or negative trend movement will show a shorter length."""

    VMP = (high - low.shift()).abs()

    VMPx = VMP.rolling(window=period).sum()
    tr = TR(high, low, close).rolling(window=period).sum()

    return (VMPx / tr).interpolate(method="index")


def ADL(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """The accumulation/distribution line was created by Marc Chaikin to determine the flow of money into or out of a security.
    It should not be confused with the advance/decline line. While their initials might be the same, these are entirely different indicators,
    and their uses are different as well. Whereas the advance/decline line can provide insight into market movements,
    the accumulation/distribution line is of use to traders looking to measure buy/sell pressure on a security or confirm the strength of a trend."""

    MFM = ((close - low) - (low - close)) / (high - low)  # Money flow multiplier
    MFV = MFM * volume
    return MFV.cumsum()


def CHAIKIN(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Chaikin Oscillator, named after its creator, Marc Chaikin, the Chaikin oscillator is an oscillator that measures the accumulation/distribution
     line of the moving average convergence divergence (MACD). The Chaikin oscillator is calculated by subtracting a 10-day exponential moving average (EMA)
     of the accumulation/distribution line from a three-day EMA of the accumulation/distribution line, and highlights the momentum implied by the
     accumulation/distribution line."""

    return (ADL(high, low, close, volume).ewm(span=3, min_periods=2).mean()
            - ADL(high, low, close, volume).ewm(span=10, min_periods=9).mean())


def EMV(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Ease of Movement (EMV) is a volume-based oscillator that fluctuates above and below the zero line.
    As its name implies, it is designed to measure the 'ease' of price movement.
    prices are advancing with relative ease when the oscillator is in positive territory.
    Conversely, prices are declining with relative ease when the oscillator is in negative territory."""

    distance = ((high + low) - (high.shift() + low.shift())) / 2
    box_ratio = volume / (high - low) / 1000000.

    _emv = distance / box_ratio

    return _emv.rolling(window=period).mean()


def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, constant: float = 0.015) -> pd.Series:
    """Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
    CCI measures the current price level relative to an average price level over a given period of time.
    The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100.
    Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition.
    As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels.

    source: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    :param high: pd.Series
    :param low: pd.Series
    :param close: pd.Series
    :param period: int - number of periods to take into consideration
    :param constant: float factor: the constant at .015 to ensure that approximately 70 to 80 percent of CCI values would fall between -100 and +100.
    :return pd.Series: result is pandas.Series
    """

    tp = TP(high, low, close)
    tp_rolling = tp.rolling(window=period, min_periods=0)
    # calculate MAD (Mean Deviation)
    # https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/other-measures-of-spread/a/mean-absolute-deviation-mad-review
    mad = (tp_rolling - tp_rolling.mean()).abs().mean()
    return (tp - tp_rolling.mean()) / (constant * mad)


def BASP_buy(low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.Series:
    """BASP indicator serves to identify buying and selling pressure."""

    bp = close - low
    bpavg = bp.ewm(span=period).mean()

    nbp = bp / bpavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    return nbp * nv


def BASP_sell(high: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.Series:
    """BASP indicator serves to identify buying and selling pressure."""

    sp = high - close
    spavg = sp.ewm(span=period).mean()

    nsp = sp / spavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    return nsp * nv


def BASP_df(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.DataFrame:
    """BASP indicator serves to identify buying and selling pressure."""

    sp = high - close
    bp = close - low
    spavg = sp.ewm(span=period).mean()
    bpavg = bp.ewm(span=period).mean()

    nbp = bp / bpavg
    nsp = sp / spavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    nbfraw = nbp * nv
    nbfraw.name = "BASP_buy"
    nsfraw = nsp * nv
    nsfraw.name = "BASP_sell"

    return pd.concat([nbfraw, nsfraw], axis=1)


def BASPN_buy(low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.Series:
    """
    Normalized BASP indicator
    """

    bp = close - low
    bpavg = bp.ewm(span=period).mean()

    nbp = bp / bpavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    return (nbp * nv).ewm(span=20).mean()


def BASPN_sell(high: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.Series:
    """
    Normalized BASP indicator
    """

    sp = high - close
    spavg = sp.ewm(span=period).mean()

    nsp = sp / spavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    return (nsp * nv).ewm(span=20).mean()


def BASPN_df(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 40) -> pd.DataFrame:
    """
    Normalized BASP indicator
    """

    sp = high - close
    bp = close - low
    spavg = sp.ewm(span=period).mean()
    bpavg = bp.ewm(span=period).mean()

    nbp = bp / bpavg
    nsp = sp / spavg

    varg = volume.ewm(span=period).mean()
    nv = volume / varg

    nbf = (nbp * nv).ewm(span=20).mean()
    nbf.name = "BASPN_buy"
    nsf = (nsp * nv).ewm(span=20).mean()
    nsf.name = "BASPN_sell"

    return pd.concat([nbf, nsf], axis=1)


def CHANDELIER_long(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22, k: int = 3) -> pd.Series:
    """
    Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR).

    The indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends.

    Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend.
    """

    return high.rolling(window=period).max() - ATR(high, low, close, 22) * k


def CHANDELIER_short(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22, k: int = 3) -> pd.Series:
    """
    Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR).

    The indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends.

    Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend.
    """

    return low.rolling(window=period).min() + ATR(high, low, close, 22) * k


def CHANDELIER_df(high: pd.Series, low: pd.Series, close: pd.Series, short_period: int = 22, long_period: int = 22,
                  k: int = 3) -> pd.DataFrame:
    """
    Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR).

    The indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends.

    Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend.
    """

    l = CHANDELIER_long(high, low, close, long_period, k)
    l.name = "long"
    s = CHANDELIER_short(high, low, close, short_period, k)
    l.name = "short"
    return pd.concat([s, l], axis=1)


def QSTICK(close: pd.Series, open_price: pd.Series, period: int = 14) -> pd.Series:
    """
    QStick indicator shows the dominance of black (down) or white (up) candlesticks, which are red and green in Chart,
    as represented by the average open to close change for each of past N days."""

    _close = close.tail(period)
    _open = open_price.tail(period)

    return (_close - _open) / period


def TMF(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """Indicator by Colin Twiggs which improves upon CMF.
    source: https://user42.tuxfamily.org/chart/manual/Twiggs-Money-Flow.html"""

    close_shifted = close.shift(1)
    ll = np.minimum(low, close_shifted)
    hh = np.maximum(high, close_shifted)

    range = 2 * ((close - ll) / (hh - ll)) - 1

    ema_v = volume.ewm(span=period).mean()
    ema_rv = (range * volume).ewm(span=period).mean()

    return 100. * ema_rv / ema_v


def WTO_1(high: pd.Series, low: pd.Series, close: pd.Series, channel_length: int = 10,
          average_length: int = 21) -> pd.Series:
    """
    Wave Trend Oscillator
    source: http://www.fxcoaching.com/WaveTrend/
    """

    ap = TP(high, low, close)
    esa = ap.ewm(span=average_length).mean()
    d = pd.Series(
        (ap - esa).abs().ewm(span=channel_length).mean(), name="d"
    )
    ci = (ap - esa) / (0.015 * d)

    return ci.ewm(span=average_length).mean()


def WTO_2(high: pd.Series, low: pd.Series, close: pd.Series, channel_length: int = 10,
          average_length: int = 21) -> pd.Series:
    """
    Wave Trend Oscillator
    source: http://www.fxcoaching.com/WaveTrend/
    """

    wt1 = WTO_1(high, low, close, channel_length, average_length)
    return wt1.rolling(window=4).mean()


def WTO_df(high: pd.Series, low: pd.Series, close: pd.Series, channel_length: int = 10,
           average_length: int = 21) -> pd.DataFrame:
    """
    Wave Trend Oscillator
    source: http://www.fxcoaching.com/WaveTrend/
    """

    ap = TP(high, low, close)
    esa = ap.ewm(span=average_length).mean()
    d = pd.Series(
        (ap - esa).abs().ewm(span=channel_length).mean(), name="d"
    )
    ci = (ap - esa) / (0.015 * d)

    wt1 = pd.Series(ci.ewm(span=average_length).mean(), name="WT1.")
    wt2 = pd.Series(wt1.rolling(window=4).mean(), name="WT2.")

    return pd.concat([wt1, wt2], axis=1)


def FISH(high: pd.Series, low: pd.Series, period: int = 10) -> pd.Series:
    """
    Fisher Transform was presented by John Ehlers. It assumes that price distributions behave like square waves.
    """

    med = (high + low) / 2
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    raw = (2 * ((med - ndaylow) / (ndayhigh - ndaylow))) - 1
    smooth = raw.ewm(span=5).mean()
    _smooth = smooth.fillna(0)

    return (np.log((1 + _smooth) / (1 - _smooth))).ewm(span=3).mean()


def ICHIMOKU_tenkan(high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """

    return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2  ## conversion line


def ICHIMOKU_kijun(high: pd.Series, low: pd.Series, period: int = 26) -> pd.Series:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """

    return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2  ## base line


def ICHIMOKU_senkou_a(high: pd.Series, low: pd.Series, tenkan_period: int = 9, kijun_period: int = 26) -> pd.Series:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """
    tenkan_sen = ICHIMOKU_tenkan(high, low, tenkan_period)
    kijun_sen = ICHIMOKU_kijun(high, low, kijun_period)

    return ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)  ## Leading span


def ICHIMOKU_senkou_b(high: pd.Series, low: pd.Series, kijun_period: int = 26, senkou_period: int = 52) -> pd.Series:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """
    return ((high.rolling(window=senkou_period).max() + low.rolling(window=senkou_period).min()) / 2).shift(
        kijun_period)


def ICHIMOKU_chikou(close: pd.Series, period: int = 26) -> pd.Series:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """

    return close.shift(-period)


def ICHIMOKU_df(high: pd.Series, low: pd.Series, close: pd.Series, tenkan_period: int = 9, kijun_period: int = 26,
                senkou_period: int = 52, chikou_period: int = 26) -> pd.DataFrame:
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.

    Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
    """

    tenkan_sen = ICHIMOKU_tenkan(high, low, tenkan_period)  ## conversion line
    tenkan_sen.name = "TENKAN"

    kijun_sen = ICHIMOKU_kijun(high, low, kijun_period)
    kijun_sen.name = "KIJUN"

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)  ## Leading span
    senkou_span_a.name = "SENKOU_span_a"

    senkou_span_b = ICHIMOKU_senkou_b(high, low, kijun_period, senkou_period)
    senkou_span_b.name = "SENKOU_span_b"

    chikou_span = ICHIMOKU_chikou(close, chikou_period)
    chikou_span.name = "CHIKOU"

    return pd.concat(
        [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1
    )


def VPT(high: pd.Series, low: pd.Series, close: pd.Series, open_price: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Price Trend
    The Volume Price Trend uses the difference of price and previous price with volume and feedback to arrive at its final form.
    If there appears to be a bullish divergence of price and the VPT (upward slope of the VPT and downward slope of the price) a buy opportunity exists.
    Conversely, a bearish divergence (downward slope of the VPT and upward slope of the price) implies a sell opportunity.
    """

    hilow = (high - low) * 100.
    openclose = (close - open_price) * 100.
    vol = volume / hilow
    spreadvol = (openclose * vol).cumsum()

    return spreadvol + spreadvol


def FVE(high: pd.Series, low: pd.Series, close: pd.Series, open_price: pd.Series, volume: pd.Series, period: int = 22,
        factor: int = 0.3) -> pd.Series:
    """
    FVE is a money flow indicator, but it has two important innovations: first, the F VE takes into account both intra and
    interday price action, and second, minimal price changes are taken into account by introducing a price threshold.
    """

    hl2 = (high + low) / 2
    tp = TP(high, low, close)
    smav = volume.rolling(window=period).mean()
    mf = close - hl2 + tp.diff()
    fclose_mf = factor * close / 100. + mf

    vol_shift = pd.Series(np.where(fclose_mf > 0, volume, np.where(fclose_mf < 0, -volume, 0)))

    _sum = vol_shift.rolling(window=period).sum()

    return _sum / smav / period * 100.


def VFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        period: int = 130, smoothing_factor: int = 3, factor: int = 0.2,
        vfactor: int = 2.5) -> pd.Series:
    """
    This indicator tracks volume based on the direction of price
    movement. It is similar to the On Balance Volume Indicator.
    For more information see "Using Money Flow to Stay with the Trend",
    and "Volume Flow Indicator Performance" in the June 2004 and
    July 2004 editions of Technical Analysis of Stocks and Commodities.

    :period: Specifies the number of Periods used for VFI calculation
    :factor: Specifies the fixed scaling factor for the VFI calculation
    :vfactor: Specifies the cutoff for maximum volume in the VFI calculation
    :smoothing_factor: Specifies the number of periods used in the short moving average
    """

    typical = TP(high, low, close)
    # historical interday volatility and cutoff
    inter = typical.apply(np.log).diff()
    # stdev of linear1
    vinter = inter.rolling(window=30).std()
    cutoff = factor * vinter * close
    cutoff.fillna(0, inplace=True)

    price_change = typical.diff()  # price change
    price_change.fillna(0, inplace=True)
    mav = volume.rolling(center=False, window=period).mean()

    fmav = factor * mav

    added_vol = np.where(volume > fmav, fmav, volume)
    multiplier = np.where(price_change - cutoff > 0, 1, np.where(price_change + cutoff < 0, -1, 0))

    raw_sum = pd.Series(multiplier * added_vol).rolling(window=period).sum()
    raw_value = raw_sum / mav.shift()

    return raw_value.ewm(ignore_na=False, min_periods=smoothing_factor - 1, span=smoothing_factor).mean()


def WILLIAMS_FRACTAL_df(high: pd.Series, low: pd.Series, period: int = 2) -> pd.DataFrame:
    """
    Williams Fractal Indicator
    Source: https://www.investopedia.com/terms/f/fractal.asp

    :param pd.Series high: data
    :param pd.Series low: data
    :param int period: how many lower highs/higher lows the extremum value should be preceded and followed.
    :return DataFrame: fractals identified by boolean
    """

    window_size = period * 2 + 1
    high_rolling = high.rolling(window=window_size, center=True)
    low_rolling = low.rolling(window=window_size, center=True)

    bearish_fractals = high_rolling.apply(lambda x: True if x[period] == max(x) else False, raw=True)
    bearish_fractals.name = "WILLIAMS_FRACTAL_bearish"
    bullish_fractals = low_rolling.apply(lambda x: True if x[period] == min(x) else False, raw=True)
    bullish_fractals.name = "WILLIAMS_FRACTAL_bullish"

    return pd.concat([bearish_fractals, bullish_fractals], axis=1)


def WILLIAMS_FRACTAL_bearish(high: pd.Series, period: int = 2) -> pd.Series:
    """
    Williams Fractal Indicator
    Source: https://www.investopedia.com/terms/f/fractal.asp

    :param pd.Series high: data
    :param int period: how many lower highs/higher lows the extremum value should be preceded and followed.
    :return DataFrame: fractals identified by boolean
    """

    window_size = period * 2 + 1
    high_rolling = high.rolling(window=window_size, center=True)

    return high_rolling.apply(lambda x: True if x[period] == max(x) else False, raw=True)


def WILLIAMS_FRACTAL_bullish(low: pd.Series, period: int = 2) -> pd.Series:
    """
    Williams Fractal Indicator
    Source: https://www.investopedia.com/terms/f/fractal.asp

    :param pd.Series low: data
    :param int period: how many lower highs/higher lows the extremum value should be preceded and followed.
    :return DataFrame: fractals identified by boolean
    """

    window_size = period * 2 + 1
    low_rolling = low.rolling(window=window_size, center=True)

    return low_rolling.apply(lambda x: True if x[period] == min(x) else False, raw=True)


def VC_df(high: pd.Series, low: pd.Series, close: pd.Series, open_price: pd.Series, period: int = 5) -> pd.DataFrame:
    """Value chart
    Implementation based on a book by Mark Helweg & David Stendahl: Dynamic Trading Indicators: Winning with Value Charts and Price Action Profile

    :period: Specifies the number of Periods used for VC calculation
    """

    float_axis = ((high + low) / 2).rolling(window=period).mean()
    vol_unit = (high - low).rolling(window=period).mean() * 0.2

    value_chart_high = (high - float_axis) / vol_unit
    value_chart_high.name = "Value Chart High"
    value_chart_low = (low - float_axis) / vol_unit
    value_chart_low.name = "Value Chart Low"
    value_chart_close = (close - float_axis) / vol_unit
    value_chart_close.name = "Value Chart Close"
    value_chart_open = (open_price - float_axis) / vol_unit
    value_chart_open.name = "Value Chart Open"

    return pd.concat([value_chart_high, value_chart_low, value_chart_close, value_chart_open], axis=1)


def VC_high(high: pd.Series, low: pd.Series, period: int = 5) -> pd.Series:
    """Value chart
    Implementation based on a book by Mark Helweg & David Stendahl: Dynamic Trading Indicators: Winning with Value Charts and Price Action Profile

    :period: Specifies the number of Periods used for VC calculation
    """

    float_axis = ((high + low) / 2).rolling(window=period).mean()
    vol_unit = (high - low).rolling(window=period).mean() * 0.2

    return (high - float_axis) / vol_unit


def VC_low(high: pd.Series, low: pd.Series, period: int = 5) -> pd.Series:
    """Value chart
    Implementation based on a book by Mark Helweg & David Stendahl: Dynamic Trading Indicators: Winning with Value Charts and Price Action Profile

    :period: Specifies the number of Periods used for VC calculation
    """

    float_axis = ((high + low) / 2).rolling(window=period).mean()
    vol_unit = (high - low).rolling(window=period).mean() * 0.2

    return (low - float_axis) / vol_unit


def VC_close(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 5) -> pd.DataFrame:
    """Value chart
    Implementation based on a book by Mark Helweg & David Stendahl: Dynamic Trading Indicators: Winning with Value Charts and Price Action Profile

    :period: Specifies the number of Periods used for VC calculation
    """

    float_axis = ((high + low) / 2).rolling(window=period).mean()
    vol_unit = (high - low).rolling(window=period).mean() * 0.2

    return (close - float_axis) / vol_unit


def VC_open(high: pd.Series, low: pd.Series, open_price: pd.Series, period: int = 5) -> pd.Series:
    """Value chart
    Implementation based on a book by Mark Helweg & David Stendahl: Dynamic Trading Indicators: Winning with Value Charts and Price Action Profile

    :period: Specifies the number of Periods used for VC calculation
    """

    float_axis = ((high + low) / 2).rolling(window=period).mean()
    vol_unit = (high - low).rolling(window=period).mean() * 0.2

    return (open_price - float_axis) / vol_unit
