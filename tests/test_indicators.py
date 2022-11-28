import os
import pandas as pd
import talib

from panti.indicators import MACD, MACD_signal, MACD_df, MOM, ROC, RSI, BBANDS_df, BBANDS_middle, BBANDS_lower, \
    BBANDS_upper, \
    UO, TRIX

from panti.moving_averages import EMA

data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/sample.json')

ohlc = pd.read_json(data_file).drop(columns=["time"]).reset_index(drop=True)


def test_trix():
    """test TRIX"""

    ma = TRIX(ohlc["close"], 20)
    talib_ma = talib.TRIX(ohlc['close'], timeperiod=20)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_macd():
    """test MACD"""

    talib_macd = talib.MACD(ohlc['close'])

    # TALIB is bugged when computing the fast EMA:
    #   in TALIB initialization of fast EMA is wrong (first term) because TALIB considers a beginning at
    #   period_slow - period_fast instead of first index of the time serie
    #   i.e. macd_as_talib = EMA(ser.iloc[period_slow - period_fast:], period=period_fast) - EMA(ser, period=period_slow)
    # As EMA is decaying with time, we must compare the last part of the signal (we compare here from the third
    # to the end)
    macd_as_talib = EMA(ohlc["close"].iloc[26 - 12:], period=12) - EMA(ohlc["close"], period=26)
    assert ((macd_as_talib - talib_macd[0]).abs().fillna(0.) < 1.e-10).all()

    macd = MACD(ohlc["close"])
    compare_idx = len(macd) // 3
    assert ((macd.iloc[compare_idx:] - talib_macd[0].iloc[compare_idx:]).abs().fillna(0.) < 1.e-10).all()

    signal = MACD_signal(ohlc["close"])
    assert ((signal.iloc[compare_idx:] - talib_macd[1].iloc[compare_idx:]).abs().fillna(0.) < 1.e-10).all()

    macd_df = MACD_df(ohlc["close"])
    assert ((macd_df["MACD"].iloc[compare_idx:] - talib_macd[0].iloc[compare_idx:]).abs().fillna(0.) < 1.e-10).all()
    assert ((macd_df["MACD_signal"].iloc[compare_idx:] - talib_macd[1].iloc[compare_idx:]).abs().fillna(
        0.) < 1.e-10).all()
    assert ((macd_df["MACD_hist"].iloc[compare_idx:] - talib_macd[2].iloc[compare_idx:]).abs().fillna(
        0.) < 1.e-10).all()


def test_mom():
    """test MOM"""

    mom = MOM(ohlc["close"], 15)
    talib_mom = talib.MOM(ohlc['close'], 15)

    assert ((mom - talib_mom).abs().fillna(0.) < 1.e-10).all()


def test_roc():
    """test ROC"""

    roc = ROC(ohlc["close"], 10)
    talib_roc = talib.ROC(ohlc["close"], 10)

    assert ((roc - talib_roc).abs().fillna(0.) < 1.e-10).all()


def test_rsi():
    """test RSI"""

    rsi = RSI(ohlc["close"], 9)
    talib_rsi = talib.RSI(ohlc['close'], 9)

    assert ((rsi - talib_rsi).abs().fillna(0.) < 1.e-10).all()


def test_bbands():
    """test BBANDS"""
    period = 20
    std_multiplier = 2.

    talib_bb = talib.BBANDS(ohlc['close'], timeperiod=period, nbdevup=std_multiplier, nbdevdn=std_multiplier)

    # check middle band
    bb_df = BBANDS_df(ohlc['close'], period, std_multiplier=std_multiplier)
    assert ((bb_df["BB_middle"] - talib_bb[1]).abs().fillna(0.) < 1.e-10).all()

    # check the upper band
    assert ((bb_df["BB_upper"] - talib_bb[0]).abs().fillna(0.) < 2.e-9).all()

    # check the lower band
    assert ((bb_df["BB_lower"] - talib_bb[2]).abs().fillna(0.) < 2.e-9).all()

    # test simple functions
    bb_middle = BBANDS_middle(ohlc['close'], period)
    assert ((bb_middle - talib_bb[1]).abs().fillna(0.) < 1.e-10).all()

    bb_upper = BBANDS_upper(ohlc['close'], period, std_multiplier)
    assert ((bb_upper - talib_bb[0]).abs().fillna(0.) < 2.e-9).all()

    bb_lower = BBANDS_lower(ohlc['close'], period, std_multiplier)
    assert ((bb_lower - talib_bb[2]).abs().fillna(0.) < 2.e-9).all()


def test_uo():
    """test UO"""

    uo = UO(ohlc["close"], ohlc["high"], ohlc["low"], ohlc["close"])
    talib_uo = talib.ULTOSC(ohlc["high"], ohlc["low"], ohlc["close"])

    assert ((uo - talib_uo).abs().fillna(0.) < 1.e-10).all()
