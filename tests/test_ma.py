import os
import pandas as pd

import talib
import pandas_ta

from panti.moving_averages import SMA, EMA, DEMA, WMA, TEMA, TRIMA, KAMA, RMA, SMMA

data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/sample.json')

ohlc = pd.read_json(data_file).drop(columns=["time"]).reset_index(drop=True)


def test_sma():
    """test SMA"""

    ma = SMA(ohlc["close"], 14)
    talib_ma = talib.SMA(ohlc['close'], timeperiod=14)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_ema():
    """test EMA"""
    # ma2 = EMA2(ohlc["close"], 50)
    ma = EMA(ohlc["close"], 50)
    talib_ma = talib.EMA(ohlc['close'], timeperiod=50)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_dema():
    """test DEMA"""

    ma = DEMA(ohlc["close"], 20)
    talib_ma = talib.DEMA(ohlc['close'], timeperiod=20)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_wma():
    """test WVMA"""

    ma = WMA(ohlc["close"], period=20)
    talib_ma = talib.WMA(ohlc['close'], timeperiod=20)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_kama():
    """test KAMA"""

    ma = KAMA(ohlc["close"], period=30)
    talib_ma = talib.KAMA(ohlc['close'], timeperiod=30)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_tema():
    """test TEMA"""

    ma = TEMA(ohlc["close"], 50)
    talib_ma = talib.TEMA(ohlc['close'], timeperiod=50)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_trima():
    """test TRIMA"""

    ma = TRIMA(ohlc["close"], 19)
    talib_ma = talib.TRIMA(ohlc['close'], 19)

    assert ((ma - talib_ma).abs().fillna(0.) < 1.e-10).all()


def test_rma():
    """test RMA"""

    period = 19
    ma = RMA(ohlc["close"], period)
    pandas_ta_ma = pandas_ta.rma(ohlc["close"], period)

    # most financial technical analysis/indicators do not initialize the first term
    # of the ewm with SMA
    # more over they use the parameter adjust=True to adjust the beginning periods for imbalanced in relative weightings
    # i.e.
    rma_as_pandas_ta = ohlc["close"].ewm(alpha=1.0 / period, adjust=True).mean()
    assert ((rma_as_pandas_ta - pandas_ta_ma).abs().fillna(0.) < 1.e-10).all()

    # however the differences decay as we are using an ewm windowing
    # thus we compare the last part of the signal
    compare_idx = len(ma) // 2
    assert ((ma.iloc[compare_idx:] - pandas_ta_ma.iloc[compare_idx:]).abs().fillna(0.) < 1.e-4).all()
