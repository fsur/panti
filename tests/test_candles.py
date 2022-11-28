import os
import pandas as pd

import talib

from panti.moving_averages import RMA
from panti.candles import TR, ATR, WILLIAMS, MFI

data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/sample.json')

ohlc = pd.read_json(data_file).drop(columns=["time"]).reset_index(drop=True)


def test_tr():
    """test TR"""

    tr = TR(ohlc["high"], ohlc["low"], ohlc["close"])
    talib_tr = talib.TRANGE(ohlc['high'], ohlc['low'], ohlc['close'])

    assert ((tr - talib_tr).abs().fillna(0.) < 1.e-10).all()


def test_atr():
    """test ATR"""
    talib_tr = talib.ATR(ohlc['high'], ohlc['low'], ohlc['close'],
                         timeperiod=14)

    # TALIB is bugged when computing the fast ATR:
    #   in TALIB uses the Wilder's Moving Average with TR (true range) time series beginning from index 1 instead of 0.
    #   i.e. atr_as_talib = ATR(ser[1:], period=14, avg_method="rma")
    # As RMA is decaying with time, we must compare the last part of the signal (we compare here from the 3/5
    # to the end)
    atr_as_talib = RMA(TR(ohlc['high'], ohlc['low'], ohlc['close'])[1:], period=14)
    assert ((atr_as_talib - talib_tr).abs().fillna(0.) < 1.e-10).all()

    tr = ATR(ohlc['high'], ohlc['low'], ohlc['close'], period=14, avg_method="rma")
    compare_idx = 3 * len(tr) // 5
    assert ((tr.iloc[compare_idx:] - talib_tr.iloc[compare_idx:]).abs().fillna(0.) < 1.e-10).all()


def test_mfi():
    """test MFI"""

    mfi = MFI(ohlc['high'], ohlc['low'], ohlc['close'], ohlc['volume'], 9)
    talib_mfi = talib.MFI(ohlc['high'], ohlc['low'], ohlc['close'], ohlc['volume'], 9)

    assert ((mfi - talib_mfi).abs().fillna(0.) < 1.e-10).all()


def test_williams():
    """test WILLIAMS"""

    will = WILLIAMS(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    talib_will = talib.WILLR(ohlc["high"], ohlc["low"], ohlc["close"], 14)

    assert ((will - talib_will).abs().fillna(0.) < 1.e-10).all()
