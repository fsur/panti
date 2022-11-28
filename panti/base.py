# this modules is used to avoid cyclic import when a moving average has to use an indicator to be computed
# functions present here are not exported as is in panti
# there names have to finish with an underscore "_" (see function ER_ in base.py and ER in indicators for example)
import pandas as pd


def ER_(ser: pd.Series, period: int = 10) -> pd.Series:
    """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100,
    where zero is the center point.
     +100 is upward forex trending market and -100 is downwards trending markets."""

    change = ser.diff(period).abs()
    volatility = ser.diff().abs().rolling(window=period).sum()

    return change / volatility
