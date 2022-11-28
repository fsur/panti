__version__ = "0.1"

import types
from functools import wraps
from typing import List

import pandas as pd

import panti.moving_averages as pti_ma
import panti.indicators as pti_i
import panti.candles as pti_cdl
import panti.experimental as pti_exp

ma_dict = {attr: val for attr, val in pti_ma.__dict__.items() if
           (isinstance(val, types.FunctionType) and (attr.lower().endswith("ma"))
            and not (attr.endswith("_")) and not (attr.startswith("_")))}

indicators_dict = {attr: val for attr, val in pti_i.__dict__.items() if
                   (isinstance(val, types.FunctionType) and not (attr.endswith("_"))
                    and not (attr.startswith("_")))}

cdl_dict = {attr: val for attr, val in pti_cdl.__dict__.items() if
            (isinstance(val, types.FunctionType) and not (attr.endswith("_"))
             and not (attr.startswith("_")))}

# register functions of each sub-module in the main module
ser_pti_dict = {**ma_dict, **indicators_dict}
pti_dict = {**ser_pti_dict, **cdl_dict}

exp_dict = {attr: val for attr, val in pti_exp.__dict__.items() if
            (isinstance(val, types.FunctionType) and (attr not in ma_dict.keys()) and
             (attr not in indicators_dict.keys()) and (attr not in cdl_dict.keys()))}

globals().update(pti_dict)
globals().update(exp_dict)


def get_indicators() -> List:
    """
    Gives all the indicators available in the PANTI library.

    :return: list of indocators of panti library
    """
    return sorted(list(set(pti_dict.keys())))


# this function can be either used as decorator to decorate a function directly
# or in the __new__ method of a class to register a function as a method
def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self._obj, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator


class MetaExperimental(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        for attr, val in exp_dict.items():
            if isinstance(val, types.FunctionType):
                add_method(cls)(val)
        return cls


class PantiExp(metaclass=MetaExperimental):
    __version__ = __version__

    def __init__(self, pandas_obj):
        self._obj = pandas_obj


class MetaPantiSer(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        # attach validated moving averages/indicators
        for attr, val in ser_pti_dict.items():
            if isinstance(val, types.FunctionType):
                add_method(cls)(val)
        return cls


@pd.api.extensions.register_series_accessor("pti")
class PantiSer(metaclass=MetaPantiSer):
    __version__ = __version__

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.exp = PantiExp(pandas_obj)
