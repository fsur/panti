import os
import types
import pandas as pd

import panti as pti
import panti.indicators as pti_i


def test_register_dicts():
    def _checker(reg_dict, submodule):
        for func in reg_dict.values():
            assert (func.__name__ in submodule.__dict__.keys())
            assert (func == submodule.__dict__[func.__name__])

    # validated dicts
    _checker(pti.ma_dict, pti)
    _checker(pti.indicators_dict, pti)
    _checker(pti.cdl_dict, pti)
    # experimental dicts
    _checker(pti.exp_dict, pti)

    # global dicts
    _checker(pti.pti_dict, pti)


def test_module_accessors():
    def _checker(reg_dict):
        for func in reg_dict.values():
            assert (func.__name__ in pti.__dict__.keys())
            assert (func == pti.__dict__[func.__name__])

    _checker(pti.ma_dict)
    _checker(pti.indicators_dict)
    _checker(pti.cdl_dict)

    _checker(pti.pti_dict)


def test_series_accessors():
    ser = pd.Series([1., 2.])

    # check that panti attribute is attached to a pd.Series
    assert (getattr(ser, "pti", None) is not None)

    # check that all moving_averages are attached
    for val in pti.ma_dict.items():
        if isinstance(val, types.FunctionType):
            assert (getattr(ser.pti, val.__name__, None) is not None)

    # check that all indicators are attached
    for val in pti.indicators_dict.items():
        if isinstance(val, types.FunctionType):
            assert (getattr(ser.pti, val.__name__, None) is not None)

    # check that all indicators are attached
    for val in pti.exp_dict.values():
        if isinstance(val, types.FunctionType):
            assert (getattr(ser.pti.exp, val.__name__, None) is not None)


def test_accessor_values():
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sample.json")
    df = pd.read_json(data_file)

    assert ((df["close"].pti.SMA() - pti_i.SMA(df["close"])).fillna(0.) < 1.e-12).all()
