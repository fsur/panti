# PANTI (Pandas Technical Indicators)  
  
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  
[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/download/releases/3.9.0/)  
<!---  
[![Build Status](https://travis-ci.org/peerchemist/finta.svg?branch=master)](https://travis-ci.org/peerchemist/finta)]  
-->  
[![Bitcoin Donate](https://badgen.net/badge/Bitcoin/Donate/F19537?icon=bitcoin)](https://www.blockchain.com/btc/address/187KPmoxZwE8QBDRjGyYuxCjN26uUE9M7o)  
[![Ethereum Donate](https://badgen.net/badge/Ethereum/Donate/F19537?icon=https://cryptologos.cc/logos/ethereum-eth-logo.svg)](https://www.blockchain.com/eth/address/0xBc5Ad57112E98b052076D11C7d92a6F9CA60843d)  
[![USDT Donate](https://badgen.net/badge/USDT/Donate/F19537?icon=https://cryptologos.cc/logos/tether-usdt-logo.svg)](https://www.blockchain.com/eth/address/0xbc5ad57112e98b052076d11c7d92a6f9ca60843d)  
  
Common financial technical indicators implemented in Pandas.  
  
**PANTI** is designed to help quantitative developers easily use and access technical financial indicators. 

**PANTI** was built on the premise that the current libraries for calculating technical financial indicators ([pandas_ta](https://github.com/twopirllc/pandas-ta), [finta](https://github.com/peerchemist/finta), [stockstats](https://github.com/jealous/stockstats)) are either too cumbersome in their implementation, losing any user-friendliness, or go beyond the simple scope of calculating technical indicators, or are unreliable because they are poorly tested.
  
**PANTI** wants to take the better of all worlds : easy to use, comprehensible and fully tested with regards to the reference technical analysis library since 1999 : [TALIB](https://www.ta-lib.org/)

To do so, **PANTI** is built on a set of functions that can be used directly, but are also attached to a `pd.Series`:
- *Functions* mode
```python  
import pandas as pd  
import panti as pti
  
ser = pd.Series(...)  
sma = pti.SMA(ser, ...)  
```  
  
- `pd.Series` accessor: **PANTI** adds the **`pti`** module to a `pd.Series`, accessible through the `ser.pti` accessor.  
  
```python  
import pandas as pd  
import panti
  
ser = pd.Series(...)  
sma = ser.pti.SMA(...)  
```  

The complete list of indicators is available via the `get_indicators()` function.  

```python  
import panti as pti

pti.get_indicators()
>> ['ATR', 'BBANDS_df', 'BBANDS_lower', 'BBANDS_middle', 'BBANDS_upper', 'DEMA', 'EMA', 'ER', 'KAMA', 'MACD', 'MACD_df', 'MACD_hist', 'MACD_signal', 'MFI', 'MOM', 'RMA', 'ROC', 'RSI', 'SMA', 'SMMA', 'TEMA', 'TP', 'TR', 'TRIMA', 'TRIX', 'UO', 'WILLIAMS', 'WMA', ...]

```  

------------------------------------------------------------------------
## Supported indicators:  
  
**PANTI** supports a lot of indicators :

### Moving averages :
> - Double Exponential Moving Average 'DEMA'
> - Exponential Moving Average 'EMA'
> - Kaufman's Adaptive Moving Average 'KAMA'
> - WildeR's Moving Average 'RMA'
> - Simple Moving Average 'SMA'
> - Smoothed Moving Average 'SMMA' (equivalent to RMA)
> - Triple Exponential Moving Average 'TEMA'
> - Triangular Moving Average 'TRIMA'
> - Weighted Moving Average 'WMA'

### Indicators :
 > - Bollinger Bands 'BBANDS'
 > - Kaufman Efficiency Indicator 'ER'
 > - Moving Average Convergence Divergence 'MACD'
 > - Market Momentum 'MOM'
 > - Rate-of-Change 'ROC'
 > - Relative Strenght Index 'RSI'
 > - Triple Exponential Moving Average Oscillator 'TRIX'
 > - Ultimate Oscillator 'UO'

### Candles :
 > - Average True Range 'ATR'
 > - Money Flow Index 'MFI'
 > - Typical Price 'TP'
 > - True Range 'TR'
 > - Williams %R 'WILLIAMS'

------------------------------------------------------------------------
## Testing
All these indicators are fully tested with regards to the reference technical analysis librairy since 1999 : [TALIB](https://www.ta-lib.org/).
If one indicator is not present un [TALIB](https://www.ta-lib.org/), other Python technical analysis libraries are used ([pandas_ta](https://github.com/twopirllc/pandas-ta), [finta](https://github.com/peerchemist/finta), [stockstats](https://github.com/jealous/stockstats)) with specific attention on the coherency with [TALIB](https://www.ta-lib.org/) philosophy and implementation.

You can check that all provided signals is **PANTI** are well tested by having a glance at the *test* folder.
To run tests you will need [pytest](https://docs.pytest.org/en/7.2.x/) dependency as well as Python wrapper of [TALIB](https://www.ta-lib.org/): [TA-LIB](https://pypi.org/project/TA-Lib/).
Depdencies to [pandas_ta](https://github.com/twopirllc/pandas-ta), [finta](https://github.com/peerchemist/finta) or [stockstats](https://github.com/jealous/stockstats) may be necessary depending on the presence of the testing indicator in [TALIB](https://www.ta-lib.org/) or not.

------------------------------------------------------------------------
## Experimental (not tested yet) moving averages / indicators / candles
**PANTI** being constantly under improvement, a lot of indicators have been implemented but not fully tested yet.

For these, some inaccuracies may occur if compared directly to [TALIB](https://www.ta-lib.org/) (especially on initialization when working with moving windows).

The `experiment.py` file contains in bulk all moving averages / series based indicators / candle base indicators.
Once one moving average / indicator function has its test in `test_ma.py` or `test_indicators.py` or `test_candles.py`, it is moved to the corresponding `moving_everages.py`, `indicators.py` or `candles.py` file and that's it ! The function is automatically registered by **PANTI** as new moving average / indicator / candle indicator and accessible from a `pd.Series`.

> - Accumulation-Distribution Line 'ADL'
> - Average Directional Index 'ADX'
> - Awesome Oscillator 'AO'
> - Adaptive Price Zone 'APZ'
> - Normalized BASP 'BASPN'
> - Buy and Sell Pressure 'BASP'
> - Bollinger Bands Width 'BBWIDTH'
> - Balance Of Power 'BOP'
> - Commodity Channel Index 'CCI'
> - Cummulative Force Index 'CFI'
> - Chaikin Oscillator 'CHAIKIN'
> - Chandelier Exit 'CHANDELIER'
> - Chande Momentum Oscillator 'CMO'
> - Coppock Curve 'COPP'
> - Directional Movement Indicator 'DMI'
> - Directional Movement 'DM'
> - Donchian Channel 'DO'
> - Dynamic Momentum Index 'DYMI'
> - Bull power and Bear Power 'EBBP'
> - Elder's Force Index 'EFI'
> - Ease of Movement 'EMV'
> - Schaff Trend Cycle using EV_MACD 'EVSTC'
> - Elastic Volume Moving Average 'EVWMA'
> - Elastic-Volume weighted MACD 'EV_MACD'
> - Fisher Transform 'FISH'
> - Fractal Adaptive Moving Average 'FRAMA'
> - Finite Volume Element 'FVE'
> - Hull Moving Average 'HMA'
> - Ichimoku Cloud 'ICHIMOKU'
> - Inverse Fisher Transform RSI 'IFT_RSI'
> - Keltner Channels 'KC'
> - Know Sure Thing 'KST'
> - Mass Index 'MI'
> - Momentum Breakout Bands 'MOBO'
> - Moving Standard deviation 'MSD'
> - On Balance Volume 'OBV'
> - Percent B 'PERCENT_B'
> - Pivot Points 'PIVOT'
> - Fibonacci Pivot Points 'PIVOT_FIB'
> - Percentage Price Oscillator 'PPO'
> - Parabolic SAR 'PSAR'
> - Price Zone Oscillator 'PZO'
> - Qstick 'QSTICK'
> - Stop-and-Reverse 'SAR'
> - Squeeze Momentum Indicator 'SQZMI'
> - Schaff Trend Cycle 'STC'
> - Stochastic Oscillator %K 'STOCH'
> - Stochastic oscillator %D 'STOCHD'
> - Stochastic RSI 'STOCHRSI'
> - Twiggs Money Index 'TMF'
> - True Strength Index 'TSI'
> - Volume Adjusted Moving Average 'VAMA'
> - Volatility-Based-Momentum 'VBM'
> - Value Chart 'VC'
> - Volume Flow Indicator 'VFI'
> - Vortex Indicator 'VORTEX'
> - Volume Price Trend 'VPT'
> - Volume Weighted Average Price 'VWAP'
> - Volume-Weighted MACD 'VW_MACD'
> - Volume Zone Oscillator 'VZO'
> - Mark Whistler's WAVE PM 'WAVEPM'
> - 'WILLIAMS_FRACTAL',
> - Weighter OBV 'WOBV'
> - Wave Trend Oscillator 'WTO'
> - Zero Lag Exponential Moving Average 'ZLEMA'

------------------------------------------------------------------------
## Dependencies:  
  
### Library  
  
- python (3.10+)  
- pandas (1.5.0+)
- nmupy (1.23.3+)
  
### Tests  
  
- [pytest](https://pypi.org/project/pytest/) (7.2.0+)
- [ta-lib](https://pypi.org/project/TA-Lib/) (0.4.25+): Python ta-lib is only a binding of the [C library TALIB](https://www.ta-lib.org/). To use the indicators from Python you will need to compile the C sources. The compilation process for any platform is very well explained [here](https://pypi.org/project/TA-Lib/). If compilation seems too complicated, unofficial pre-built binaries are downloadable [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) (not recommanded).
- [pandas_ta](https://pypi.org/project/pandas-ta/) (0.3.14+)
  
**PANTI** functions are very well documented and there should be no trouble exploring it and using with your data.  
Candles functions in `candles.py` use series with specific meaning : high, low, close, open or open price.  

------------------------------------------------------------------------
## Install
  
Latest development version:  
  
`pip install git+git://github.com/fsur/panti.git`  

------------------------------------------------------------------------
## Usage  
  
Use **PANTI** with functions:  
```python  
import pandas as pd
import panti as pti

ser = pd.Series(...)
pti.SMA(ser, period=41) # Simple Moving Average
pti.EMA(ser, period=9) # Exponential Moving Average
pti.MACD(ser, period_fast=12, period_slow=26) # MACD

high = pd.Series(...)
low = pd.Series(...)
close = pd.Series(...)
pti.TR(high, low, close) # True Range
```  
  
Use **PANTI** with `pd.Series` accessors:  
```python  
import pandas as pd
import panti as pti
  
ser = pd.Series(...)
ser.pti.SMA(period=41) # Simple Moving Average
ser.pti.EMA(period=9) # Exponential Moving Average
ser.pti.MACD(period_fast=12, period_slow=26) # MACD
```  
Candles functions cannot be attached to `pd.Series` as they are not a mathematical transform of a single time serie but a combination of several with specific meanings, i.e. `pti.TR(high, low, close)`.
  
------------------------------------------------------------------------  
  
## Contributing
I welcome pull requests with new indicators or fixes for existing ones.  
Please submit only indicators that belong in public domain and are  
royalty free.

**PANTI** file structure is very simple and follows the following pattern:
```
.
├── panti
│   ├── __init__.py
│   ├── candles.py
│   ├── experimental.py
│   ├── indicators.py
│   └── moving_averages.py
├── tests
│   ├── data    # folder that contains sample data used in tests
│   ├── __init__.py
│   ├── test_candles.py
│   ├── test_indicators.py
│   └── test_ma.py
```


To leave your contribution, please follow the following procedure:
1. Fork it (https://github.com/fsur/panti/fork)  
2. Study how it's implemented.  
3. Create your feature branch (`git checkout -b my-new-feature`).  
4. Implement your new moving average / indicator / candle indicator in `experimental.py`.
5. If you have a test associated with, implement it in the corresponding file `test_ma.py`, `test_indicators.py` or `test_candles.py`.
6. Run your favorite code formatter on each file to ensure uniform code style. Recommended to  
   use [black](https://github.com/ambv/black).  
5. Commit your changes (`git commit -am 'Add some feature'`).  
6. Push to the branch (`git push origin my-new-feature`).  
7. Create a new Pull Request.
8. I will check the implementation and test before any final merge.