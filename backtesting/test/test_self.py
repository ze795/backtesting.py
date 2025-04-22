import inspect
import multiprocessing as mp
import os
import sys
import time
import unittest
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from glob import glob
from runpy import run_path
from tempfile import NamedTemporaryFile, gettempdir
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting import Backtest, Strategy
from backtesting._stats import compute_drawdown_duration_peaks
from backtesting._util import _Array, _as_str, _Indicator, patch, try_
from backtesting.lib import (
    FractionalBacktest, MultiBacktest, OHLCV_AGG,
    SignalStrategy,
    TrailingStrategy,
    barssince,
    compute_stats,
    cross,
    crossover,
    plot_heatmaps,
    quantile,
    random_ohlc_data,
    resample_apply,
)
from backtesting.test import BTCUSD, EURUSD, GOOG, SMA


SHORT_DATA = GOOG.iloc[:20]  # Short data for fast tests with no indicator lag


@contextmanager
def _tempfile():
    with NamedTemporaryFile(suffix='.html') as f:
        if sys.platform.startswith('win'):
            f.close()
        yield f.name


@contextmanager
def chdir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class BmaCross(Strategy):
    __atr = None

    def set_atr_periods(self, periods: int = 14):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def init(self):
        super().init()
        self.set_atr_periods()
        self.ma5 = self.I(SMA, self.data.Close, 5)
        self.ma10 = self.I(SMA, self.data.Close, 10)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma60 = self.I(SMA, self.data.Close, 60)

    def next(self):
        if (self.data.Close > self.ma5[-1] > self.ma10[-1] > self.ma20[-1]) and self.gather(self.ma60, self.ma10, self.ma20):
            # self.position.close()
            self.buy()
        elif (self.data.Close < self.ma5[-1] < self.ma10[-1] < self.ma20[-1]) and self.gather(self.ma60, self.ma10, self.ma20):
            # self.position.close()
            self.sell()
        elif self.position.is_long and crossover(self.ma20, self.ma5):
            self.position.close()
        elif self.position.is_short and crossover(self.ma5, self.ma20):
            self.position.close()

    def gather(self, a, b, c):
        # distance1 = abs(a[-2] - b[-2])
        # distance2 = abs(b[-2] - c[-2])
        # distance3 = abs(a[-2] - c[-2])
        distance1 = abs(a[-1] - b[-1])
        distance2 = abs(b[-1] - c[-1])
        distance3 = abs(a[-1] - c[-1])
        dist_ls = [distance1, distance2, distance3]
        gather = all(dist < self.__atr[-1] * 0.2 for dist in dist_ls)
        return gather

class SmaCross(Strategy):
    # NOTE: These values are also used on the website!
    fast = 10
    slow = 30

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

def _read_file(filename):
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)

if __name__ == '__main__':
    CY_1M = _read_file('CY_1M.csv')
    bt = Backtest(CY_1M, BmaCross, hedging=True)
    # bt = Backtest(EURUSD, SmaCross)
    # bt = Backtest(GOOG, SmaCross)
    # bt = Backtest(CY_1M, BmaCross, hedging=True)
    # bt = Backtest(BTCUSD, SmaCross)
    stats = bt.run()
    trades = stats['_trades']
    print(stats)
    # bt.plot()
    # pd.set_option('display.max_columns', 30)
    # trades.info()
    # print(type(trades))
    # print(repr(trades[:10]))
    # trades.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
    # a = {"a": 1, "aa": 2}
