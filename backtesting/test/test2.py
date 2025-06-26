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

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

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



# from pr, vol, atr, duofangpao/ fengyunzaiqi(zuiduo 5k)
# shangshengsanfa

class FengYun(Strategy):
    def set_atr_periods(self, periods: int = 14):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr
        self.__cur_atr = tr

    def cur_atr(self, index=-1):
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        return tr[index]

    def init(self):
        self.window_length = 5
        self.p60 = self.I(SMA, self.data.Close, 60) # or EMA
        self.v5 = self.I(SMA, self.data.Volume, 5) # or EMA
        self.set_atr_periods()

    def filter_window(self):
        is_pass = True
        # 1 box atr vol
        first_index = -1 * self.window_length
        first_k_box = abs(self.data.Close[first_index] - self.data.Open[first_index])
        fist_k_minmax = abs(self.data.High[first_index] - self.data.Low[first_index])
        first_k_atr = self.cur_atr(first_index)
        first_k_vol = self.data.Volume[first_index]
        if (first_k_box / fist_k_minmax > 0.7) and first_k_atr > 0.9 * self.__atr[first_index] and first_k_vol > 1.5 * self.v5[first_index-1]:
            is_pass = is_pass and True

        last_index = -1
        last_k_box = abs(self.data.Close[last_index] - self.data.Open[last_index])
        last_k_minmax = abs(self.data.High[last_index] - self.data.Low[last_index])
        last_k_atr = self.cur_atr(last_index)
        last_k_vol = self.data.Volume[last_index]
        if (last_k_box / last_k_minmax > 0.7) and last_k_atr > 0.9 * self.__atr[last_index] and last_k_vol > 2 * self.v5[last_index-1]:
            is_pass = is_pass and True

        range_min, range_max = min(self.data.Low[first_index], self.data.Low[last_index]), \
            max(self.data.High[first_index], self.data.High[last_index])

        mid_max = max(self.data.High[first_index+1: last_index])
        mid_min = max(self.data.Low[first_index+1: last_index])
        mid_atr_pass = all(self.__atr[first_index+1: last_index] < self.__cur_atr[first_index+1: last_index])
        mid_maxvol = max(self.data.Volume[first_index+1: last_index])
        if mid_max <= range_max and mid_min >= range_min and mid_atr_pass \
            and mid_maxvol < (first_k_vol*0.9) and mid_maxvol < (last_k_vol*0.9):
            is_pass = is_pass and True

        pass

    def next(self):
        # 1 maline
        short_cond1 = all(self.data.Close[-i] < self.ma60[-i] for i in range(1, 5))
        long_cond1 = all(self.data.Close[-i] > self.ma60[-i] for i in range(1, 5))
        # get window and filter it
        # 2 first and last: box\atr\vol


        pass









def _read_file(filename):
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)

# 完整的系统
# zuixiao yongyu 60m, false signal less than 15m
#   分析模块，根据过去的关键point，峰谷排布，确定state=trend or range
# if range, use boll or wait breakout
# if trend, use feidao

if __name__ == '__main__':
    CY_1M = _read_file('RB0_dayK.csv')[:]
    # CY_1M = _read_file('CY_1M.csv')
    bt = Backtest(CY_1M, FeiDao, hedging=True)
    # bt = Backtest(BTCUSD, SmaCross)
    stats = bt.run()
    trades = stats['_trades']
    print(stats)
    day_curve = stats.loc['_equity_curve']
    print(day_curve)
    # bt.plot()
    # pd.set_option('display.max_columns', 30)

    # print(type(trades))
    # print(repr(trades[:10]))
    trades.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
    day_curve.to_excel('day_curve.xlsx', sheet_name='Sheet1', index=False)


# long ?
# 1. past N-k-c is on m20 [tr, can be identify by fg point]
# 2. past 4k, first 3k is continuous up; forth is pb or tm, atr > 0.6~0.8
# 3. if pbtm, in mid, if atr is small, wait next k
