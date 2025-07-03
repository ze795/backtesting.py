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
import warnings
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

def ATR(High, Low, Close, periods: int) -> pd.Series:
    hi, lo, c_prev = High, Low, pd.Series(Close).shift(1)
    tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
    atr = pd.Series(tr).rolling(periods).mean()
    """
    Returns `n`-period simple moving average of array `arr`.
    """

    return atr, tr

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


def BBANDS(High, Low, Close, n_lookback=20, n_std=2) -> pd.Series:
    hlc3 = pd.Series((High + Low + Close) / 3)
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return mean, upper, lower

class Boll_TL(Strategy):
    def init(self):
        self.mean, self.upper, self.lower = self.I(BBANDS, self.data.High, self.data.Low, self.data.Close)
        self.atr, self.tr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)

        # 5K变化多少算小呢？作为一个参数要搜索。5K内的abs(max-min)
        self.flat_threshold = 2 # 过去5K如果角度很小，就随机上下开单。如果角度很大，要等到角度变小再开。
        self.flat_window = 5 # 5~10


    def flat(self):
        past_max = max(self.mean[-self.flat_window:])
        past_min = min(self.mean[-self.flat_window:])
        range_max = abs(past_max - past_min) # 窗口内波动太大不做
        angle_max = abs(self.mean[-self.flat_window] - self.mean[-1]) # 前后变化太大不做，说明走了单边突破了
        if range_max < self.flat_threshold:
            if self.mean[-self.flat_window] > self.mean[-1]:
                # print("flat:1 ", self.mean[-self.flat_window], self.mean[-1], range_max)
                return -1
            else:
                # print("flat: -1 ", self.mean[-self.flat_window], self.mean[-1], range_max)
                return 1

        return 0

    def next(self):
        if len(self.data) < 5:
            return
        # self.handle_ops()
        limit = None
        cur_mean = self.mean[-1]
        cur_lower = self.lower[-1]
        cur_upper = self.upper[-1]
        cur_index = self.data.index[-1]

        if self.position:
            # if we close pos?
            cur_index = self.data.index[-1]
            for trade in self.trades:
                # if trade.is_long:
                if trade:
                    if trade.is_long and cur_upper > trade.entry_price:
                        trade.tp = cur_upper
                        # print(0, 1,  cur_upper, trade.entry_price)
                    elif trade.is_short and cur_lower < trade.entry_price:
                        trade.tp = cur_lower
                        # print(0, -1, cur_upper, trade.entry_price)

        if self.position:
            return
        direction = self.flat()
        if direction == 1:
            # limit = cur_mean
            limit = cur_lower
            stop_p = limit - 1.0 * (cur_mean - cur_lower)
            self.buy(size=1, sl=stop_p, limit=limit, tag=(cur_index, 0))
            # print(1, cur_upper, cur_mean, cur_lower, cur_index, stop_p)
        elif direction == -1:
            # limit = cur_mean
            limit = cur_upper
            stop_p = limit + 1.0 *  (cur_upper - cur_mean)
            self.sell(size=1, sl=stop_p, limit=limit, tag=(cur_index, 0))
            # print(-1, cur_upper, cur_mean, cur_lower, cur_index, stop_p)
        else:
            return


def _read_file(filename):
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)

# 完整的系统
# zuixiao yongyu 60m, false signal less than 15m
#   分析模块，根据过去的关键point，峰谷排布，确定state=trend or range
# if range, use boll or wait breakout
# if trend, use feidao

def correct_diff_csv(data):
    df = pd.DataFrame(data)
    adjusted_values = df + 100
    return adjusted_values

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    # CY_1M = _read_file('RB0_dayK.csv')[:]
    CY_1M = _read_file('CY_1M.csv')[-10000:]
    # CY_1M = _read_file('SP2509-2511_5m.csv')[-100:]
    # print(CY_1M)
    # CY_1M = correct_diff_csv(CY_1M)[-1000:]
    # print(CY_1M)

    bt = Backtest(CY_1M, Boll_TL, hedging=True)
    # bt = Backtest(BTCUSD, SmaCross)
    stats = bt.run()
    trades = stats['_trades']
    print(stats)
    day_curve = stats.loc['_equity_curve']
    print(day_curve)
    bt.plot(open_browser=False)
    # pd.set_option('display.max_columns', 30)

    # print(type(trades))
    # print(repr(trades[:10]))
    trades.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
    day_curve.to_excel('day_curve.xlsx', sheet_name='Sheet1', index=False)


# long ?
# 1. past N-k-c is on m20 [tr, can be identify by fg point]
# 2. past 4k, first 3k is continuous up; forth is pb or tm, atr > 0.6~0.8
# 3. if pbtm, in mid, if atr is small, wait next k
