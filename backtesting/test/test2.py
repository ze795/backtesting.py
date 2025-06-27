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



# from pr, vol, atr, duofangpao/ fengyunzaiqi(zuiduo 5k)
# shangshengsanfa

class FengYun(Strategy):
    def init(self):
        self.window_length = 5
        self.ma60 = self.I(SMA, self.data.Close, 60) # or EMA
        self.v5 = self.I(SMA, self.data.Volume, 5) # or EMA
        self.atr, self.tr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)

    def is_red(self, data_index):
        return self.data.Close[data_index] > self.data.Open[data_index]

    def filter_window(self):
        is_pass = True
        # 1 box atr vol
        first_index = -1 * self.window_length
        first_k_box = abs(self.data.Close[first_index] - self.data.Open[first_index])
        first_k_minmax = abs(self.data.High[first_index] - self.data.Low[first_index])
        first_k_vol = self.data.Volume[first_index]
        first_box_cond = first_k_box / first_k_minmax > 0.7
        first_atr_cond = self.tr[first_index] > 1.0 * self.atr[first_index]
        if first_box_cond and first_atr_cond and first_k_vol > 1.5 * self.v5[first_index-1]:
            is_pass = is_pass and True

        last_index = -1
        last_k_box = abs(self.data.Close[last_index] - self.data.Open[last_index])
        last_k_minmax = abs(self.data.High[last_index] - self.data.Low[last_index])
        last_k_vol = self.data.Volume[last_index]
        last_box_cond = last_k_box / last_k_minmax > 0.7
        last_atr_cond = self.tr[last_index] > 1.5 * self.atr[last_index]
        if last_box_cond and last_atr_cond and last_k_vol > 2 * self.v5[last_index-1]:
            is_pass = is_pass and True

        range_min, range_max = min(self.data.Low[first_index], self.data.Low[last_index]), \
            max(self.data.High[first_index], self.data.High[last_index])

        mid_max = max(self.data.High[first_index+1: last_index])
        mid_min = max(self.data.Low[first_index+1: last_index])
        mid_atr_pass = all(self.tr[first_index+1: last_index] < self.atr[first_index+1: last_index] * 0.9)
        mid_maxvol = max(self.data.Volume[first_index+1: last_index])
        if mid_max <= range_max and mid_min >= range_min and mid_atr_pass \
            and mid_maxvol < (first_k_vol*0.9) and mid_maxvol < (last_k_vol*0.9):
            is_pass = is_pass and True

        # 2 shangsheng sanfa
        if self.is_red(first_index) and self.is_red(last_index) and is_pass:
            return 1
        elif self.is_red(first_index) == 0 and self.is_red(last_index) == 0 and is_pass:
            return -1
        return 0

    def next(self):
        if len(self.data.Close) < 60:
            return

        # self.handle_pos()
        if self.position:
            return
        # 1 maline
        short_cond1 = all(self.data.Close[-i] < self.ma60[-i] for i in range(1, 5))
        long_cond1 = all(self.data.Close[-i] > self.ma60[-i] for i in range(1, 5))
        # get window and filter it
        # 2 first and last: box\atr\vol
        limit = None
        tp = None
        cur_p, index = self.data.Close[-1], self.data.index[-1]
        if self.filter_window() == 1:
            stop_p = self.data.Low[-1]
            # stop_p = cur_p - 1.0 * self.atr[-1]
            loss = cur_p - stop_p
            tp = cur_p + 1 * loss
            self.buy(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p, loss))
        elif self.filter_window() == -1:
            stop_p = self.data.High[-1]
            # stop_p = cur_p + 1.0 * self.atr[-1]
            # stop_p = max(self.data.High[-2:]) + 1
            index = self.data.index[-1]
            loss = stop_p - cur_p
            tp = cur_p - 1 * loss
            self.sell(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p, loss))

    def handle_pos(self):
        if not self.position:
            return
        # 2R baoben, 3R bao1R, 3~5R,baoyiban, 5Ryishang, 30%
        # past_max_profit = self.data.High - self.position.
        # Can't use index=-1 because self.atr is not an Indicator type
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                index, in_p, risk = trade.tag
                past_max_profit = self.data.High[-1] - in_p
                if past_max_profit >= 4 * risk:
                    new_sl = in_p + past_max_profit * 0.7
                    trade.sl = max(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 3 * risk:
                    new_sl = in_p + risk
                    trade.sl = max(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 2 * risk:
                    new_sl = in_p
                    trade.sl = max(trade.sl or -np.inf, new_sl)

            if trade.is_short:
                index, in_p, risk = trade.tag
                # print("handle: " + str(in_p) + " " + str(risk) + " " + str(self.data.Low[-1]))
                past_max_profit = in_p - self.data.Low[-1]
                if past_max_profit >= 4 * risk:
                    # print("into 4R")
                    new_sl = in_p - past_max_profit * 0.7
                    trade.sl = min(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 3 * risk:
                    # print("into 3R")
                    new_sl = in_p - risk
                    trade.sl = min(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 2 * risk:
                    # print("into 2R")
                    new_sl = in_p
                    trade.sl = min(trade.sl or -np.inf, new_sl)

class SingleK(Strategy):
    def init(self):
        self.window_length = 5
        self.p60 = self.I(SMA, self.data.Close, 60) # or EMA
        self.v5 = self.I(SMA, self.data.Volume, 5) # or EMA
        self.atr, self.tr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.timeloss = 30

    def handle_ops(self):
        cur_index = self.data.index[-1]
        for trade in self.trades:
            # if trade.is_long:
            if trade:
                index, in_p = trade.tag
                if (cur_index - index).days >= self.timeloss:
                    trade.close()

    def next(self):
        # yang, box, atr, vol, guadan
        # self.handle_ops()
        if self.position:
            return

        red = self.data.Close[-1] > self.data.Open[-1]
        box = abs(self.data.Close[-1] - self.data.Open[-1])
        minmax = abs(self.data.High[-1] - self.data.Low[-1])
        box_prop = box / minmax
        vol = self.data.Volume[-1]
        if (box_prop > 0.8) and (vol > self.v5[-1] * 1.5) and (self.tr[-1] > 1.5 * self.atr[-1]) and not self.position:
        # if not self.position:
            cur_p = self.data.Close[-1]
            limit = None
            if red:
                stop_p = min(self.data.Low[-2:]) - 1
                index = self.data.index[-1]
                loss = cur_p - stop_p
                tp = cur_p + 1 * loss
                self.buy(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p))
            else:
                stop_p = max(self.data.High[-2:]) + 1
                index = self.data.index[-1]
                loss = stop_p - cur_p
                tp = cur_p - 1 * loss
                self.sell(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p))


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
    CY_1M = _read_file('CY_1M.csv')
    bt = Backtest(CY_1M, FengYun, hedging=True)
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
