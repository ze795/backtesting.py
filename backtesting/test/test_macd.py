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

def MACD(close, n_fast=12, n_slow=26):
    emafast = pd.Series(close).rolling(n_fast).mean()
    emaslow = pd.Series(close).rolling(n_slow).mean()
    diff = pd.Series(emafast - emaslow)
    dea = pd.Series(diff).rolling(9).mean()
    macd = pd.Series(diff - dea) * 2
    return diff, dea, macd

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

class MacdCross(Strategy):
    def init(self):
        self.atr, self.tr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.diff, self.dea, self.macd = self.I(MACD, self.data.Close)
        self.m60 = self.I(SMA, self.data.Close, 60) # or EMA

    def next(self):
        if len(self.data.Close) < 30:
            return

        self.handle_pos()
        # self.handle_pos()
        # print( self.diff, self.dea, self.macd)
        # sys.exit()

        limit = None
        tp = None
        is_long = is_short = 1
        cur_p, index = self.data.Close[-1], len(self.data)
        cur_up = self.data.Close[-1] > self.data.Open[-1]
        cur_down = self.data.Close[-1] < self.data.Open[-1]
        # is_long = all(self.diff[-i] > 0 for i in range(1, 5))
        is_long = self.diff[-1] > 0
        # is_short = all(self.diff[-i] < 0 for i in range(1, 5))
        is_short = self.diff[-1] < 0
        # short_cond1 = all(self.data.Close[-i] < self.m60[-i] for i in range(1, 20))
        # long_cond1 = all(self.data.Close[-i] > self.m60[-i] for i in range(1, 20))
        if crossover(self.diff, self.dea) and is_long and cur_up and 1:
            stop_p = min(self.data.Low[-2:])
            # stop_p = cur_p - 1.0 * self.atr[-1]
            loss = cur_p - stop_p
            # tp = cur_p + 1 * loss
            self.buy(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p, loss))
        elif crossover(self.dea, self.diff) and is_short and cur_down and 1:
            stop_p = max(self.data.High[-2:])
            # stop_p = cur_p + 1.0 * self.atr[-1]
            # stop_p = max(self.data.High[-2:]) + 1
            loss = stop_p - cur_p
            # tp = cur_p - 1 * loss
            self.sell(size=1, sl=stop_p, limit=limit, tp=tp, tag=(index, cur_p, loss))

    def handle_pos(self):
        if not self.position:
            return
        # 2R baoben, 3R bao1R, 3~5R,baoyiban, 5Ryishang, 30%
        # past_max_profit = self.data.High - self.position.
        # Can't use index=-1 because self.atr is not an Indicator type
        cur_index = len(self.data) - 1
        for trade in self.trades:
            in_index, in_p, risk = trade.tag
            time_elapse = cur_index - in_index
            if trade.is_long:
                # if cur_index > in_index + 5:
                #     trade.sl = trade.entry_price + 1
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
                # if cur_index > in_index + 5:
                #     trade.sl = trade.entry_price - 1
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
    CY_1M = _read_file('RB_15m.csv')
    # CY_1M = _read_file('RB_15m.csv')[-10000:]
    # CY_1M = _read_file('eth_1m.csv')
    bt = Backtest(CY_1M, MacdCross, hedging=True)
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
