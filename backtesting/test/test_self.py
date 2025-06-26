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


class FeiDao(Strategy):
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

    def cur_atr(self):
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        return tr[-1]
    # def init(self):
    #     super().init()
    #     self.set_atr_periods()
    #     # self.ma5 = self.I(SMA, self.data.Close, 5)
    #     # self.ma10 = self.I(SMA, self.data.Close, 10)
    #     self.ma20 = self.I(SMA, self.data.Close, 20)
    #     self.ma60 = self.I(SMA, self.data.Close, 60)

    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma60 = self.I(SMA, self.data.Close, 60)
        self.set_atr_periods()

    def next(self):
        if len(self.data.Close) < 5:  # 确保有足够的数据
            return

        self.handle_pos()
        if self.position:
            return
        # 必须条件：atr > 0.6
        if self.cur_atr() < 0.6 * self.__atr[-1]:
            return

        day_n = 4

        # 条件1：最近N1天的K线都在M60以下
        short_cond1 = all(self.data.Close[-i] < self.ma60[-i] for i in range(1, 5))
        # 条件2：最近N2天的收盘价满足特定关系

        c0, c1, c2, c3, c4 = self.data.Close[-5], self.data.Close[-4], \
            self.data.Close[-3], self.data.Close[-2], self.data.Close[-1]
        cur = self.data.Close[-1]

        if day_n == 4:
            short_cond2 = (c3 > c2 > c1 > c0) and (c4 < c3)
        elif day_n == 3:
            short_cond2 = (c3 > c2 > c1) and (c4 < c3)

        if short_cond1 and short_cond2 and not self.position:
            # 设置止损和止盈价格
            stop_price = max(self.data.High[-1], self.data.High[-2]) + 2
            limit = None
            if (stop_price - cur) > 1.5 * self.__atr[-1]:
                # limit it
                limit = cur + (self.data.High[-1] - self.data.Close[-1]) * 0.4
                cur = limit
            loss = stop_price - cur # risk
            target = cur - 3 * loss
            # self.sell(tp=target, sl=stop_price, tag=(cur, loss))
            # print("save: "+ str(cur) + " " + str(loss) + " " + str(target))
            self.sell(sl=stop_price, limit=limit, tag=(cur, loss))

        # ---------------------------------------------------------------------------------------------
        # 多单
        long_cond1 = all(self.data.Close[-i] > self.ma60[-i] for i in range(1, 5))
        if day_n == 4:
            long_cond2 = (c3 < c2 < c1 < c0) and (c4 > c3)
        elif day_n == 3:
            long_cond2 = (c3 < c2 < c1) and (c4 > c3)

        if long_cond1 and long_cond2 and not self.position:
            # 设置止损和止盈价格
            limit = None
            stop_price = min(self.data.Low[-1], self.data.Low[-2]) - 2
            if (cur - stop_price) > 1.5 * self.__atr[-1]:
                # limit it
                limit = cur - (self.data.Close[-1] - self.data.Low[-1]) * 0.4
                cur = limit
            loss = cur - stop_price
            target = cur + 2 * loss
            # self.buy(sl=stop_price, tp=target)
            self.buy(sl=stop_price, limit=limit, tag=(cur, loss))

    def handle_pos(self):
        if not self.position:
            return
        # 2R baoben, 3R bao1R, 3~5R,baoyiban, 5Ryishang, 30%
        # past_max_profit = self.data.High - self.position.
        # Can't use index=-1 because self.__atr is not an Indicator type
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                in_p, risk = trade.tag
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
                in_p, risk = trade.tag
                # print("handle: " + str(in_p) + " " + str(risk) + " " + str(self.data.Low[-1]))
                past_max_profit = in_p - self.data.Low[-1]
                if past_max_profit >= 5 * risk:
                    # print("into 4R")
                    new_sl = in_p - past_max_profit * 0.7
                    trade.sl = min(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 3 * risk:
                    # print("into 3R")
                    new_sl = in_p - risk
                    trade.sl = min(trade.sl or -np.inf, new_sl)
                elif past_max_profit >= 2 * risk:
                    print("into 2R")
                    new_sl = in_p
                    trade.sl = min(trade.sl or -np.inf, new_sl)


            # self.buy()


class SmaCross(Strategy):
    # NOTE: These values are also used on the website!
    fast = 10
    slow = 30

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        self.handle_pos()
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy(tag=(self.data.Close[-1], 10))
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell(tag=(self.data.Close[-1], 10))








def _read_file(filename):
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)

if __name__ == '__main__':
    CY_1M = _read_file('RB0_dayK.csv')[-1000:]
    # CY_1M = _read_file('CY_1M.csv')
    bt = Backtest(CY_1M, FeiDao, hedging=True)
    # bt = Backtest(BTCUSD, SmaCross)
    stats = bt.run()
    trades = stats['_trades']
    print(stats)
    # bt.plot()
    # pd.set_option('display.max_columns', 30)

    # print(type(trades))
    # print(repr(trades[:10]))
    trades.to_excel('output.xlsx', sheet_name='Sheet1', index=False)


# long ?
# 1. past N-k-c is on m20 [tr, can be identify by fg point]
# 2. past 4k, first 3k is continuous up; forth is pb or tm, atr > 0.6~0.8
# 3. if pbtm, in mid, if atr is small, wait next k
