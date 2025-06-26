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
    slow = 20
    # bigframe = 240

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.fast)
        self.sma2 = self.I(SMA, self.data.Close, self.slow)
        # self.sma3 = self.I(SMA, self.data.Close, self.bigframe)

        self.daily_close = resample_apply('D', None, self.data.Close, 60)

    @property
    def uping(self):
        if self.data.Close[-1] > self.daily_close[-1] and self.data.Close[-2] > self.daily_close[-2] and self.data.Close[-3] > self.daily_close[-3]:
            return 1
        elif self.data.Close[-1] < self.daily_close[-1] and self.data.Close[-2] < self.daily_close[-2] and self.data.Close[-3] < self.daily_close[-3]:
            return -1
        return 0
    
    def next(self):
        if crossover(self.sma1, self.sma2) and self.uping == 1:
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1) and self.uping == -1:
            self.position.close()
            self.sell()


# 这个策略的问题在于需要过滤震荡趋势，以及过滤掉大的止损的盈亏比较差的买点，以及浮盈后保本的问题等，以及跟踪止盈，单就买点来说是不错的。
# 所有中低频的趋势策略。就是要识别趋势，确定买点和止损，然后设置跟踪止盈。这里最难的是识别趋势
class KlinePatternStrategy(Strategy):
    def init(self):
        # 计算60日均线
        self.ma60 = self.I(SMA, self.data.Close, 60) # bt.indicators.SMA(self.data.close, period=60)

    def next(self):
        # 条件1：最近N1天的K线都在M60以下
        cond1 = all(self.data.Close[-i] < self.ma60[-i] for i in range(1, 5))
        
        # 条件2：最近N2天的收盘价满足特定关系
        if len(self.data.Close) < 5:  # 确保有足够的数据
            return
        
        day_n = 4
        c0 = self.data.Close[-5]    # 第一天收盘价
        c1 = self.data.Close[-4]    # 第一天收盘价
        c2 = self.data.Close[-3]  # 第二天收盘价
        c3 = self.data.Close[-2]  # 第三天收盘价
        c4 = self.data.Close[-1]  # 第四天收盘价
        
        cond2 = (c3 > c2 > c1 > c0) and (c4 < c3)
        
        if cond1 and cond2 and not self.position:
            # 设置止损和止盈价格
            cur = self.data.Close[-1]
            stop_price = max(self.data.High[-1], self.data.High[-2])
            loss = stop_price - cur 
            target = cur - 3 * loss
            self.sell(tp=target, sl=stop_price)

        
        # ---------------------------------------------------------------------------------------------
        # 多单
        # 条件1：最近N1天的K线都在M60以shang
        cond1 = all(self.data.Close[-i] > self.ma60[-i] for i in range(1, 5))
        
        # 条件2：最近N2天的收盘价满足特定关系
        if len(self.data.Close) < 5:  # 确保有足够的数据
            return
        
        c0 = self.data.Close[-5]    # 第一天收盘价
        c1 = self.data.Close[-4]    # 第一天收盘价
        c2 = self.data.Close[-3]  # 第二天收盘价
        c3 = self.data.Close[-2]  # 第三天收盘价
        c4 = self.data.Close[-1]  # 第四天收盘价
        
        cond2 = (c3 < c2 < c1 < c0) and (c4 > c3)
        
        if cond1 and cond2 and not self.position:
            # 设置止损和止盈价格
            cur = self.data.Close[-1]
            stop_price = min(self.data.Low[-1], self.data.Low[-2]) - 2
            loss = cur - stop_price
            target = cur + 2 * loss
            self.buy(tp=target, sl=stop_price)

class KlineDoubleTopStrategy(Strategy):
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
        # 计算60日均线
        self.ma60 = self.I(SMA, self.data.Close, 60) # bt.indicators.SMA(self.data.close, period=60)
        self.set_atr_periods()
        self.cur_atr = self.__atr[-1]
        self.daily_close = resample_apply('D', None, self.data.Close)
        self.daily_close_m60 = self.I(SMA, self.daily_close, 60)
        # self.daily_close_m60  = resample_apply('D', SMA, self.data.Close, 60)

    def identify_double_top(self, k_data):
        """
        识别双顶形态
        
        参数:
        k_data (pd.DataFrame): 包含K线数据的DataFrame，需包含['High', 'Low', 'Close']列
        atr (float): 平均真实波幅
        
        返回:
        tuple: (bool, int, int)，分别表示是否识别到双顶形态、左峰索引、颈线索引
        """
        
        cur_idx = len(k_data) - 1  # 当前K线索引
        x = 0
        atr = self.cur_atr

        # 遍历，先找出波峰K线，标准是最高点大于附近的两根K线， 然后再筛选这些K线左右两侧是否满足atr标准。如果用15根K线太多了，会错过机会。
        peak_indices = []
        for i in range(2, len(k_data) - 2):
            current_high = k_data.iloc[i]['High']
            prev1_high = k_data.iloc[i - 1]['High']
            prev2_high = k_data.iloc[i - 2]['High']
            next1_high = k_data.iloc[i + 1]['High']
            next2_high = k_data.iloc[i + 2]['High']

            past_k_num = min(5, i)
            past_max = k_data.iloc[0: i]['High'].max()
            mean_price = k_data.iloc[i - past_k_num: i]['Close'].mean()
            if current_high > prev1_high and current_high > prev2_high and \
            current_high > next1_high and current_high > next2_high and \
                current_high > past_max - 0.3 * atr: # 大于左侧最大值，留有一些余量用于假突破和压力位区间，大于右侧两根K线的最大值。
                peak_indices.append(i)


        for peak_idx in reversed(peak_indices): # 从最后一个波峰开始识别，最后一个能成立就直接返回了，不用再看前面的，前面的自有下一个滑动窗口去识别。
            peak_high = k_data.iloc[peak_idx]['High']
            
            # 计算波峰左侧的最低点（从起始到波峰前一根K线）
            left_min = k_data.iloc[:peak_idx]['Low'].min()
            
            # 计算波峰右侧的最低点（从波峰后一根K线到数据末尾）
            bottom = k_data.iloc[peak_idx + 1:]['Low'].min()
            bottom_idx = k_data.iloc[peak_idx + 1:]['Low'].idxmin()

            # 判断波峰是否满足左右的atr条件
            if (peak_high - left_min) > 2.5 * atr and (peak_high - bottom) > 2.5 * atr:
                peak = k_data.iloc[peak_idx]['High']

                # 检查波峰到当前K线内是否有收盘价大于kx_max + 2,  真突破，而非假突破
                if (k_data.iloc[peak_idx + 1: cur_idx + 1]['Close'] > peak + 2).any():
                    continue

                # 当前K线，至少回调到一半
                # if not (k_data.iloc[bottom_idx + 1: cur_idx + 1]['High'] > (peak + bottom) * 0.5).any():
                if not (k_data.iloc[cur_idx]['Close'] > (peak + bottom) * 0.5).any():
                    continue
                if not k_data.iloc[cur_idx]['Close'] > k_data.iloc[cur_idx]['Open'] or \
                    k_data.iloc[cur_idx]['Close'] > peak:
                    continue

                # 所有条件满足，返回True和索引
                # print("we find zuofeng", peak_idx, peak, left_min, bottom_idx, atr)
                # print("we find zuofeng", k_data.iloc[peak_idx]['Date'])
                # 到这里，它是一个及格的信号。但还有动量和持仓量成交量需要关注。
                # 下降的动量（下降期间的阴线实体面积）不能明显小于上涨的动量（上涨期间的阳线实体面积）[最重要，这代表形态]
                # 当前的持仓量，必须小于最高点的持仓量（下降期间可以是多方平仓或空头开仓，上涨期间绝不能是多方开仓，只能是空方获利平仓，因此持仓量不能超过前峰）
                # 波峰到当前期间，阳线的平均成交量不能明显大于阴线的平均成交量

                # 动量-实体面积，还是动量-所用时间？。先做面积 -------------- start --------------------
                subset = k_data.iloc[peak_idx:cur_idx+1].copy()

                # 计算每根K线的实体面积（开盘价与收盘价差值的绝对值）
                subset['BodyArea'] = (subset['Close'] - subset['Open']).abs()
                
                # 区分阳线和阴线（收盘价大于开盘价为阳线，反之为阴线）
                yang_candles = subset[subset['Close'] > subset['Open']]
                yin_candles = subset[subset['Close'] < subset['Open']]
                
                # 计算平均实体面积
                avg_yang_body = yang_candles['BodyArea'].mean() if not yang_candles.empty else 0
                avg_yin_body = yin_candles['BodyArea'].mean() if not yin_candles.empty else 0
                if not avg_yin_body > avg_yang_body * 0.8:
                    continue
                # 动量-实体面积，还是动量-所用时间？。先做面积 -------------- end --------------------

                # 增加阴线进场条件？
                return True, peak_high
        return False, None

    def next(self):
        if len(self.data.df) < 51:  # 确保有足够的数据
            return
        # try:
        # print(self.data.df)
        # except:
            # pass
        # sys.exit()
        # if not self.daily_close[-1] < self.daily_close_m60[-1] :
        #     return
        if not self.data.Close[-1] > self.ma60[-1]:
            return

        kline_data = self.data.df.iloc[-50:].copy()
        is_db, peak = self.identify_double_top(kline_data)
        cur = self.data.Close[-1]
        
        if is_db and (0 < (peak - cur) <= 50) and not self.position:
            # 设置止损和止盈价格
            stop_price = max(peak, cur) + 2
            loss = stop_price - cur 
            target = cur - 1 * loss
            self.sell(tp=target, sl=stop_price)


def _read_file(filename):
    from os.path import dirname, join
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)

if __name__ == '__main__':
    CY_1M = _read_file('/home/ze/repo/qt/data/RB0_dayK.csv')[:]
    # print(CY_1M[:10])
    # exit()
    # bt = Backtest(CY_1M, SmaCross, hedging=True)
    # bt = Backtest(EURUSD, SmaCross)
    bt = Backtest(CY_1M, KlinePatternStrategy, trade_on_close=TimeoutError)
    # bt = Backtest(CY_1M, BmaCross, hedging=True)
    # bt = Backtest(BTCUSD, SmaCross)
    stats = bt.run()
    trades = stats['_trades']
    print(stats)
    bt.plot(filename='output.html', reverse_indicators=True)
    # pd.set_option('display.max_columns', 30)
    # trades.info()
    # print(type(trades))
    # print(repr(trades[:10]))
    # trades.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
    # a = {"a": 1, "aa": 2}
