import sys
import time
import threading
import random
import datetime
import pandas as pd
import akshare as ak

from typing import List, Dict, Tuple, Optional

if __name__ == '__main__':
  df1 = ak.futures_zh_minute_sina(symbol="SP0", period="1")
  df2 = ak.futures_zh_minute_sina(symbol="SP2511", period="5")
  df1.to_csv('SP0_1m.csv')
  # print(futures_zh_minute_sina_df)

#   print(df1)

#   assert df1.index.equals(df2.index), "Indexes do not match"
#   assert df1.columns.equals(df2.columns), "Columns do not match"

#   # 作差
# # 抽取数值部分并转换为浮点数
#   df1_values = df1.drop(columns='datetime').astype(float)
#   df2_values = df2.drop(columns='datetime').astype(float)

#   # 进行作差运算
#   df_diff_values = df1_values - df2_values + 100

#   # 将日期列添加回结果中
#   df_diff = pd.concat([df1['datetime'], df_diff_values], axis=1)

#   # 作差
#   # df_diff = df1_float - df2_float

#   # 打印结果
#   print("Difference DataFrame:")
#   print(df_diff)

#   # 保存结果到 CSV 文件
#   df_diff.to_csv('difference.csv')
  # futures_zh_spot_df = ak.futures_zh_spot(symbol='RB0', market="CF", adjust='0')
  # print(futures_zh_spot_df)

# futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="RB0", period="15")
# print(futures_zh_minute_sina_df)

# futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="RB0", period="60")
# print(futures_zh_minute_sina_df)
