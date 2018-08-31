# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from CommonAPI.base import Base

class RawData(object):
    def __init__(self, date, open, close, high, low, volume):
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume

def stockhistory(code_list, start_date, end_date, conns):
    base = Base()
    financial_data = conns['financial_data']
    # res = pd.DataFrame()
    for code in code_list:
        res = pd.DataFrame()
        df = ts.get_k_data(code, start=start_date, end=end_date, autype=None)
        if df.empty:
            continue
        res = res.append(df, ignore_index=True)
        table = 'stock_' + code
        # print(table)
        base.batchwri(res, table, financial_data)

    # print(res)
    # base.batchwri(res, 'stock-000001', financial_data)

def mysql2RawData(code, conns):
    """
    将存储在MySql上的股票数据逐条转化为RawData类。
    :param code: 股票代码
    :param conns: 数据库连接
    :return: RawData类实例数据的列表list，每一个实例代表股票一天的记录。
    """
    base = Base()
    financial_data = conns['financial_data']
    raw_data = []
    table = 'stock_' + code
    df = financial_data.getdata(table)
    # print(df)
    for index in df.index:
        data = df.iloc[index].values[0:-1]
        # print(data)
        raw_data.append(RawData(data[0], data[1], data[2], data[3], data[4], data[5]))
    # print(len(raw_data))
    sorted_data = sorted(raw_data, key=lambda x:x.date)
    return sorted_data



if __name__ == "__main__":
    base = Base()
    financial_data = base.conn('financial_data')
    conns = {'financial_data': financial_data}
    code_list = ['000001']
    # stockhistory(code_list, '2017-07-01', '2018-07-31', conns)
    mysql2RawData('000001', conns)
