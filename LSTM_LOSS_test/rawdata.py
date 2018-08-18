# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from CommonAPI.base import Base


def stockhistory(code_list, start_date, end_date, conns):
    base = Base()
    financial_data = conns['financial_data']
    res = pd.DataFrame()
    for code in code_list:
        df = ts.get_k_data(code, start=start_date, end=end_date, autype=None)
        if df.empty:
            continue
        res = res.append(df, ignore_index=True)

    print(res)

if __name__ == "__main__":
     base = Base()
     financial_data = base.conn('financial_data')
     conns = {'financial_data': financial_data}
     code_list = ['000001']
     stockhistory(code_list, '2017-07-01', '2018-07-31', conns)
