# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from CommonAPI.base import Base
from CommonAPI import loggingData

logger = loggingData()

def stockhistory(code_list, start_date, end_date, conns):
    base = Base()
    financial_data = conns['financial_data']
    fin = []
    for code in code_list:
        df = ts.get_k_data(code, start=start_date, end=end_date, autype=None)
        if df.empty:
            continue
        fin.append(df)
        print(fin)

    res = pd.DataFrame(fin)
    print(res)

if __name__ == "__main__":
     # base = Base()
     # financial_data = base.conn('financial_data')
     # conns = {'financial_data': financial_data}
     # code_list = ['000001']
     # stockhistory(code_list, '2018-08-01', '2018-08-17', conns)
     print("..")