# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from CommonAPI.base import Base
import logging.config
import configparser

# conf = configparser.ConfigParser()
# conf.read("/Users/cloudin/PycharmProjects/Stock_trade_predition/CommonAPI/test.conf")
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# fh = logging.FileHandler(conf.get('path','log_path'), mode='a')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# logger.addHandler(fh)

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