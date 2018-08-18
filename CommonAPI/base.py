# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
from CommonAPI.DBcommon import mysql2pd
import configparser

conf = configparser.ConfigParser()
conf.read("./test.conf")
# print(conf.options('log_path'))

class Base():
    def __init__(self):
        self.dbconfig = {
            'financial_data':(conf.get('db_test','host'), conf.get('db_test','port'),conf.get('db_test','db'),
                              conf.get('db_test', 'user'), conf.get('db_test','pwd')),
        }

    def conn(self, db):
        if db == 'financial_data':
            return mysql2pd(*self.dbconfig[db])

    def batchwri(self, res, table, conn):
        if res.empty:
            print(table + ' is None')
        else:
            print(res.shape)
            total = res.shape[0]
            nowrow = 0
            while nowrow < total - 1000:
                conn.write2mysql(res[nowrow : nowrow + 1000], table)
                nowrow += 1000
            conn.write2mysql(res[nowrow :], table)