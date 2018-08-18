# -*- coding: utf-8 -*-

import os
import time
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import logging.config
from DBUtils.PooledDB import PooledDB
import configparser

conf = configparser.ConfigParser()
conf.read("./test.conf")
# print('测试:{}'.format(conf.get('db_test','host')))

class mysql2pd(object):
    def __init__(self, host, port, db, user, pwd, retry_num=3, env_lang='utf8'):
        '''
        :param host: 主机ip
        :param port: 端口号
        :param db: 数据库
        :param user: 用户名
        :param pwd: 密码
        :param retry_num: 做多重新连接次数
        :param env_lang: 字符格式
        '''

        for i in range(1, retry_num+1):
            try:
                self.pool = self.connect(host, port, user, pwd, db, env_lang)
                break
            except Exception as e:
                if i == retry_num:
                    raise e
                time.sleep(10)
            self.db = db
            self.user = user
            self.host = host
            self.pwd = pwd
            self.port = port

    def connect(self, host, port, user, pwd, db, chartset='utf8'):
        pool = PooledDB(pymysql, 5, host=host, port=int(port), user=user, passwd=pwd, db=db, chartset=chartset)
        return pool

    def close(self):
        self.pool.close()