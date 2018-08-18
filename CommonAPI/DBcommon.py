# -*- coding: utf-8 -*-

import os
import traceback
import time
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import logging.config
from DBUtils.PooledDB import PooledDB
import configparser

conf = configparser.ConfigParser()
conf.read("./test.conf")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(conf.get('log_path', 'log_path'), mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
# print('测试:{}'.format(conf.get('log_path','log_path')))

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
                logger.warning('Retry get mysql connection, the [%d] times, err %s' %(i, e.message))
                if i == retry_num:
                    logger.warning(traceback.format_exc())
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

    def getdata(self, table, pars=None, tjs=None, blimit=None, elimit=None):
        '''
        从数据库中取出数据放到dataframe中
        :param table: 数据源表
        :param pars: list类型，列出想要提取的字段名，若为空则查询所有字段
        :param blimit: 数据行数最小值限制
        :param elimit: 数据行数最大值限制
        :return: dataframe类型查询结果
        '''
        self.conn = self.pool.connection()
        self.cursor = self.conn.cursor()
        if pars == None:
            item = '*'
        else:
            item = ','.join(pars)
        sql1 = 'select '+item+' from '+table
        if blimit != None or elimit != None:
            sql1 += ' limit '
            if blimit != None and elimit != None:
                sql1 += str(int(blimit)-1) + ',' + str(int(elimit) - int(blimit))
            elif elimit != None:
                sql1 += str(elimit)
            else:
                sql_count = "select table_rows from information_schema.tables where table_name='"+table+"'"
                self.cursor.excute(sql_count)
                n = self.cursor.fetchone()[0]
                sql1 += str(int(blimit) - 1) + ',' + str(n - int(blimit))
        if tjs != None:
            if sql1.find('where') != -1:
                sql1 = sql1.replace('where', 'where '+' and '.join(tjs)+'and')
            else:
                sql1 += ' where '+' and '.join(tjs)
        try:
            res = pd.read_sql(sql1, self.conn)
        except Exception as e:
            logger.info(u"执行失败：" + sql1 + "\n" + u"失败原因：")
            logger.info(e)
            raise e
        return res

    def write2mysql(self, dataframe, table):
        self.conn = self.pool.connection()
        self.cursor = self.conn.cursor()
        res = False
        try:
            engine = create_engine("mysql+pymysql://"+self.user+":"+self.pwd+"@"+self.host+":"
                                   +self.port+"/"+self.db+"?charset=utf8")
            dataframe.to_sql(name=table, con=engine, if_exists='qppend', index=False, index_label=False)
            res = True
        except Exception as e:
            print(e)
            logger.info(u"执行失败：write2mysql\n"+u"失败原因：")
            logger.info(e)
        return res