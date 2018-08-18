#-*- coding: utf-8 -*-

import logging.config
import configparser
import threading
from time import ctime
import pandas as pd

conf = configparser.ConfigParser()
conf.read("./test.conf")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(conf.get('path','log_path'), mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
# print('这里是测试内容：{}'.format(conf.get('path','log_path')))

#多线程处理
class ThreadFunc(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.res = None

    def getResult(self):
        return self.res

    def run(self):
        self.res = self.func(*self.args)

#测试
def exce_threads(func, clsj, *args, mul_t=4):
    '''
    测试多线程处理
    :param func: 多线程运行函数
    :param mul_t: 线程数量
    :param clsj: 多线程要处理数据，列表类型
    :param args: 函数其余条件
    :return: 整合多线程处理后的数据，返回dataframe格式
    '''

    # 将处理数据按线程数量均分
    try:
        k = int(len(clsj) / mul_t)
    except Exception as e:
        logger.info(u"均分失败！\n" + u"失败原因：")
        logger.info(e)
        raise e
    sj_list = []
    for i in range(mul_t):
        if i != (mul_t - 1):
            sj = clsj[i * k:(i + 1) * k].reset_index(drop=True)
        else:
            sj = clsj[i * k:].reset_index(drop=True)
        sj_list.append(sj)

    start = ctime()
    print('starting at: ', start)
    # 多线程
    threads = []
    try:
        for i in range(mul_t):
            t = ThreadFunc(func, (sj_list[i], *args), func.__name__)
            threads.append(t)
        for i in range(mul_t):
            threads[i].start()
        for i in range(mul_t):
            threads[i].join()
    except Exception as e:
        logger.info(u"多线程失败！\n" + u"失败原因：")
        logger.info(e)
        raise e
    print('all DONE at: ', ctime(), '  start at: ', start)

    #将线程处理后的数据整合
    res = []
    for t in threads:
        res.append(t.getResult())
    result = pd.concat(res).reset_index(drop=True)
    return result


