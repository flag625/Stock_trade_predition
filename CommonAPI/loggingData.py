# -*- coding: utf-8 -*-

import logging.config
import configparser

def loggingdata():
    conf = configparser.ConfigParser()
    conf.read("/Users/cloudin/PycharmProjects/Stock_trade_predition/CommonAPI/test.conf")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(conf.get('path','log_path'), mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger