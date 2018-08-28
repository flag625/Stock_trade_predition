# -*- coding: utf-8 -*-

import numpy as np
import talib

class CharFeatures(object):
    def __init__(self, selector):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI",
                          "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"}
        self.feature = []

    def moving_extract(self, window=30, open=None, close=None, high=None, low=None, volume=None,
                       with_label=True, flatten=True):
        self.extract(open, close, high, low, volume)
        featrue_arr = np.asarray(self.feature)
        print("feature_arr : " + str(featrue_arr.shape))
        p = 0
        rows = featrue_arr.shape[0]
        print("feature dimention : %d" %rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window < featrue_arr.shape[1]:
                x = featrue_arr[:, p:p+window]
                p_change = (close[p+window] - close[p+window-1]) / close[p+window-1]
                y = p_change
                if flatten:
                    x = x.flatten("F")
                    print("flatten : " + str(x.shape))
                moving_features.append(np.nan_to_num(x))
                moving_labels.append(y)
                p+=1
            return np.asarray(moving_features), np.asarray(moving_labels)
        else:
            moving_features = []
            while p + window < featrue_arr.shape[1]:
                x = featrue_arr[:, p:p+window]
                if flatten:
                    x = x.flatten("F")
                    print("flatten : " + str(x.shape))
                moving_features.append(np.nan_to_num(x))
                p+=1
            return np.asarray(moving_features)


    def extract(self, open=None, close=None, high=None, low=None, volume=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                print("extractint feature : %s" %feature_type)
                self.extract_by_type(feature_type, open=open, close=close, high=high, low=low, volume=volume)
            else:
                print("feature type not supported : %s" %feature_type)
        return self.feature

    def extract_by_type(self, feature_type, open=None, close=None, high=None, low=None, volume=None):
        if feature_type == 'ROCP':
            rocp = talib.ROCP(close, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

            # 标准化处理，负值归一化为-1，正值归一化为1，Nan值为0。
            norm_signal = np.minimum(np.maximum(np.nan_to_num(signal), -1), 1)
            norm_hist = np.minimum(np.maximum(np.nan_to_num(hist), -1), 1)
            norm_macd = np.minimum(np.maximum(np.nan_to_num(macd), -1), 1)

            # num.diff():计算离散差值，后一个元素减去前一个元素，shape = [1, macd_n - 1]
            # np.concatenate(): 组合0 和 离散差值 [0, diff]， shape = [1, macd_n]
            # np.minimum(np.maximun(x,-1),1):标准化处理，负值归一化为-1，正值归一化为1，Nan值为0。
            zero = np.asarray([0])
            macdrocp = np.minimum(np.maximum(np.concatenate((zero, np.diff(np.nan_to_num(macd)))), -1), 1)
            signalrocp = np.minimum(
                np.maximum(np.concatenate((zero, np.diff(np.nan_to_num(signal)))), -1), 1)
            histrocp = np.minimum(np.maximum(np.concatenate((zero, np.diff(np.nan_to_num(hist)))), -1), 1)

            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close, timeperiod=6)
            rsi12 = talib.RSI(close, timeperiod=12)
            rsi24 = talib.RSI(close, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            # np.maximum(x,1)：元素最小值设定为1。
            # np.nan_to_num：Nan值为0。
            # np.arctan：对矩阵a中每个元素取反正切
            vrocp = np.arctan(np.nan_to_num(talib.ROCP(np.maximum(volume, 1), timeperiod=1)))
            self.feature.append(vrocp)
        if feature_type == 'MA':
            ma5 = np.nan_to_num(talib.MA(close, timeperiod=5))
            ma10 = np.nan_to_num(talib.MA(close, timeperiod=10))
            ma20 = np.nan_to_num(talib.MA(close, timeperiod=20))
            ma30 = np.nan_to_num(talib.MA(close, timeperiod=30))
            ma60 = np.nan_to_num(talib.MA(close, timeperiod=60))
            ma90 = np.nan_to_num(talib.MA(close, timeperiod=90))
            ma120 = np.nan_to_num(talib.MA(close, timeperiod=120))
            ma180 = np.nan_to_num(talib.MA(close, timeperiod=180))
            ma360 = np.nan_to_num(talib.MA(close, timeperiod=360))
            ma720 = np.nan_to_num(talib.MA(close, timeperiod=720))
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - close) / close)
            self.feature.append((ma10 - close) / close)
            self.feature.append((ma20 - close) / close)
            self.feature.append((ma30 - close) / close)
            self.feature.append((ma60 - close) / close)
            self.feature.append((ma90 - close) / close)
            self.feature.append((ma120 - close) / close)
            self.feature.append((ma180 - close) / close)
            self.feature.append((ma360 - close) / close)
            self.feature.append((ma720 - close) / close)
        if feature_type == 'VMA':
            ma5 = np.nan_to_num(talib.MA(volume, timeperiod=5))
            ma10 = np.nan_to_num(talib.MA(volume, timeperiod=10))
            ma20 = np.nan_to_num(talib.MA(volume, timeperiod=20))
            ma30 = np.nan_to_num(talib.MA(volume, timeperiod=30))
            ma60 = np.nan_to_num(talib.MA(volume, timeperiod=60))
            ma90 = np.nan_to_num(talib.MA(volume, timeperiod=90))
            ma120 = np.nan_to_num(talib.MA(volume, timeperiod=120))
            ma180 = np.nan_to_num(talib.MA(volume, timeperiod=180))
            ma360 = np.nan_to_num(talib.MA(volume, timeperiod=360))
            ma720 = np.nan_to_num(talib.MA(volume, timeperiod=720))

            # np.nan_to_num：Nan值为0。
            # np.arctan：对矩阵a中每个元素取反正切
            ma5rocp = np.arctan(np.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = np.arctan(np.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = np.arctan(np.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = np.arctan(np.nan_to_num(talib.ROCP(ma30, timeperiod=1)))
            ma60rocp = np.arctan(np.nan_to_num(talib.ROCP(ma60, timeperiod=1)))
            ma90rocp = np.arctan(np.nan_to_num(talib.ROCP(ma90, timeperiod=1)))
            ma120rocp = np.arctan(np.nan_to_num(talib.ROCP(ma120, timeperiod=1)))
            ma180rocp = np.arctan(np.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            ma360rocp = np.arctan(np.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            ma720rocp = np.arctan(np.nan_to_num(talib.ROCP(ma720, timeperiod=1)))

            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)

            self.feature.append(np.arctan(np.nan_to_num((ma5 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma10 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma20 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma30 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma60 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma90 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma120 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma180 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma360 - volume) / (volume + 1))))
            self.feature.append(np.arctan(np.nan_to_num((ma720 - volume) / (volume + 1))))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close, timeperiod=1)
            vrocp = np.arctan(np.nan_to_num(talib.ROCP(np.maximum(volume, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)

def extract_features(rawdata, selector, windows=30, with_label=True, flatten=True):
    char_featrues = CharFeatures(selector)
    sorted_data = sorted(rawdata, key=lambda x:x.date)
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    for item in rawdata:
        closes.append(item.close)
        opens.append(item.open)
        highs.append(item.high)
        lows.append(item.low)
        volumes.append(float(item.volume))
    closes = np.asarray(closes)
    opens = np.asarray(opens)
    highs = np.asarray(highs)
    lows = np.asarray(lows)
    volumes = np.asarray(volumes)
    if with_label:
        moving_features, moving_labels = char_featrues.moving_extract(window=windows, open=opens, close=closes,
                                                                      high=highs, low=lows, volume=volumes,
                                                                      with_label=with_label, flatten=flatten)
        return moving_features, moving_labels
    else:
        moving_features = char_featrues.moving_extract(window=windows, open=opens, close=closes, high=highs,
                                                       low=lows, volume=volumes, with_label=with_label, flatten=flatten)
        return moving_features

from CommonAPI.base import Base
from LSTM_LOSS_test.rawdata import mysql2RawData
# test
if __name__ == '__main__':
    base = Base()
    financial_data = base.conn('financial_data')
    conns = {'financial_data': financial_data}
    rawdata = mysql2RawData('000001', conns)
    moving_featrues, moving_labels = extract_features(rawdata, ["ROCP", "MACD"])
    print("moving_featrun: " +str(moving_featrues.shape))
    print("moving_labels " + str(moving_labels.shape))




