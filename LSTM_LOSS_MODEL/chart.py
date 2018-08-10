# -*- coding: utf-8 -*-

"""
    对原始数据进行处理，获取神经网络模型输入的特征值。
    @author:chenli0830(李辰)
    @source:https://github.com/happynoom/DeepTrade
"""

import numpy
import talib
from LSTM_LOSS_MODEL.rawdate import read_sample_data

class ChartFeature(object):
    def __init__(self, selector):
        self.selector = selector
        self.suppoorted = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"}
        self.feature = []

    def moving_extract(self, window=30, open=None, close=None, high=None, low=None,
                       volumes=None, with_label=True, flatten=True):
        self.extract(open=open, close=close, high=high, low=low,volumes=volumes)
        feature_arr = numpy.asarray(self.feature)
        print("featurn_arr: " + str(feature_arr.shape))
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" %rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window < feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                p_change = (close[p + window] - close[p + window - 1]) / close[p + window - 1]
                y = p_change
                # 返回一个折叠成一维的数组，"F"按列展平。
                if flatten:
                    x = x.flatten("F")
                    print("flatten: " + str(x.shape))
                moving_features.append(numpy.nan_to_num(x))
                moving_labels.append(y)
                p += 1
            return numpy.asarray(moving_features), numpy.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                p += 1
            return moving_features

    def extract(self, open=None, close=None, high=None, low=None, volumes=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.suppoorted:
                print("extracting featuen : %s" %feature_type)
                self.extract_by_type(feature_type, open=open, close=close, high=high, low=low, volumes=volumes)
            else:
                print("feature type not supported: %s" %feature_type)
        self.feature_distribution()
        return self.feature

    def feature_distribution(self):
        k = 0
        for feature_column in self.feature:
            fc = numpy.nan_to_num(feature_column)
            mean = numpy.mean(fc)
            var = numpy.var(fc)
            max_value = numpy.max(fc)
            min_value = numpy.min(fc)
            print("[%s_th feature] mean: %s, var: %s, max: %s, min: %s" %(k, mean, var, max_value,min_value))
            k = k + 1

    def extract_by_type(self, feature_type, open=None, close=None, high=None, low=None, volumes=None):
        if feature_type == 'ROCP':
            rocp = talib.ROCP(close, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1)
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1), 1)
            signalrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1), 1)

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
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            self.feature.append(vrocp)
        if feature_type == 'MA':
            ma5 = numpy.nan_to_num(talib.MA(close, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(close, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(close, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(close, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(close, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(close, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(close, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(close, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(close, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(close, timeperiod=720))
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
            ma5 = numpy.nan_to_num(talib.MA(volumes, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(volumes, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(volumes, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(volumes, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(volumes, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(volumes, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(volumes, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(volumes, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(volumes, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(volumes, timeperiod=720))
            ma5rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma30, timeperiod=1)))
            ma60rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma60, timeperiod=1)))
            ma90rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma90, timeperiod=1)))
            ma120rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma120, timeperiod=1)))
            ma180rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            ma360rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            ma720rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma720, timeperiod=1)))
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
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma5 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma10 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma20 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma30 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma60 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma90 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma120 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma180 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma360 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma720 - volumes) / (volumes + 1))))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close, timeperiod=1)
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)

def extract_feature(raw_data, selector, window=30, with_label=True, flatten=True):
    chart_feature = ChartFeature(selector)
    sorted_data = sorted(raw_data, key=lambda x:x.date)
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    for item in sorted_data:
        closes.append(item.close)
        opens.append(item.open)
        highs.append(item.high)
        lows.append(item.low)
        volumes.append(float(item.volume))
    closes = numpy.asarray(closes)
    opens = numpy.asarray(opens)
    highs = numpy.asarray(highs)
    lows = numpy.asarray(lows)
    volumes = numpy.asarray(volumes)
    if with_label:
        moving_features, moving_labels = chart_feature.moving_extract(window=window, open=opens,close=closes, high=highs,
                                                                      low=lows, volumes=volumes, with_label=with_label,
                                                                      flatten=flatten)
        return moving_features, moving_labels
    else:
        moving_features = chart_feature.moving_extract(window=window, open=opens,close=closes, high=highs, low=lows,
                                                       volumes=volumes, with_label=with_label, flatten=flatten)
        return moving_features


# from LSTM_LOSS_MODEL.rawdate import read_sample_data
#
# #test
# if __name__ == "__main__":
#     raw_data = read_sample_data("data/000001.csv")
#     moving_featrues, moving_labels = extract_feature(raw_data, ["ROCP","MACD"])
#     print("moving_featrun: " +str(moving_featrues.shape))
#     print("moving_labels " + str(moving_labels.shape))