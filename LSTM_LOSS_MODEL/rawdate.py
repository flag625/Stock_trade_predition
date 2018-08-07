# -*- coding: utf-8 -*-

"""
    截取data文件夹的.csv文件的数据。
    @author:chenli0830(李辰)
    @source:https://github.com/happynoom/DeepTrade
"""

class RawData(object):
    def __init__(self, date, open, high, low, close, volume):
        self.date = date
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume


def read_sample_data(path):
    print("reading histories...")
    raw_data = []
    separator = "\t"
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("date"):
                continue
            l = line[:-1]
            fields = l.split(separator)
            if len(fields) > 5:
                raw_data.append(RawData(fields[0], float(fields[1]), float(fields[2]),
                                        float(fields[3]), float(fields[4]), float(fields[5])))
    sorted_data = sorted(raw_data, key=lambda x:x.date)
    print("got %s records." %len(sorted_data))
    return sorted_data

#test
if __name__ == "__main__":
    print(read_sample_data("data/000001.csv"))