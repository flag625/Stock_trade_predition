# -*- coding: utf-8 -*-

import numpy as np

class DataSet(object):
    def __init__(self, images, labels):
        """
        :param images: numpy.asarry格式
        :param labels: numpy.asarry格式
        """
        assert images.shape[0] == labels.shape[0], \
            ("images.shape: %s, labels.shape: %s" %(images.shape, labels.shape))
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # 当样本走完一遍而训练还要继续时：
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # 打乱数据顺序
            prem = np.arange(self._num_examples)
            np.random.shuffle(prem)
            np.random.shuffle(prem)
            np.random.shuffle(prem)
            np.random.shuffle(prem)
            np.random.shuffle(prem)
            self._images = self._images[prem]
            self._labels = self._labels[prem]
            # 开始选取下一个batch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]