# -*- coding: utf-8 -*-

"""
    合并数据集, 密集数据合并。
    @author:chenli0830(李辰)
    @source:https://github.com/happynoom/DeepTrade
"""

import numpy

def dense_2_one_hot(labels_dense, num_classes):
    """Convert class label from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_labels))
    labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
    return labels_one_hot

class DataSet(object):
    def __int__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "image.shape: %s labels.shape: %s" %(images.shape,
                                                 labels.shape)
        )
        self._num_examples = images.shape[0]
        images = images.astype(numpy.float32)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            #Finished epoch
            self._epochs_completed += 1
            #Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            #Start next batch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
