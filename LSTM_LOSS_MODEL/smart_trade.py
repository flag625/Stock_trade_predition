# -*- coding: utf-8 -*-

"""
    LSTM网络结构与LOSS函数。
    @author:chenli0830(李辰)
    @source:https://github.com/happynoom/DeepTrade
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import os

from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer
from LSTM_LOSS_MODEL.rawdate import RawData, read_sample_data
from LSTM_LOSS_MODEL.chart import extract_feature
import numpy
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import sys
from numpy.random import seed


class SmartTrade(object):
    def __init__(self, num_step, input_size, init_learning_rate, hidden_size, nclasses,
                 decay_step=500, decay_rate=1.0, cost=0.0002):
        """
        建立SmartTrade参数
        :param num_step:
        :param input_size:
        :param init_learning_rate:
        :param hidden_size:
        :param nclasses:
        :param decay_step:
        :param decay_rate:
        :param cost:
        """
        self.num_step = num_step
        self.input_size = input_size
        self.global_step = None
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.learning_rate = None
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.position = None
        self.summary_op = None
        self.weights = None
        self.biases = None
        self.cost = cost
        self.loss = None
        self.avg_position = None
        self.keep_prob = None
        self.x = None
        self.y = None
        self.is_training = None

    def _create_learning_rate(self):
        '''
        create learning rate
        :return:
        '''
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step,
                                                            self.decay_step, self.decay_rate, staircase=True,
                                                            name="learning rate")

    def _create_placeholder(self):
        with tf.variable_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.num_step, self.input_size], name='history_feature')
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target')
            self.is_training = tf.placeholder(tf.bool, name='mode')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def _create_weight(self):
        with tf.variable_scope("weights"):
            self.weights = {
                'out': tf.get_variable("weights", [self.hidden_size, self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }
            self.biases = {
                'out': tf.get_variable("bias", [self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }

    def batch_norm_layer(self, signal, scope):
        '''
        在激活之间批量归一化的层
        :param signal: input signal
        :param scope: name scope
        :return: normalized signal
        '''
        return tf.cond(self.is_training,
                       lambda : batch_norm(signal, is_training=True,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2,5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1., scope=scope),
                       lambda : batch_norm(signal, is_training=False,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2,5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1.,
                                           scope=scope, reuse=True))

    def _create_loss(self):
        '''
        风险评估损失函数
        Loss = -100. * mean(P * (R-c))
        P : self.position, output, the planed position we should hold to next day
        R : self.y, the change rate of next day
        c : cost
        :return:
        '''
        xx = tf.unstack(self.x, self.num_step, 1)
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer())
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob,
                                      output_keep_prob=self.keep_prob, state_keep_prob=self.keep_prob)
        outputs, states = rnn.static_rnn(dropout_cell, xx, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        norm_signal = self.batch_norm_layer(signal, scope=scope)
        self.position = tf.nn.relu6(norm_signal, name="relu_limit") / 6.
        self.avg_position = tf.reduce_mean(self.position)
        self.loss = -100. * tf.reduce_mean(tf.multiply((self.y - self.cost), self.position, name='estimated_risk'))

    def _create_optimizer(self):
        '''
        优化
        :return:
        '''
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="optimizer").\
            minimize(self.loss, global_step=self.global_step)

    def _create_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        tf.summary.scalar('average position', self.avg_position)
        tf.summary.histogram("histogram position", self.avg_position)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholder()
        self._create_weight()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

def train(trade, train_set, val_set, train_steps=10000, batch_size=32, keep_rate=1.):
    initial_stap = 1
    val_features = val_set.images
    val_labals = val_set.labels
    VERBOSE_STEP = 10
    VALIDATION_STEP = VERBOSE_STEP * 100

    saver = tf.train.Saver()
    min_validation_loss = 100000000.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs", sess.graph)
