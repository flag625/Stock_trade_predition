# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer
from tensorflow.contrib.layers.python.layers.layers import batch_norm


class LSTMgraph(object):
    def __init__(self, num_step, input_size, init_learning_rate, hidden_size, nclasses,
                 decay_step=500, decay_rate=1.0, cost=0.0002):
        """
        设定初始参数。
        :param num_step: 窗口值，时间步长数
        :param input_size: 输入值数量
        :param init_learning_rate: 初始学习速率
        :param hidden_size: 隐藏层数量
        :param nclasses: 输出值数量
        :param decay_step: 衰变速度
        :param decay_rate: 衰变系数
        :param cost: 成本系数
        """
        self.num_step = num_step
        self.input_size = input_size
        self.nclasses = nclasses
        self.hidden_size = hidden_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate = None
        self.global_step = None
        self.decay_step = decay_rate
        self.decay_rate = decay_rate
        self.keep_prob = None
        self.batch_size = None
        self.summary_op = None
        self.weights = None
        self.biases = None
        self.cost = cost
        self.loss = None
        self.x = None
        self.y = None
        self.position = None
        self.avg_position = None
        self.is_training = None

    def _create_learning_rate(self):
        '''
        create learning rate
        :return:
        '''
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step, self.decay_step,
                                                            self.decay_rate, staircase=True, name="learning_rate")

    def _create_placeholder(self):
        with tf.variable_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.num_step, self.input_size], name="history_feature")
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="target")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.is_training = tf.placeholder(tf.bool, name="mode")

    def _create_weight(self):
        with tf.variable_scope("weights"):
            self.weights = {
                'out': tf.get_variable("weights", [self.hidden_size, self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }
            self.biases = {
                'out': tf.get_variable("out", [self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }

    def batch_norm_layer(self, signal, scope):
        '''
        batch normalization layer before activation
        :param signal:
        :param scope: name scope
        :return: normalization signal
        '''
        return tf.cond(self.is_training,
                       lambda : batch_norm(signal, is_training=True,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2.5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1,
                                           scope=scope),
                       lambda : batch_norm(signal, is_training=False,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2.5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1,
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
        # self.x.shape = (batch_size, num_step, input_size)
        # xx.shape = [num_step, [batch_size, input_size]]
        xx = tf.unstack(self.x, self.num_step, 1)
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer)
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob,
                                      state_keep_prob=self.keep_prob)
        # outputs.shape = [num_step, [batch_size, hidden_size]]
        outputs, states = rnn.static_rnn(dropout_cell, xx, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        norm_signal = self.batch_norm_layer(signal, scope=scope)

        self.position = tf.nn.relu6(norm_signal, name="relu_limit") / 6.
        self.avg_position = tf.reduce_mean(self.position)
        self.loss = -100. * tf.reduce_mean(tf.multiply((self.y - self.cost), self.position, name="estimated_risk"))

    def _create_optimizer(self):
        '''
        create optimizer.
        最小化损失函数。
        :return:
        '''
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="optimizer").\
            minimize(self.loss, global_step=self.global_step)

    # 保存训练过程以及参数分布图并在tensorboard显示。
    def _create_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram_loss", self.loss)
        tf.summary.scalar("average_position", self.avg_position)
        tf.summary.histogram("histogram_position", self.position)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholder()
        self._create_weight()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()


