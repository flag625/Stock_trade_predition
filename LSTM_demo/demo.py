# -*- coding: utf-8 -*-

"""LSTM框架"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import numpy
from numpy.random import seed

class LSTM_model(object):
    def __init__(self, num_step, input_size, init_learning_rate, hidden_size, nclasses,
                 decay_step, decay_rate=1.0):
        """
        Initialize parameters for LSTM model.
        :param num_step: 一个输入系列包含的对象数量
        :param input_size: 一个输入对象的特征值数量
        :param init_learning_rate: 初始学习速率
        :param hidden_size: 隐藏层数量，一个 LSTM cell 的节点数量
        :param nclasses: 输出值的数量
        :param decay_step: 衰减速度
        :param decay_rate: 衰减系数
        """

        self.num_step = num_step
        self.input_size = input_size
        self.global_step = None  # 总迭代次数
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.learning_rate = None
        self.hidden_size = hidden_size
        self.nclasses= nclasses
        self.summary_op = None
        self.weights = None # 权重
        self.biases = None # 偏置
        self.keep_prob = None
        self.x = None
        self.y = None
        self.is_training = None
        self.loss = None # 损失函数

    def _create_learning_rate(self):
        '''
        create learning rate
        :return:
        '''

        # 定义 parameter 变量空间，包括全局迭代次数和学习速率
        with tf.variable_scope("parameter"):

            # trainable 如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
            self.global_step = tf.Variable(0, trainable=False, name="global_step")

            # 指数衰减法：decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
            # decayed_learning_rate为每一轮优化时使用的学习率；
            # learning_rate为事先设定的初始学习率；decay_rate为衰减系数；decay_steps为衰减速度。
            # 如果staircase=True，那就表明每decay_step次计算学习速率变化，更新原始学习速率，
            # 如果是False，那就是每一步都更新学习速率。
            self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step,
                                                            self.decay_step, self.decay_rate,
                                                            staircase=True, name="learning_rate")

    def _create_placeholders(self):
        '''
        Initialize input, output.
        :return:
        '''

        # 定义 input 变量空间，包括输入x，输出y，训练集标志，保留率
        with tf.variable_scope("input"):

            # tf.placeholder(dtype, shape=None, name=None)
            # placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值。
            # dtype：数据类型。常用的是tf.float32, tf.float64等数值类型。
            # shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2, 3], [None, 3]表示列是3，行不定。
            # name：名称。
            self.x = tf.placeholder(tf.float32, shape=[None, self.num_step, self.input_size], name="input_feature")
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="output")
            self.is_training = tf.placeholder(tf.bool, name="mode")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _create_weights(self):
        # 定义 weights 变量空间，包括权重W 和 偏置b
        with tf.variable_scope("weights"):
            self.weights = {
                #输出层的权值
                'out': tf.get_variable("weights", [self.hidden_size, self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }
            # tf.random_normal_initializer：用正态分布产生张量的初始化器
            self.biases = {
                #输出层的偏置值
                'out': tf.get_variable("bias", [self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }

    def batch_norm_layer(self, signal, scope):
        '''
        batch normalization layer before activation
        :param signal: input signal
        :param scope: name scope
        :return: normalization signal
        '''
        # 注意： is_training 是 tf.palceholder(tf.bool) 类型
        # tf.cond 类似于c语言中的if...else...，用来控制数据流向
        # Batch Normalization通过减少内部协变量加速神经网络的训练
        return tf.cond(self.is_training,
                       lambda : batch_norm(signal, is_training=True,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2.5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1, scope=scope),
                       lambda : batch_norm(signal, is_training=False,
                                           param_initializers={"beta": tf.constant_initializer(3.),
                                                               "gamma": tf.constant_initializer(2.5)},
                                           center=True, scale=True, activation_fn=tf.nn.relu, decay=1,
                                           scope=scope, reuse=True))

    def _create_loss(self):
        '''
        loss function.
        :return:
        '''
        # #储存在内存中
        # with tf.device("/cpu:0"):
        # 矩阵分解,沿列第self.num_step个维分解，将张量 self.x 分割成 self.num_step 个张量数组
        xx = tf.unstack(self.x, self.num_step, 1)

        # 建立LSTM cell
        # orthogonal_initializer()：正交矩阵的初始化器
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer())
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob,
                                      output_keep_prob=self.keep_prob, state_keep_prob=self.keep_prob)
        # LSTM层的 输出 和 状态
        outputs, states = rnn.static_rnn(dropout_cell, xx, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        # 对LSTM 的 输出 进行标准化
        norm_signal = self.batch_norm_layer(signal, scope=scope)
        # self.loss = "损失函数公式"

    def _create_optimizer(self):
        '''
        create optimizer.
        最小化损失函数。
        :return:
        '''
        # with tf.device("/cpu:0"):
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="optimizer").\
            minimize(self.loss, global_step=self.global_step)

    # 保存训练过程以及参数分布图并在tensorboard显示。
    def _create_summary(self):
        # 显示标量信息
        tf.summary.scalar("loss", self.loss)
        # 显示直方图信息
        tf.summary.histogram("histogram loss", self.loss)
        # 将所有summary全部保存到磁盘，以便tensorboard显示。
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()




