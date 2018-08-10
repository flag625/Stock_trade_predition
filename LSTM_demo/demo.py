# -*- coding: utf-8 -*-

"""LSTM框架"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer
from tensorflow.contrib.layers.python.layers.layers import batch_norm

import numpy
from numpy.random import seed

import os

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
        self.X = None
        self.Y = None
        self.is_training = None
        self.loss = None # 损失函数
        self.output = None # 输出

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
            self.X = tf.placeholder(tf.float32, shape=[None, self.num_step, self.input_size], name="input_feature")
            self.Y = tf.placeholder(tf.float32, shape=[None, 1], name="output")
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
        # 矩阵分解,沿列第self.num_step个维分解，将张量 self.X 分割成 self.num_step 个张量数组[batch_size, input_size]
        XX = tf.unstack(self.X, self.num_step, 1)

        # 建立LSTM cell
        # orthogonal_initializer()：正交矩阵的初始化器
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer())
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob,
                                      output_keep_prob=self.keep_prob, state_keep_prob=self.keep_prob)
        # LSTM层的 输出 和 状态
        outputs, states = rnn.static_rnn(dropout_cell, XX, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        # 对LSTM 的 输出 进行标准化
        norm_signal = self.batch_norm_layer(signal, scope=scope)
        self.output = tf.nn.relu6(norm_signal, name="relu_limit") / 6.
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
        tf.summary.scalar("output", self.output)
        tf.summary.histogram("histogram output", self.output)
        # 将所有summary全部保存到磁盘，以便tensorboard显示。
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()


def train(model, train_set, val_set, train_steps=10000, batch_size=32, keep_prob=1.):
    initial_step = 1
    val_features = val_set.images
    val_labels = val_set.lebels
    VERBOSE_STEP = 10 # int(len(train_feature) / batch_size)
    VALIDATION_STEP = VERBOSE_STEP * 100

    # Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。
    # Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值 。
    saver = tf.train.Saver()
    min_validation_loss = 100000000.
    with tf.Session() as sess:
        # 初始化模型的参数
        sess.run(tf.global_variables_initializer())
        # 将摘要协议缓冲区写入事件文件
        writer = tf.summary.FileWriter("./graphs", sess.graph)
        for i in range(initial_step, initial_step + train_steps):
            batch_features, batch_labels = train_set.next_batch(batch_size)
            _, loss, output, summary = sess.run([model.optimizer, model.loss, model.output, model.summary_op],
                                                 feed_dict={model.X: batch_features, model.Y: batch_labels,
                                                            model.is_training: True, model.keep_prob: keep_prob})
            writer.add_summary(summary, global_step=i)
            # 每10次迭代更新损失函数和输出值。
            if i % VERBOSE_STEP == 0:
                hint = None
                if i % VALIDATION_STEP == 0:
                    val_loss, val_output = sess.run([model.loss, model.output],
                                        feed_dict={model.X: val_features,model.Y: val_labels,
                                                   model.is_training: False, model.keep_prob: 1.})
                    hint = 'Average Train Loss at step {}: {:.7f} Output {:.7f}, Validation Loss: {:.7f} Output: {:.7f}'.\
                        format(i, loss, output, val_loss, val_output)
                    if val_loss < min_validation_loss:
                        min_validation_loss = val_loss
                        saver.save(sess, "./checkpoint/best_model", i)
                else:
                    hint = 'Average loss at step {}: {:.7f} Output {:.7f}'.format(i, loss, output)
                print(hint)

def calculate_cumulative_return(labels, preb):
    cr = []
    if len(labels) <= 0:
        return cr
    cr.append(1. * (1. + labels[0] * preb[0]))
    for l in range(1,len(labels)):
        cr.append(cr[l - 1] * (1 + labels[l] * preb[l]))
    for i in range(len(cr)):
        cr[i] = cr[i] - 1
    return cr

def predict(val_feature, val_labels, num_step, input_size, learning_rate, hidden_size, nclassess):
    features = val_feature
    labels = val_labels
    model = LSTM_model(num_step, input_size, learning_rate, hidden_size, nclassess)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        pred = sess.run([model.output],
                        feed_dict={model.X: features, model.Y: labels,
                                   model.is_training:False, model.keep_prob: 1.})
        cr = calculate_cumulative_return(labels, pred)
        print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
        for i in range(len(labels)):
            print(str(labels[i]) + "\t" + str(pred[i]) + "\t" + str(cr[i] + 1.) + "\t" + str(cr[i]))

def main(operation='train'):
    # 第一步：设置参数
    num_step = None
    input_size = None
    batch_size = None
    learning_rate = None
    hidden_size = None
    nclasses = None
    validation_size = None
    keep_rate = None

    selector = None  # 参数值类型选择，列表类型。

    # 第二步：执行训练或预测：
    if operation == "train":
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []
        # 1、读取原始数据：rawdata.py
        raw_data = None
        # 2、提取特征值，划分window(num_step)，得到格式为[n, input_size, num_step]矩阵：chart.py
        moving_features = None # 从raw_data中生成。
        moving_labels = None # 从raw_data中生成。
        # 3、划分训练集和测试集
        train_features.extend(moving_features[:-validation_size])
        val_features.extend(moving_features[-validation_size:])
        train_labels.extend(moving_labels[:-validation_size])
        val_labels.extend(moving_labels[-validation_size:])
        # 4、训练集转置并转化为类，使用next_batch在每次迭代中输入合适的[batch_size, num_step, input_size]矩阵：dataset.py
        train_features = numpy.transpose(numpy.asarray(train_features),[0, 2, 1])
        train_labels = numpy.asarray(train_labels)
        train_labels = numpy.reshape(train_labels, [train_labels.shape[0], 1])
        val_features = numpy.transpose(numpy.asarray([val_features]), [0, 2, 1])
        val_labels = numpy.asarray(val_labels)
        val_labels = numpy.reshape(val_labels, [val_labels.shape[0], 1])
        # 整合features 和 labels 集合在 DataSet 类， 利用next_batch迭代输入batch_size个数据。
        train_set = None
        val_set = None
        # 5、创建LSTM_model类实例，执行train函数
        model = LSTM_model(num_step, input_size, learning_rate, hidden_size, nclasses)
        model.build_graph()
        train(model, train_set, val_set, train_set, batch_size=batch_size, keep_rate=keep_rate)
    elif operation == "predict":
        # 1、读取原始数据：rawdata.py
        raw_data = None
        # 2、提取特征值，划分window(num_step)，得到格式为[n, input_size, num_step]矩阵：chart.py
        moving_features = None  # 从raw_data中生成。
        moving_labels = None  # 从raw_data中生成。
        # 3、训练集转置并转化为类，[batch_size, num_step, input_size]矩阵：dataset.py
        moving_features = numpy.asarray(moving_features)
        moving_features = numpy.transpose(moving_features, [0, 2, 1])
        moving_labels = numpy.asarray(moving_labels)
        moving_labels = numpy.reshape(moving_labels, [moving_labels.shape[0], 1])
        val_set = None
        # 4、创建LSTM_model类实例，执行predice函数
        predict(val_set, num_step=num_step, input_size=input_size, learning_rate=learning_rate,
                hidden_size=hidden_size, nclassess=nclasses)