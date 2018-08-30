# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from CommonAPI.base import Base
from LSTM_LOSS_test.LSTMgraph import LSTMgraph
from LSTM_LOSS_test.rawdata import mysql2RawData
from LSTM_LOSS_test.charVal import extract_features
from LSTM_LOSS_test.dataset import DataSet

def train(tensor, train_set, val_set, train_steps=10000, batch_size=32, keep_prob=1., code=None):
    initial_step = 1
    val_features = val_set.images
    val_labels = val_set.labels
    VERBOSE_STEP = 10
    VALIDATION_STEP = VERBOSE_STEP * 10
    saver = tf.train.Saver()
    min_validation_loss = 100000000.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer_path = os.path.join(("./Results/stock-" + code), "graphs")
        writer = tf.summary.FileWriter(writer_path, sess.graph)
        for i in range(initial_step, initial_step + train_steps):
            batch_features, batch_labels = train_set.next_batch(batch_size)
            _, loss, avg_pos, summary = sess.run([tensor.optimizer, tensor.loss, tensor.avg_position, tensor.summary_op],
                                                 feed_dict={tensor.x: batch_features, tensor.y: batch_labels,
                                                            tensor.is_training: True, tensor.keep_prob: keep_prob})
            writer.add_summary(summary, global_step=i)

            if i % VERBOSE_STEP == 0:
                hint = None
                if i % VALIDATION_STEP == 0:
                    val_loss, val_avg_pos = sess.run([tensor.loss, tensor.avg_position],
                                                     feed_dict={tensor.x: val_features, tensor.y: val_labels,
                                                                tensor.is_training: False, tensor.keep_prob: keep_prob})
                    hint = "Average Train Loss at step {}: {:.7f} Average position {:.7f}, Validation Loss: {:.7f} " \
                           "Average position: {:.7f}".format(i, loss, avg_pos, val_loss, val_avg_pos)
                    if val_loss < min_validation_loss:
                        min_validation_loss = val_loss
                        ckpt_path = os.path.join(("./Results/stock-" + code), "checkpoint/best_model")
                        saver.save(sess, ckpt_path, i)
                else:
                    hint = "Average loss at step {}: {:.7f} Average position {:.7f}".format(i, loss, avg_pos)
                print(hint)

def predict(val_set, num_step, input_size, learning_rate, hiden_size, nclasses, code=None):
    featrues = val_set.images
    labels = val_set.labels
    with tf.Graph().as_default():
        tensor = LSTMgraph(num_step, input_size, learning_rate, hiden_size, nclasses)
        tensor.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt_path = os.path.join(("./Results/stock-" + code), 'checkpoint/checkpint')
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            pred, avg_pos = sess.run([tensor.position, tensor.avg_position],
                            feed_dict={tensor.x: featrues, tensor.y: labels,
                                       tensor.is_training: False, tensor.keep_prob: 1.})
            cr = calculate_cumulative_return(labels, pred)
            print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
            for i in range(len(labels)):
                print(str(labels[i]) + "\t" + str(pred[i]) + "\t" + str(cr[i] + 1.) + "\t" + str(cr[i]))

def calculate_cumulative_return(labels, preb):
    cr = []
    if len(labels) <= 0:
        return cr
    cr.append(1. * (1. + labels[0] * preb[0]))
    for l in range(1, len(labels)):
        cr.append(cr[l-1] * (1 + labels[l] * preb[l]))
    for i in range(len(cr)):
        cr[i] = cr[i] - 1
    return cr

def execute(operation="train", traincodes=None, predcodes=None):
    """

    :param operation:
    :param traincodes: 测试的股票代码集
    :param predcodes: 预测的股票代码集
    :return:
    """
    num_step = 30
    input_size = 58
    batch_size = 32
    learning_rate = 0.001
    hidden_size = 14
    nclasses = 1
    train_steps = 500
    validation_prob = 0.3
    keep_rate = 0.7
    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]
    input_shape = [30, 61]

    base = Base()
    financial_data = base.conn('financial_data')
    conns = {'financial_data': financial_data}

    if operation == "train":
        if not traincodes:
            print("ERROR：Missing stock codes！")
        else:
            for code in traincodes:
                print("processing stock code: " + code)
                raw_data = None
                try:
                    raw_data = mysql2RawData(code, conns)
                except Exception as e:
                    raise e
                train_features = []
                train_labels = []
                val_features = []
                val_labels = []
                moving_features, moving_labels = extract_features(rawdata=raw_data, selector=selector,
                                                                    windows=input_shape[0],
                                                                    with_label=True, flatten=False)
                validation_size = int(len(moving_features) * validation_prob)
                train_features.extend((moving_features[: -validation_size]))
                train_labels.extend(moving_labels[:-validation_size])
                val_features.extend((moving_features[-validation_size:]))
                val_labels.extend(moving_labels[-validation_size:])

                train_features = np.transpose(np.asarray(train_features), [0, 2, 1])
                train_labels = np.asarray(train_labels)
                train_labels = np.reshape(train_labels, [train_labels.shape[0], 1])

                val_features = np.transpose(np.asarray(val_features), [0, 2, 1])
                val_labels = np.asarray(val_labels)
                val_labels = np.reshape(val_labels, [val_labels.shape[0], 1])

                train_set = DataSet(train_features, train_labels)
                val_set = DataSet(val_features, val_labels)

                with tf.Graph().as_default():
                    tensor = LSTMgraph(num_step, input_size, learning_rate, hidden_size, nclasses)
                    tensor.build_graph()
                    train(tensor, train_set, val_set, train_steps, batch_size=batch_size, keep_prob=keep_rate, code=code)

    elif operation == "predict" and predcodes:
        for code in predcodes:
            print("processing code: " + code)
            raw_data = mysql2RawData(code, conns)
            moving_features, moving_labels = extract_features(rawdata=raw_data, selector=selector,
                                                              windows=input_shape[0],
                                                              with_label=True, flatten=False)
            moving_features = np.transpose(np.asarray(moving_features), [0, 2, 1])
            moving_labels = np.asarray(moving_labels)
            moving_labels = np.reshape(moving_labels, [moving_labels.shape[0], 1])

            validation_size = int(len(moving_features) * validation_prob)
            val_set = DataSet(moving_features[-validation_size:], moving_labels[-validation_size:])
            predict(val_set, num_step=num_step, input_size=input_size, learning_rate=learning_rate,
                    hiden_size=hidden_size, nclasses=nclasses, code=code)
    else:
        print("Operation not supported!")

# test
if __name__ == "__main__":
    codes = ['000001', '000002', '000004', '000005']
    execute(operation="train", traincodes=codes)
    # execute(operation="predict", predcodes=['000001', '000004'])