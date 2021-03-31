import csv
import os
import time
import xlrd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import keras.optimizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from coef import TRAIN_COEF, PATH


def my_normalize(tmp):
    a = tmp
    colmax = np.max(a, axis=0)
    for i in range(140):
        a[:, i] = a[:, i] / colmax[i]
    return a


def read_data(file_name, topPath):
    count = 0
    num1, num2 = TRAIN_COEF['shape']
    data = np.zeros((len(file_name), num1, num2))
    di = 0
    for name in file_name:
        tmp = np.zeros((num1, num2))
        with open(os.path.join(topPath, name + '.csv')) as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i >= num1:
                    break
                gg = np.array(row[0:140], dtype='float64')
                tmp[i, :] = gg
                i += 1
        # 归一化
        tmp = my_normalize(tmp)
        data[di,] = tmp[:, :]
        di += 1
        count = count + 1
        if count % 20 == 0:
            print('had read %d data...' % count)
    return data


def get_train_betch(X_train, y_train, batch_size, topPath):
    x_data = None
    maxNum = 900
    x_len = len(X_train)
    if len(X_train) > maxNum:
        x_data = read_data(X_train[:maxNum], topPath)
    else:
        x_data = read_data(X_train, topPath)
    # X_train = x
    num1, num2 = TRAIN_COEF['shape']
    # while True:
    #     for i in range(0, len(X_train), batch_size):
    #         if i + batch_size > len(X_train):
    #             break
    #         x = read_data(X_train[i:i + batch_size], topPath)
    #         y = y_train[i:i + batch_size]
    #         yield (x, y)
    while True:
        for i in range(0, x_len, batch_size):
            if i + batch_size > x_len:
                break
            if i + batch_size < maxNum:
                x = x_data[i:i + batch_size, ]
                y = y_train[i:i + batch_size]
            else:
                if i < maxNum:
                    gap = maxNum - i
                    x = np.zeros((batch_size, num1, num2))
                    y = y_train[i:i + batch_size]
                    x[0:maxNum - i, ] = x_data[i:maxNum, ]
                    x[maxNum - i:, ] = read_data(X_train[maxNum:maxNum + batch_size - gap], topPath)
                else:
                    x = read_data((X_train[i:i + batch_size]), topPath)
                    y = y_train[i:i + batch_size]

            yield (x, y)


if __name__ == '__main__':
    time_start = time.time()
    song_name = []
    song_label = []
    with open('out/myEmotion.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            song_name.append(row[0])
            song_label.append(eval(row[1]))

    # print(song_name[0:20])
    # 数据集划分 8:2
    board = len(song_name)
    # board=200
    print("song number=%d" % board)
    song_label = np_utils.to_categorical(song_label)
    X_train, X_test, y_train, y_test = train_test_split(song_name[:board], song_label[:board], test_size=0.2,
                                                        random_state=42)

    batch_size = TRAIN_COEF['batch_size']
    # batch_size = len(X_train)
    # print(batch_size)

    # dataX = read_data(X_train, topPath)  # 读取
    # X = np.reshape(dataX, (len(dataX), 3600, 107))

    a = input("是否使用模型? : 1:0")
    if a == '0':
        y = y_train
        # 搭建网络
        model = Sequential()
        num1, num2 = TRAIN_COEF['shape']
        model.add(LSTM(100, input_shape=TRAIN_COEF['shape']))
        # model.add(LSTM(32, input_shape=(3600, 107)))
        # model.add(LSTM(100, batch_input_shape=(batch_size, 3800,107), return_sequences=True))
        # model.add(LSTM(100, batch_input_shape=(batch_size, num1,num2), stateful=True))
        # model.add(LSTM(16, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dense(2, activation='softmax'))
        # sgd = keras.optimizers.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=True)
        KerasOptimizer = keras.optimizers.RMSprop(lr=0, rho=0.9, epsilon=1e-08, decay=0.0)  # last is 0.001
        model.compile(loss='categorical_crossentropy', optimizer=KerasOptimizer, metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # it actually work
        model.fit(get_train_betch(X_train, y_train, batch_size=batch_size, topPath=PATH['EMO_FEATURES_PATH']),
                  epochs=TRAIN_COEF['epoch'],
                  verbose=2, steps_per_epoch=int(len(X_train) / batch_size))

        # model.fit(X, y, epochs=epoch, batch_size=1, verbose=2)

        # for i in range(epoch):
        #     model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2)
        #     model.reset_states()

        # result = model.fit_generator(generator=get_train_betch(X_train, y_train, batch_size=batch_size, topPath=topPath),
        #                                  verbose=2,epochs=20)

        # 保存模型
        print("Saving model to disk \n")
        mp = "/Users/chenfeng/Desktop/IAE/emotion/emotion_predict.h5"
        model.save(mp)

        # 评估
        dataX_test = read_data(X_test, PATH['EMO_FEATURES_PATH'])
        x1, x2 = TRAIN_COEF['shape']
        dataX_test = np.reshape(dataX_test, (len(dataX_test), x1, x2))
        datay_test = y_test

        scores = model.evaluate(dataX_test, datay_test, batch_size=batch_size, verbose=0)
        model.reset_states()
        print("Model Accuracy: %.2f%%" % (scores[1] * 100))

        # 打印运行时间
        time_end = time.time()
        import datetime

        k = datetime.timedelta(seconds=int((time_end - time_start)))
        print('cost time:', k)
    else:  # a='1':
        from keras.models import load_model

        model = load_model("/Users/chenfeng/Desktop/IAE/emotion/emotion_predict.h5")
        # 评估
        dataX_test = read_data(X_test, PATH['EMO_FEATURES_PATH'])
        x1, x2 = TRAIN_COEF['shape']
        dataX_test = np.reshape(dataX_test, (len(dataX_test), x1, x2))
        datay_test = y_test
        for i in range(len(X_test)):
            k = np.reshape(dataX_test[i], (1, x1, x2))
            tmp = model.predict(k)
            print(X_test[i], tmp,datay_test[i])
