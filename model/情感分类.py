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


# def my_normalize(tmp):
#     a = tmp
#     colmax = np.max(a, axis=0)
#     for i in range(107):
#         a[:, i] = a[:, i] / colmax[i]
#     return a
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
                gg = np.array(row[:], dtype='float64')
                tmp[i, :] = gg
                i += 1
        # 归一化
        data[di,] = tmp[:, :]
        di += 1
        count = count + 1
        if count % 20 == 0:
            print('had read %d data...' % count)
    return data


def get_train_betch(X_train, y_train, batch_size, topPath):
    x_data = None
    maxNum = 2500
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
    with open('myEmotionDEAM.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            song_name.append(row[0])
            song_label.append(eval(row[1]))

    # print(song_name[0:20])
    # 数据集划分 8:2
    board = len(song_name)
    # board = 500
    print("song number=%d" % board)
    song_label = np_utils.to_categorical(song_label)
    X_train, X_test, y_train, y_test = train_test_split(song_name[:board], song_label[:board], test_size=0.2,
                                                        random_state=42)

    batch_size = TRAIN_COEF['batch_size']
    # batch_size = len(X_train)
    # print(batch_size)

    # 防止宕机的一些设置
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # dataX = read_data(X_train, topPath)  # 读取
    # X = np.reshape(dataX, (len(dataX), 3600, 107))

    y = y_train
    # 搭建网络
    model = Sequential()
    num1, num2 = TRAIN_COEF['shape']
    model.add(LSTM(64, input_shape=TRAIN_COEF['shape'], return_sequences=True))
    # model.add(LSTM(32))
    # model.add(LSTM(32, input_shape=(3600, 107)))
    # model.add(LSTM(100, batch_input_shape=(batch_size, 3800,107), return_sequences=True))
    # model.add(LSTM(100, batch_input_shape=(batch_size, num1,num2), stateful=True))
    # model.add(LSTM(16, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dense(2, activation='softmax'))
    # KerasOptimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #last is 0.001
    # model.compile(loss='categorical_crossentropy', optimizer=KerasOptimizer, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # it actually work
    model.fit(get_train_betch(X_train, y_train, batch_size=batch_size, topPath=PATH['DEAM_FEATURE_PATH']),
              epochs=TRAIN_COEF['epoch'],
              verbose=1, steps_per_epoch=int(len(X_train) / batch_size))

    # X1=read_data(X_train, PATH['DEAM_FEATURE_PATH'])
    # X=np.reshape(X1,(len(X1),num1,num2))
    # model.fit(X, y, epochs=TRAIN_COEF['epoch'], batch_size=TRAIN_COEF['batch_size'], verbose=1)

    # for i in range(epoch):
    #     model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2)
    #     model.reset_states()

    # result = model.fit_generator(generator=get_train_betch(X_train, y_train, batch_size=batch_size, topPath=topPath),
    #                                  verbose=2,epochs=20)

    # 保存模型
    print("Saving model to disk \n")
    mp = "D:\\DataSet\\model\\tmp.h5"
    model.save(mp)

    # 评估
    dataX_test = read_data(X_test, PATH['DEAM_FEATURE_PATH'])
    x1, x2 = TRAIN_COEF['shape']
    dataX_test = np.reshape(dataX_test, (len(dataX_test), x1, x2))
    datay_test = y_test

    scores = model.evaluate(dataX_test, datay_test, verbose=0)
    # model.reset_states()
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # 打印运行时间
    time_end = time.time()
    import datetime

    k = datetime.timedelta(seconds=int((time_end - time_start)))
    print('cost time:', k)

