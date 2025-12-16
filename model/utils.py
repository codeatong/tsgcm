import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path,fft=False):
    train = np.load(path + 'train.npz')
    test = np.load(path + 'test.npz')
    X_train, Y_train = train['X_train'], train['Y_train']
    X_test, Y_test = test['X_test'], test['Y_test']
    del train, test
    if fft:
        for i in range(len(X_train)):
            ori_matrix_fft = np.fft.fft(X_train[i, :, :], axis=0)
            X_train[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_test)):
            ori_matrix_fft = np.fft.fft(X_test[i, :, :], axis=0)
            X_test[i, :, :] = np.real(ori_matrix_fft)
    y_train = pd.DataFrame(Y_train, columns=['Y_train'])
    print(y_train.value_counts())
    print('-----------------------------------------')
    print('X_train.shape:{}'.format(X_train.shape))
    print('X_test.shape:{}'.format(X_test.shape))
    X_train = data_reshape(X_train)
    X_test = data_reshape(X_test)
    return X_train, X_test, Y_train, Y_test


def load_data_with_diffent_ratio(path,test_size,fft=False):
    all = np.load(path)
    X, Y = all['X'], all['Y']
    del all
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size[0], random_state=0, stratify=Y)
    if fft:
        for i in range(len(X_train)):
            ori_matrix_fft = np.fft.fft(X_train[i, :, :], axis=0)
            X_train[i, :, :] = np.real(ori_matrix_fft)
        for i in range(len(X_test)):
            ori_matrix_fft = np.fft.fft(X_test[i, :, :], axis=0)
            X_test[i, :, :] = np.real(ori_matrix_fft)
    y_train = pd.DataFrame(Y_train, columns=['Y_train'])
    print(y_train.value_counts())
    print('-----------------------------------------')
    print('X_train.shape:{}'.format(X_train.shape))
    print('X_test.shape:{}'.format(X_test.shape))
    X_train = data_reshape(X_train)
    X_test = data_reshape(X_test)
    return X_train, X_test, Y_train, Y_test
def data_reshape(X_train):
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])
    return X_train