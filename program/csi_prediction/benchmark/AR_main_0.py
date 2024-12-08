import h5py
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def sgcs(W_true2_sample, W_pre2_sample):
    score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
    # a = abs(score_tmp)
    score_cos = abs(score_tmp) * abs(score_tmp)  # 这样算对吗
    return score_cos

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num = np.abs(num)
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1 * num2))
    return cos
def preprocess_dataset(data):
    """
    data.shape: 12, 32, 2, time_slot, samples
    """
    pass

def sgcs_matrix(W_true2_sample, W_pre2_sample):
    vector_a = np.mat(W_true2_sample)
    vector_b = np.mat(W_pre2_sample)
    num = vector_a * vector_b.H
    num = np.abs(num)
    num1 = np.sqrt(np.trace(vector_a * vector_a.H))
    num2 = np.sqrt(np.trace(vector_b * vector_b.H))
    num = np.trace(num)
    cos = (num / (num1 * num2)).real
    return cos * cos



if __name__ == '__main__':
    debug_mode = False
    sgcs_mode = True
    print("hello world!")
    # read the datasets -- h5py
    # file = h5py.File('C:\\Users\\user1\\lzh_use\\quadriga_H_CSI.mat')  # (12, 32, 2, 1000, 285)
    # # data = np.random.randint(1, 10, size=(12, 32, 2, 1000, 285)) #test
    # data = np.array(file['wtxLayer1']) #main
    # data = np.transpose(data, [2, 1, 0, 3, 4])
    file = h5py.File('C:\\Users\\user1\\lzh_use\\AR_w3g_1_2GHz.mat')  # (12, 32, 2, 1000, 285)
    data = np.array(file['wtxLayer1']) #main
    # data = np.transpose(data, [2, 1, 0, 3, 4])
    print('finished reading data')
    print(data.shape)
    subcarrier, tx_num, rx_num, slot_length, user_num = data.shape

    # AR config
    lag_order = 5
    predict_step = 4 #predict the next 4 slots
    # initialization
    all_bench1_sgcs = []
    all_bench2_sgcs = []
    user_num = 285
    # Begin AR
    for i in range(5):
        true_slot_length = int(np.floor(slot_length/5))
        slot = [i + 5 * j for j in range(true_slot_length)]
        train_slot = slot[150:-4]
        test_slot = slot[-4:]
        data_test = data[0, :, 0, test_slot, :]['real'] + 1j * data[0, :, 0, test_slot, :]['imag']
        # debug: (4, 32, 285)
        data_test = data_test.transpose(1, 0, 2)
        data_pred = np.zeros((tx_num, predict_step, user_num), dtype='complex')
        data_bench1_pred = np.zeros((tx_num, predict_step, user_num), dtype='complex')
        for userid in range(user_num):
            for tx in range(tx_num):
                data_bench1 = {}
                data_pred_1d = {}
                for j in ['real', 'imag']:
                    train_data = data[0, tx, 0, train_slot, userid][j] # we set subcarrier and rx 0
                    data_bench1[j] = np.array([train_data[-1]]*predict_step)
                    AR_model = AutoReg(train_data, lags=lag_order, trend='n')
                    result = AR_model.fit()
                    seq_start = len(train_slot)
                    seq_end = len(train_slot) + predict_step - 1
                    prediction = result.predict(start=seq_start, end=seq_end)
                    data_pred_1d[j] = prediction

                data_bench1_pred[tx, :, userid] = data_bench1['real'] + 1j * data_bench1['imag']
                data_pred[tx, :, userid] = data_pred_1d['real'] + 1j * data_pred_1d['imag']
                
                if debug_mode:
                    # draw picture
                    if not np.random.randint(0, 10):
                        plt.figure(figsize=(12,6))
                        for j in ['real', 'imag']:
                            W_pre = data_pred_1d['real'] + 1j * data_pred_1d['imag']
                            W_true = data_test[tx, :, userid]
                            sgcs_value = sgcs(W_pre, W_true)
                            if j == 'real':
                                plt.subplot(121)
                                plt.title('user_id:%s, tx:%s, subcarrier:%s, rx:%s, real, sgcs:%f'
                                  %(userid, tx, subcarrier, 0, sgcs_value))
                            if j == 'imag':
                                plt.subplot(122)
                                plt.title('user_id:%s, tx:%s, subcarrier:%s, rx:%s, imag, sgcs:%f'
                                  %(userid, tx, subcarrier, 0, sgcs_value))
                            # true full picture:
                            full_data = data[0, tx, 0, :, userid][j]
                            # plt.scatter(range(len(full_data)), full_data)

                            # AR prediction train
                            plt.scatter(train_slot[-30:], full_data[train_slot][-30:])
                            # AR prediction test true 4 slots
                            plt.scatter(test_slot, full_data[test_slot])
                            # AR prediction test pred 4 slots
                            plt.scatter(test_slot, data_pred_1d[j])
                            # AR prediction bench1 pred 4 slots
                            bench1 = full_data[train_slot[-1]]
                            bench1 = [bench1, bench1, bench1, bench1, bench1]
                            bench1_pred_x = range(train_slot[-1], train_slot[-1] + 5 + 4 * 5, 5)
                            plt.scatter(bench1_pred_x, bench1)
                            x = np.repeat(train_slot[-1] + 5, 10)
                            array = [full_data[train_slot[-1] + 5], bench1[0], data_pred_1d[j][0]]
                            y = np.linspace(min(array) - 0.5, max(array) + 0.5, 10)
                            plt.plot(x, y, linestyle='--')
                        plt.show()
        bench2_sgcs = []
        bench1_sgcs = []
        if sgcs_mode:
            
            for i in range(predict_step):
                bench1_sgcs_user = np.zeros(user_num)
                bench2_sgcs_user = np.zeros(user_num)
                for userid in range(user_num):
                    bench1 = sgcs(data_bench1_pred[:,i,userid], data_test[:,i,userid])
                    bench1_sgcs_user[userid] = bench1
                    bench2 = sgcs(data_pred[:,i,userid], data_test[:,i,userid])
                    bench2_sgcs_user[userid] = bench2
                bench1_sgcs.append(np.mean(bench1_sgcs_user))
                bench2_sgcs.append(np.mean(bench2_sgcs_user))
            all_bench1_sgcs.append(bench1_sgcs)
            all_bench2_sgcs.append(bench2_sgcs)
            print(bench1_sgcs)
            print(bench2_sgcs)
    last_bench1_sgcs = np.mean(all_bench1_sgcs, 0)
    last_bench2_sgcs = np.mean(all_bench2_sgcs, 0)
    print(f"last_bench1_sgcs: {last_bench1_sgcs}")
    print(f"last_bench2_sgcs: {last_bench2_sgcs}")
