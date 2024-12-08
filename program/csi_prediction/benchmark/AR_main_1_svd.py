import h5py
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
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
    # data = np.random.randint(1, 10, size=(32, 1000, 12, 285)) #test
    file = h5py.File('dataset\w3g_1_svd.mat')  # (32, 1000, 12, 285)
    data = np.array(file['data_svd']) #mai
    data = data.transpose(1, 0, 2, 3)
    print('finished reading data')
    slot_length, tx_num, subcarrier, user_num = data.shape

    # AR config
    lag_order = 5
    predict_step = 4 #predict the next 4 slots
    # initialization
    bench1_sgcs_time = []
    bench2_sgcs_time = []
    # Begin AR
    for i in tqdm(range(5)):
        true_slot_length = int(np.floor(slot_length/5))
        slot = [i + 5 * j for j in range(true_slot_length)]
        train_slot = slot[:-4]
        test_slot = slot[-4:]
        data_test = data[test_slot, :, :, :]['real'] + 1j * data[test_slot, :, :, :]['imag']
        # (4, 32, 12, 285)
        data_pred = np.zeros((predict_step, tx_num, subcarrier, user_num), dtype='complex')
        data_bench1_pred = np.zeros((predict_step, tx_num, subcarrier, user_num), dtype='complex')
        for userid in tqdm(range(user_num)):
            for sb in (range(subcarrier)):
                for tx in (range(tx_num)):
                    data_bench1 = {}
                    data_pred_1d = {}
                    for j in ['real', 'imag']:
                        train_data = data[train_slot, tx, sb, userid][j] 
                        data_bench1[j] = np.array([train_data[-1]]*predict_step)
                        AR_model = AutoReg(train_data, lags=lag_order, trend='n')
                        result = AR_model.fit()
                        seq_start = len(train_slot)
                        seq_end = len(train_slot) + predict_step - 1
                        prediction = result.predict(start=seq_start, end=seq_end)
                        data_pred_1d[j] = prediction

                    data_bench1_pred[:, tx, sb, userid] = data_bench1['real'] + 1j * data_bench1['imag']
                    data_pred[:, tx, sb, userid] = data_pred_1d['real'] + 1j * data_pred_1d['imag']
        print(f"{i} finished the prediction")
        bench2_sgcs = []
        bench1_sgcs = []
        if sgcs_mode:
            for t in (range(predict_step)):
                bench1_sgcs_sb_user = np.zeros((user_num, subcarrier))
                bench2_sgcs_sb_user = np.zeros((user_num, subcarrier))
                for userid in range(user_num):
                    for sb in range(subcarrier):
                        bench1 = sgcs(data_bench1_pred[t, :, sb, userid], data_test[t, :, sb, userid])
                        bench1_sgcs_sb_user[userid, sb] = bench1
                        bench2 = sgcs(data_pred[t, :, sb, userid], data_test[t, :, sb, userid])
                        bench2_sgcs_sb_user[userid, sb] = bench2
                bench1_sgcs.append(np.mean(bench1_sgcs_sb_user))
                bench2_sgcs.append(np.mean(bench2_sgcs_sb_user))
            bench1_sgcs_time.append(bench1_sgcs)
            bench2_sgcs_time.append(bench2_sgcs)
            print(bench1_sgcs)
            print(bench2_sgcs)
    sgcs1 = np.mean(bench1_sgcs_time, 0)
    sgcs2 = np.mean(bench2_sgcs_time, 0)
    print(f"last_bench1_sgcs: {sgcs1}")
    print(f"last_bench2_sgcs: {sgcs2}")
