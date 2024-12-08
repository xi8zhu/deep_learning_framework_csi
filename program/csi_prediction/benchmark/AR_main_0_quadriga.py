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
    draw_pic = True
    print("hello world!")
    # read the datasets -- h5py
    file = h5py.File('C:\\Users\\user1\\lzh_use\\quadriga_H_CSI_sample_density5.mat')  
    # (2, 32, 12, 1000, 285)
    data = np.array(file['H_CSI']) #mai
    data = data.transpose([3, 4, 2, 1, 0])
    # (1000, 285, 12, 32, 2)
    
    print('finished reading data')
    print(data.shape)
    slot_length, user_num, subcarrier, tx_num, rx_num= data.shape

    user_num = 50
    # AR config
    lag_order = 5
    predict_step = 4 #predict the next 4 slots
    # initialization
    bench1_sgcs_time = []
    bench2_sgcs_time = []
    # Begin AR
    for i in range(5):
        print('begin time is: ', i)
        true_slot_length = int(np.floor(slot_length/5))
        slot = [i + 5 * j for j in range(true_slot_length)]
        full_data = data[slot, :,:,:, 0]
        data_train = full_data[:-predict_step, :, :, :]
        data_test = full_data[-predict_step:, :, :, :]['real'] + 1j * full_data[-predict_step:, :, :, :]['imag']
        # (4, 285, 12, 32)
        data_bench1_pred = np.zeros((predict_step, user_num, subcarrier, tx_num), dtype='complex')
        data_bench2_pred = np.zeros((predict_step, user_num, subcarrier, tx_num), dtype='complex')
        for userid in range(user_num):
            print(userid , end=',')
            for sb in (range(subcarrier)):
                for tx in (range(tx_num)):
                    res_bench1_pred = {}
                    res_ar_pred = {}
                    for j in ['real', 'imag']:
                        seq_ar = data_train[:, userid, sb, tx][j] 
                        res_bench1_pred[j] = np.array([seq_ar[-1]]*predict_step)
                        AR_model = AutoReg(seq_ar, lags=lag_order, trend='n')
                        result = AR_model.fit()
                        seq_start = len(seq_ar)
                        seq_end = len(seq_ar) + predict_step - 1
                        prediction = result.predict(start=seq_start, end=seq_end)
                        res_ar_pred[j] = prediction
                        
                        if draw_pic:
                            plt.figure()
                            plt.scatter(slot[130:-predict_step], full_data[130:-predict_step, userid, sb, tx][j], c = 'blue')
                            plt.scatter(slot[-predict_step:], full_data[-predict_step:, userid, sb, tx][j], c = 'green')
                            one_value_bench1 = seq_ar[-1]
                            bench1 = [one_value_bench1] * 5
                            plt.scatter(slot[-predict_step-1:], bench1, c = 'orange')
                            plt.scatter(slot[-predict_step:], prediction, c = 'red')
                            plt.title('user_id:%s, subcarrier:%s, tx:%s, rx:%s, %s'
                                    %(userid, sb, tx, 0, j))
                            plt.show()

                    data_bench1_pred[:, userid, sb, tx] = res_bench1_pred['real'] + 1j * res_bench1_pred['imag']
                    data_bench2_pred[:, userid, sb, tx] = res_ar_pred['real'] + 1j * res_ar_pred['imag']
        print(f"{i} finished the prediction")
        bench1_sgcs = []
        bench2_sgcs = []
        for t in (range(predict_step)):
            bench1_sgcs_sb_user = np.zeros((user_num, subcarrier))
            bench2_sgcs_sb_user = np.zeros((user_num, subcarrier))
            for userid in range(user_num):
                for sb in range(subcarrier):
                    one_value_sgcs_bench1 = sgcs(data_bench1_pred[t, userid, sb, :], data_test[t, userid, sb, :])
                    bench1_sgcs_sb_user[userid, sb] = one_value_sgcs_bench1
                    one_value_sgcs_bench2 = sgcs(data_bench2_pred[t, userid, sb, :], data_test[t, userid, sb, :])
                    bench2_sgcs_sb_user[userid, sb] = one_value_sgcs_bench2
            bench1_sgcs.append(np.mean(bench1_sgcs_sb_user))
            bench2_sgcs.append(np.mean(bench2_sgcs_sb_user))
        bench1_sgcs_time.append(bench1_sgcs)
        bench2_sgcs_time.append(bench2_sgcs)
        print(bench1_sgcs)
        print(bench2_sgcs)
    sgcs1_t_sb_num = np.mean(bench1_sgcs_time, 0)
    sgcs2_t_sb_num = np.mean(bench2_sgcs_time, 0)
    print(f"sgcs1_t_sb_num: {sgcs1_t_sb_num}")
    print(f"sgcs2_t_sb_num: {sgcs2_t_sb_num}")
