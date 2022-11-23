import os

# 设置最大线程
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '14'
import numexpr as ne

ne.set_num_threads(30)

import myasgl
from sklearn.datasets import load_boston
import csv
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def SplitFile(*arg, begin_date, end_date):
    # 对三张表依据year 和 month两列进行分割
    ret = []
    beg_year, beg_month = begin_date // 100, begin_date % 100
    end_year, end_month = end_date // 100, end_date % 100

    for dt in arg:
        if beg_year == end_year:
            tmp = dt[((dt['year'] == beg_year) & (dt['month'] >= beg_month) & (dt['month'] <= end_month))]
        else:
            tmp = dt[((dt['year'] > beg_year) & (dt['year'] < end_year)) | \
                     ((dt['year'] == beg_year) & (dt['month'] >= beg_month)) | \
                     ((dt['year'] == end_year) & (dt['month'] <= end_month))]
        ret.append(tmp)

    if len(arg) == 1:
        return ret[0]
    else:
        return ret


L = 10
# lambda1 = []
lambda1_vec = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
lambda2_vec = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
error_type1 = 'MSE'
error_type2 = 'MSE'
nfolds1 = 5
nfolds2 = 5
###
# Transformation of x:
# 1. Rank Transformate -> x_trans
# 2. Using p function -> P_matrix or x
###

# The number of processing cores
mycores = None

if __name__ == "__main__":

    df = pd.read_csv(r'./month_predictors_cleaned_std.csv')
    df_col_name = df.columns

    #     df = df.drop(columns = ['gAd','AD','Adm','VAHU','RDM','RCA','FVAD','RSGL','DLME','RFI','AnA','ReA','ROO','DP','DPR'])

    window_len = 12
    window_container = []

    df['STKCD'] = df['stkcd'].astype('str').apply(lambda x: x.zfill(6))  # choose lie
    df['year'] = df['TRDMNT'] // 100
    df['month'] = df['TRDMNT'] % 100

    not_used_columns = ['mvlag', 'mv', 'ret', 'stkcd', 'Indcd']
    df = df.drop(columns=not_used_columns)

    begin_date = 200001
    end_date = 201912

    # 向前几步预测？
    horizon_len = 1

    df = SplitFile(df, begin_date=begin_date, end_date=end_date)
    df = df.sort_values(['TRDMNT', 'STKCD'])
    df = df.replace('-', np.nan)

    TRDMNT_list = list(set(df['TRDMNT']))
    TRDMNT_list.sort()

    rolling_times = len(TRDMNT_list) - horizon_len - window_len

    # 111个因子 模型有截距项
    myasgl_result_tilde_beta = np.empty(shape=(0,))
    myasgl_result_error_in = np.empty(shape=(0,))
    myasgl_result_error_out = np.empty(shape=(0,))
    myasgl_result_y_pred_in = np.empty(shape=(0,))
    myasgl_result_y_pred_out = np.empty(shape=(0,))

    #     rolling_current = 0
    for times in range(rolling_times + 1):
        print(f'Now it is the {times}th times of rolling')
        if times == 0:
            begin_temp = TRDMNT_list[0]
            end_temp = TRDMNT_list[0 + window_len + 1]
            df_temp = SplitFile(df, begin_date=begin_temp, end_date=end_temp)

        else:
            end_temp = TRDMNT_list[window_len + times]
            df_temp = SplitFile(df, begin_date=end_temp, end_date=end_temp)

        if times == 0:

            for i, gp in df_temp.groupby('TRDMNT'):

                if not len(window_container) == window_len + 1:
                    print(f'year,month {i}')

                    x_raw = gp.drop(columns=['STKCD', 'TRDMNT', 'ret_lead', 'year', 'month'])
                    x_raw = np.array(x_raw)
                    # Using p function
                    x_trans = myasgl.process_x_matrix(x_raw, L)[0]
                    print('t_shape', x_trans.shape)
                    # Rank Transformation
                    x_rank = myasgl.process_x_matrix(x_raw, L)[1]
                    print('r_shape', x_rank.shape)
                    y = np.array(gp['ret_lead'])
                    window_container.append([x_trans, x_rank, y])


                else:
                    print('win_con_len', len(window_container))
                    x_trans_wind = np.concatenate([i[0] for i in window_container[:window_len]])
                    x_rank_wind = np.concatenate([i[1] for i in window_container[:window_len]])
                    y_wind = np.concatenate([i[2] for i in window_container[:window_len]])

                    x_trans_out = window_container[-horizon_len][0]
                    x_rank_out = window_container[-horizon_len][1]
                    y_out = window_container[-horizon_len][2]

                    myasgl_result = myasgl.two_step_agl_main(x=x_trans_wind, x_out=x_trans_out, x_trans=x_rank_wind,
                                                             y=y_wind, y_out=y_out, L=L,
                                                             lambda1_vec=lambda1_vec, lambda2_vec=lambda2_vec,
                                                             error_type1=error_type1, error_type2=error_type2,
                                                             nfolds1=nfolds1, nfolds2=nfolds2, criterion='bic',
                                                             cv_seed1=None, cv_seed2=None, mycores=mycores)
                    print('Calculation Finished!')

                    myasgl_result_tilde_beta_i = myasgl_result[0]
                    myasgl_result_error_in_i = myasgl_result[1]
                    myasgl_result_error_out_i = myasgl_result[2]
                    # y_prediction in sample
                    myasgl_result_y_pred_in_i = myasgl_result[3]
                    # y_prediction out of sample
                    myasgl_result_y_pred_out_i = myasgl_result[4]

                    if myasgl_result_tilde_beta.size == 0:
                        myasgl_result_tilde_beta = myasgl_result_tilde_beta_i
                    else:
                        myasgl_result_tilde_beta = np.vstack((myasgl_result_tilde_beta,
                                                              myasgl_result_tilde_beta_i))

                    myasgl_result_error_in = np.append(myasgl_result_error_in, myasgl_result_error_in_i)

                    myasgl_result_error_out = np.append(myasgl_result_error_out, myasgl_result_error_out_i)

                    myasgl_result_y_pred_in = np.concatenate((myasgl_result_y_pred_in,
                                                              myasgl_result_y_pred_in_i))

                    myasgl_result_y_pred_out = np.concatenate((myasgl_result_y_pred_out,
                                                               myasgl_result_y_pred_out_i))

        #                 if myasgl_result_y_pred_in.size == 0:
        #                     myasgl_result_y_pred_in = myasgl_result_y_pred_in_i
        #                 else:
        #                     myasgl_result_y_pred_in = np.vstack((myasgl_result_y_pred_in,
        #                                                          myasgl_result_y_pred_in_i))

        #                 if myasgl_result_y_pred_out.size == 0:
        #                     myasgl_result_y_pred_out = myasgl_result_y_pred_out_i
        #                 else:
        #                     myasgl_result_y_pred_out = np.vstack((myasgl_result_y_pred_out,
        #                                                           myasgl_result_y_pred_out_i))

        if times > 0:

            for i, gp in df_temp.groupby('TRDMNT'):
                print(i)

                x_raw = gp.drop(columns=['STKCD', 'TRDMNT', 'ret_lead', 'year', 'month'])
                x_raw = np.array(x_raw)
                # Using p function
                x_trans = myasgl.process_x_matrix(x_raw, L)[0]
                print('rank-transformation_shape_in', x_trans.shape)
                # Rank Transformation
                x_rank = myasgl.process_x_matrix(x_raw, L)[1]
                y = np.array(gp['ret_lead'])
                window_container.append([x_trans, x_rank, y])

            print('win_con_len', len(window_container))
            x_trans_wind = np.concatenate([i[0] for i in window_container[:window_len]])
            print('rank-transformation_shape_in', x_trans_wind.shape)
            x_rank_wind = np.concatenate([i[1] for i in window_container[:window_len]])
            y_wind = np.concatenate([i[2] for i in window_container[:window_len]])

            x_trans_out = window_container[-horizon_len][0]
            print('rank-transformation_shape_out', x_trans_out.shape)
            x_rank_out = window_container[-horizon_len][1]
            y_out = window_container[-horizon_len][2]

            myasgl_result = myasgl.two_step_agl_main(x=x_trans_wind, x_out=x_trans_out, x_trans=x_rank_wind,
                                                     y=y_wind, y_out=y_out, L=L,
                                                     lambda1_vec=lambda1_vec, lambda2_vec=lambda2_vec,
                                                     error_type1=error_type1, error_type2=error_type2,
                                                     nfolds1=nfolds1, nfolds2=nfolds2, criterion='bic',
                                                     cv_seed1=None, cv_seed2=None, mycores=mycores)
            print('Calculation Finished!')

            myasgl_result_tilde_beta_i = myasgl_result[0]
            myasgl_result_error_in_i = myasgl_result[1]
            myasgl_result_error_out_i = myasgl_result[2]
            # y_prediction in sample
            myasgl_result_y_pred_in_i = myasgl_result[3]
            # y_prediction out of sample
            myasgl_result_y_pred_out_i = myasgl_result[4]

            if myasgl_result_tilde_beta.size == 0:
                myasgl_result_tilde_beta = myasgl_result_tilde_beta_i
            else:
                myasgl_result_tilde_beta = np.vstack((myasgl_result_tilde_beta,
                                                      myasgl_result_tilde_beta_i))

            myasgl_result_error_in = np.append(myasgl_result_error_in, myasgl_result_error_in_i)

            myasgl_result_error_out = np.append(myasgl_result_error_out, myasgl_result_error_out_i)

            myasgl_result_y_pred_in = np.concatenate((myasgl_result_y_pred_in,
                                                      myasgl_result_y_pred_in_i))

            myasgl_result_y_pred_out = np.concatenate((myasgl_result_y_pred_out,
                                                       myasgl_result_y_pred_out_i))

        window_container.pop(0)
