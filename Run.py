import os

# 设置最大线程
os.environ['NUMEXPR_MAX_THREADS'] = '40'
os.environ['NUMEXPR_NUM_THREADS'] = '30'
import numexpr as ne

ne.set_num_threads(30)
import asgl
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
lambda1_vec = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01]
lambda2_vec = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01]
error_type1 = 'MSE'
error_type2 = 'MSE'


if __name__ == "__main__":
    group_sizes = [np.random.randint(10, 20) for i in range(5)]
    active_groups = [np.random.randint(2) for _ in group_sizes]
    groups = np.concatenate(
        [size * [i] for i, size in enumerate(group_sizes)]
    ).reshape(-1, 1)
    num_coeffs = sum(group_sizes)
    num_datapoints = 10
    noise_std = 20
    # --------------------
    x = np.random.standard_normal((num_datapoints, num_coeffs))
    # print(x)
    # x = np.random.rand(1000, 8)

    window_container = []
    # x_trans
    x_trans, x_rank = asgl.process_x_matrix(x, L)
    # Rank Transformation
    # x_rank = myasgl.process_x_matrix(x, L)[1]

    print('x, L', x.shape, L)
    print('x_trans', x_trans.shape)
    print('x_rank', x_rank.shape)
    y = np.random.rand(10)
    import matplotlib.pyplot as plt
    import numpy as np

    ypoints = y

    plt.plot(ypoints, marker='*')
    plt.show()
    window_container.append([x_trans, x_rank, y])

    window_len = 12
    horizon_len = 1

    x_trans_wind = np.concatenate([i[0] for i in window_container[:window_len]])
    x_rank_wind = np.concatenate([i[1] for i in window_container[:window_len]])
    y_wind = np.concatenate([i[2] for i in window_container[:window_len]])

    for i in window_container[:12]:
        print(len(i))
        print('i[0].shape', i[1].shape)

    x_trans_out = window_container[-horizon_len][0]
    x_rank_out = window_container[-horizon_len][1]
    y_out = window_container[-horizon_len][2]

    print('x_trans_wind', x_trans_wind.shape)
    print('x_trans_out', x_trans_out.shape)
    print('x_rank_wind', x_rank_wind.shape)
    print('y_wind', y_wind.shape)
    print('y_out', y_out.shape)
    myasgl_result = asgl.two_step_agl_main(x=x_trans_wind, x_out=x_trans_out, x_trans=x_rank_wind,
                                             y=y_wind, y_out=y_out, L=L,
                                             lambda1_vec=lambda1_vec, lambda2_vec=lambda2_vec,
                                             error_type1=error_type1, error_type2=error_type2,
                                             criterion='bic', mycores=None)
    print('------------------------------')

    import matplotlib.pyplot as plt
    import numpy as np

    y_prediction = myasgl_result[4]

    plt.plot(y_prediction, marker='o')
    plt.show()
