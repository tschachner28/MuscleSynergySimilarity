import numpy as np
import pandas as pd
from random import randint
import glob
from script.matrix_factorization import *
from sklearn.decomposition import NMF
import datetime
import matplotlib.pyplot as plt


def rank_determine_helper(repeat_num, A, rank):
    '''
    mean(global VAF)>90% & mean(local VAF) > 80%

    Choose the H corresponding to the highest global VAF:
    The synergy set corresponding to maximum VAF was considered the representative set for a given number of synergies.

    '''
    GLOBAL_VAF = []
    local_vaf = []
    VAF_max = 0
    H_max = 0
    W_max = 0

    for repeat in range(repeat_num):
        rand_num = randint(0, 20)
        model = NMF(n_components=rank, init='random', random_state=rand_num, solver='mu')
        W_ = model.fit_transform(A)
        H_ = model.components_

        gvaf, lvaf = VAF(W_, H_, A)
        if gvaf > VAF_max:
            VAF_max = gvaf
            H_max = H_
            W_max = W_
        GLOBAL_VAF.append(gvaf)
        local_vaf.append(lvaf)
        global_mean = np.mean(np.array(GLOBAL_VAF))
        local_mean = np.mean(local_vaf, axis=0)
    if global_mean > 90 and np.all(local_mean > 80):
        print("rank", rank, " chosen did satisfied criteria")
    else:
        print("rank", rank, " chosen did not satisfy criteria")
    return global_mean, local_mean, VAF_max, H_max, W_max


if __name__ == "__main__":
    match_df = pd.read_csv('../data/referenceData_121120.csv')
    load_data = match_df.loc[(match_df['Task'] == '50%') & (match_df['Su'] != 'c06')]
    load_emg = load_data.loc[:, 'Bicep':'MidTrap'].values

    curr_mean = 0
    VAF_best = 0
    H_best = 0
    W_best = 0
    rank_chosen = 0
    for rank in range(2, 5):
        g_mean, l_mean, VAF_m, H_m, W_m = rank_determine_helper(100, load_emg, rank)
        if np.any(l_mean < 80):
            continue
        elif g_mean - curr_mean >= 3:
            curr_mean = g_mean
            VAF_best = VAF_m
            H_best = H_m
            W_best = W_m
            rank_chosen = rank
        else:
            continue
    print('best number of synergy chosen is: ', rank_chosen)
    print('H_best is:\n', H_best)

    x_axis_names = ['Bicep', 'Tricep lateral',
                    'Anterior deltoid', 'Medial deltoid',
                    'Posterior deltoid', 'Pectoralis major',
                    'Lower trapezius', 'Middle trapezius']
    for i in range(1, rank_chosen+1):
        plt.subplot(2, round(rank_chosen/2), i)
        plt.bar(x_axis_names, H_best[i-1, :])
        plt.xticks(rotation=45, ha='right')
    plt.suptitle('50% Shoulder Abduction Load during reference')
    plt.show()
