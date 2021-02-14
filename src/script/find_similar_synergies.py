import sys
sys.path.append('/Users/thereseschachner/Desktop/muscle_synergy_torque_accuracy_w3/src/')
sys.path.append('/Users/thereseschachner/Desktop/muscle_synergy_torque_accuracy_w3/src/script/')
sys.path.append('/Users/thereseschachner/Desktop/muscle_synergy_torque_accuracy_w3/src/script/find_similar_synergies.py')
sys.path.append('/Users/thereseschachner/Desktop/muscle_synergy_torque_accuracy_w3/src/script/nmf_implement.py')
from random import randint
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from script.matrix_factorization import *
from script.nmf_implement import *
from scipy.stats import truncnorm, norm
from sklearn.preprocessing import normalize

# Input: An 1D array of CSV files containing a reference data CSV and a matching data CSV
# Returns: H_refs, H_matches, W_ref, W_match
def find_all_Hs(data):
    H_ref = []
    H_match = []
    W_ref = []
    W_match = []
    for mat in data:
        match_df = pd.read_csv(mat)
        #all_participants = ['c02', 'c04', 'c05', 'c07', 'c08', 'c09', 'c10'] # omitted c06 for now
        #all_tasks = ['10%', '30%', '50%']
        # Group by task
        #all_participants = ['c02']
        all_tasks = ['50%']
        for task in all_tasks:
            load_data = match_df.loc[(match_df['Task'] == task) & (match_df['Su'] != 'c06')]
            load_emg = load_data.loc[:, 'Bicep':'MidTrap'].values

            curr_mean = 0
            VAF_best = 0
            H_best = 0
            gmeans_all_ranks = np.zeros([1,3])
            lmeans_all_ranks = np.zeros([3,8])
            Hs_all_ranks = []
            W_best = 0
            rank_chosen = 0
            for rank in range(2, 5):
                g_mean, l_mean, VAF_m, H_m, W_m = rank_determine_helper(100, load_emg, rank)
                gmeans_all_ranks[0,rank-2] = g_mean
                lmeans_all_ranks[rank-2,:] = l_mean
                Hs_all_ranks.append(H_m)
                if np.any(l_mean < 80) or g_mean < 90:
                    continue
                elif g_mean - curr_mean >= 3:
                    curr_mean = g_mean
                    VAF_best = VAF_m
                    H_best = H_m
                    W_best = W_m
                    rank_chosen = rank
                else:
                    continue

            # If none of the ranks have g_mean > 90 and np.all(l_mean) > 80)
            if type(H_best) == int: # still same as its initialized value --> no optimal H has been found yet
                print("entered second round selection")
                if np.any(np.all(lmeans_all_ranks > 80, axis=1)): # if any row/rank has all l_means > 80, pick that one
                    rank_chosen = np.where(np.all(lmeans_all_ranks > 80, axis=1))[0][0] + 2 # lowest rank/row where all l_means > 80
                    H_best = Hs_all_ranks[np.where(np.all(lmeans_all_ranks > 80, axis=1))[0][0]]
                elif np.any(gmeans_all_ranks > 90):
                    rank_chosen = np.where(gmeans_all_ranks > 90)[0][0] + 2 # else, pick lowest rank with g_mean > 90
                    H_best = Hs_all_ranks[np.where(gmeans_all_ranks > 90)[0][0]]
                else:
                    H_best = []
                    print("No H satisfies the criteria.")
                    continue


            print('best number of synergy chosen is: ', rank_chosen)
            print('H_best is:\n', H_best)
            if 'reference' in mat:
                print("Reference")
                print("H: " + str(H_best))
                print("EMG mean weights: " + str(np.mean(W_best, axis=0)))
                H_ref = H_best
                W_ref = W_best
                #x=0
            elif 'match' in mat:
                print("Match")
                print("H: " + str(H_best))
                print("EMG mean weights: " + str(np.mean(W_best, axis=0)))
                H_match = H_best
                W_match = W_best


    # Plot histogram of H_matches[0]
    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 4)
    num_emgs = 8
    for emg_count in range(0, num_emgs):
        #H_matches_values_list = list(H_matches.values())
        axs[int(emg_count/4), emg_count % 4].hist(H_match[:, emg_count], list(np.array(np.arange(0,1,0.01))))
        emg_mean = np.mean(H_match[:,emg_count])
        emg_stdev = np.std(H_match[:,emg_count])
        title_str = 'μ≈' + str(round(emg_mean, 2)) + ', σ≈' + str(round(emg_stdev, 2))
        axs[int(emg_count/4), emg_count % 4].set_title(title_str)
        axs[int(emg_count / 4), emg_count % 4].set_xlabel('EMG H Value')
        axs[int(emg_count / 4), emg_count % 4].set_ylabel('Frequency')
    plt.savefig('H_matches histogram')
    plt.show()

    return H_ref, H_match, W_ref, W_match


def scalar_prod_similarity_normal():
    datasets = np.array(['../data/referenceData_121120.csv', '../data/matchData_121120.csv'])

    # Find Hs for all the datasets
    H_ref, H_match, W_ref, W_match = find_all_Hs(datasets)
    H_ref = normalize(H_ref, norm='l1')
    H_match = normalize(H_match, norm='l1')
    if H_ref == None or H_match == None:
        return

    # Generate the random matrices
    num_emgs = 8
    #H_refs_concat= np.concatenate(list(H_refs.values())) # concatenate all H_refs into one matrix, then find their mean and stdev
    H_refs_emg_means = np.mean(H_ref, axis=0)
    H_refs_emg_stdevs = np.std(H_ref, axis=0)
    H_ref_rand = np.zeros([1000,8])
    for emg_count in range(0, num_emgs):
        H_ref_rand[:,emg_count] = np.random.normal(H_refs_emg_means[emg_count], H_refs_emg_stdevs[emg_count], 1000)
    H_ref_rand[np.where(H_ref_rand < 0)] = 0
    H_ref_rand = normalize(H_ref_rand, norm='l1')

    #H_matches_concat = np.concatenate(list(H_matches.values()))  # concatenate all H_refs into one matrix, then find their mean and stdev
    print("rows in H_matches_concat: " + str(np.shape(H_match)[0]))
    H_matches_emg_means = np.mean(H_match, axis=0)
    H_matches_emg_stdevs = np.std(H_match, axis=0)
    H_matches_rand = np.zeros([1000, 8])

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 4)
    for emg_count in range(0, num_emgs):
        H_matches_rand[:,emg_count] = np.random.normal(H_matches_emg_means[emg_count], H_matches_emg_stdevs[emg_count], 1000)
    H_matches_rand[np.where(H_matches_rand < 0)] = 0
    H_matches_rand = normalize(H_matches_rand, norm='l1')
    # Create plot
    for emg_count in range(0, num_emgs):
        axs[int(emg_count/4), emg_count % 4].hist(H_matches_rand[:, emg_count], list(np.array(np.arange(0,1,0.01))))
        title_str = 'μ≈' + str(round(H_matches_emg_means[emg_count], 2)) + ', σ≈' + str(round(H_matches_emg_stdevs[emg_count], 2))
        axs[int(emg_count/4), emg_count % 4].set_title(title_str)
        axs[int(emg_count / 4), emg_count % 4].set_xaxis('EMG H Value')
        axs[int(emg_count / 4), emg_count % 4].set_yaxis('Frequency')
    # Plot the histogram of the random values
    plt.savefig('H_matches_rand histogram (normal distrib)')
    plt.show()


    # Calculate scalar products and store them in the dictionary scalar_prods
    # key is tuple of format (href, href row, hmatch, hmatch row)
    # value is scalar product of href row (1 x 8) and hmatch row ( 1 x 8)
    scalar_prods = {}
    for i in range(0, H_ref.shape[0]):
        for j in range(0, H_match.shape[0]):
            #scalar_prods[(href, i, hmatch, j)] = np.dot(H_refs[href][i,:], H_matches[hmatch][j,:])
            scalar_prods[("H_ref", i, "H_match", j)] = np.dot(H_ref[i, :], H_match[j, :])

    # Add scalar products of H_ref_rand and H_match_rand to the dictionary
    # key is of form ("H_ref_rand", H_ref row, "H_match_rand", H_match row)
    for i in range(0, H_ref_rand.shape[0]):
        for j in range(0, H_matches_rand.shape[0]):
            scalar_prods[("H_ref_rand", i, "H_match_rand", j)] = np.dot(H_ref_rand[i,:], H_matches_rand[j,:])

    # Sort scalar_prods in ascending order by scalar product and identify the 95th percentile
    sorted_scalar_prods = sorted(scalar_prods.items(), key = lambda kv:(kv[1], kv[0]))
    percentile95_index = int(0.95*len(sorted_scalar_prods))
    keys_percentile95_and_above = list(sorted_scalar_prods)[percentile95_index:]
    participant_synergy_pairs = [pair for pair in keys_percentile95_and_above if pair[0][0] != "H_ref_rand"] # participant data, not random data
    print("participant_synergy_pairs: " + str(participant_synergy_pairs))
    syn_pairs_same_participant_and_task = [pair for pair in participant_synergy_pairs if pair[0][0] == pair[0][2]] # if both correspond to same participant and same task
    # choose rank number of synergy pairs with the highest scalar product if the syn_pairs_same_participant_and_task >= rank
    rank = np.shape(H_match)[0]
    if rank < len(syn_pairs_same_participant_and_task):
        syn_pairs_same_participant_and_task = syn_pairs_same_participant_and_task[-rank:]
    print("syn_pairs_same_participant_and_task: " + str(syn_pairs_same_participant_and_task))
    #percent_same_participant_and_task = ((len(syn_pairs_same_participant_and_task) / len(participant_synergy_pairs)) * 100) if len(participant_synergy_pairs) != 0 else 'nan'
    #print("percent_same_participant_and_task: " + str(percent_same_participant_and_task))


def scalar_prod_similarity_truncnorm():
    datasets = np.array(['../data/referenceData_121120.csv', '../data/matchData_121120.csv'])

    # Find Hs for all the datasets
    H_refs, H_matches = find_all_Hs(datasets)
    if H_refs == None or H_matches == None:
        return

    # Generate the random matrices
    num_emgs = 8
    H_refs_concat= np.concatenate(list(H_refs.values())) # concatenate all H_refs into one matrix, then find their mean and stdev
    H_refs_emg_means = np.mean(H_refs_concat, axis=0)
    H_refs_emg_stdevs = np.std(H_refs_concat, axis=0)
    H_refs_emg_mins = np.min(H_refs_concat, axis=0)
    H_refs_emg_maxes = np.max(H_refs_concat, axis=0)
    H_ref_rand = np.zeros([1000,8])
    for emg_count in range(0, num_emgs):
        H_ref_rand[:,emg_count] = truncnorm.rvs((H_refs_emg_mins[emg_count] - H_refs_emg_means[emg_count]) / H_refs_emg_stdevs[emg_count],
                                                (H_refs_emg_maxes[emg_count] - H_refs_emg_means[emg_count]) / H_refs_emg_stdevs[emg_count],
                                                loc=H_refs_emg_means[emg_count], scale=H_refs_emg_stdevs[emg_count], size=1000)
    H_ref_rand[np.where(H_ref_rand < 0)] = 0
    H_ref_rand = normalize(H_ref_rand, norm='l1')

    H_matches_concat = np.concatenate(list(H_matches.values()))  # concatenate all H_refs into one matrix, then find their mean and stdev
    print("rows in H_matches_concat: " + str(np.shape(H_matches_concat)[0]))
    H_matches_emg_means = np.mean(H_matches_concat, axis=0)
    H_matches_emg_stdevs = np.std(H_matches_concat, axis=0)
    H_matches_emg_mins = np.min(H_matches_concat, axis=0)
    H_matches_emg_maxes = np.max(H_matches_concat, axis=0)
    H_matches_rand = np.zeros([1000, 8])

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 4)
    for emg_count in range(0, num_emgs):
        #H_matches_rand[:,emg_count] = np.random.normal(H_matches_emg_means[emg_count], H_matches_emg_stdevs[emg_count], 1000)
        H_matches_rand[:, emg_count] = truncnorm.rvs(
            (H_matches_emg_mins[emg_count] - H_matches_emg_means[emg_count]) / H_matches_emg_stdevs[emg_count],
            (H_matches_emg_maxes[emg_count] - H_matches_emg_means[emg_count]) / H_matches_emg_stdevs[emg_count],
            loc=H_matches_emg_means[emg_count], scale=H_matches_emg_stdevs[emg_count], size=1000)
    H_matches_rand[np.where(H_matches_rand < 0)] = 0
    H_matches_rand = normalize(H_matches_rand, norm='l1')

    for emg_count in range(0, num_emgs):
            axs[int(emg_count/4), emg_count % 4].hist(H_matches_rand[:, emg_count], list(np.array(np.arange(0,1,0.01))))
            title_str = 'μ≈' + str(round(H_matches_emg_means[emg_count], 2)) + ', σ≈' + str(round(H_matches_emg_stdevs[emg_count], 2))
            axs[int(emg_count/4), emg_count % 4].set_title(title_str)
            axs[int(emg_count / 4), emg_count % 4].set_xaxis('EMG H Value')
            axs[int(emg_count / 4), emg_count % 4].set_yaxis('Frequency')
    # Plot the histogram of the random values
    plt.savefig('H_matches_rand histogram (truncnorm distrib)')
    plt.show()


    # Calculate scalar products and store them in the dictionary scalar_prods
    # key is tuple of format (href, href row, hmatch, hmatch row)
    # value is scalar product of href row (1 x 8) and hmatch row ( 1 x 8)
    scalar_prods = {}
    for href in H_refs.keys():
        for hmatch in H_matches.keys():
            for i in range(0, H_refs[href].shape[0]):
                for j in range(0, H_matches[hmatch].shape[0]):
                    #scalar_prods[(href, i, hmatch, j)] = np.dot(H_refs[href][i,:], H_matches[hmatch][j,:])
                    scalar_prods[("H_ref", i, "H_match", j)] = np.dot(H_refs[href][i, :], H_matches[hmatch][j, :])

    # Add scalar products of H_ref_rand and H_match_rand to the dictionary
    # key is of form ("H_ref_rand", H_ref row, "H_match_rand", H_match row)
    for i in range(0, H_ref_rand.shape[0]):
        for j in range(0, H_matches_rand.shape[0]):
            scalar_prods[("H_ref_rand", i, "H_match_rand", j)] = np.dot(H_ref_rand[i,:], H_matches_rand[j,:])

    # Sort scalar_prods in ascending order by scalar product and identify the 95th percentile
    sorted_scalar_prods = sorted(scalar_prods.items(), key = lambda kv:(kv[1], kv[0]))
    percentile95_index = int(0.95*len(sorted_scalar_prods))
    keys_percentile95_and_above = list(sorted_scalar_prods)[percentile95_index:]
    participant_synergy_pairs = [pair for pair in keys_percentile95_and_above if pair[0][0] != "H_ref_rand"] # participant data, not random data
    print("participant_synergy_pairs: " + str(participant_synergy_pairs))
    syn_pairs_same_participant_and_task = [pair for pair in participant_synergy_pairs if pair[0][0] == pair[0][2]] # if both correspond to same participant and same task
    # choose rank number of synergy pairs with the highest scalar product if the syn_pairs_same_participant_and_task >= rank
    rank = np.shape(H_matches_concat)[0]
    if rank < len(syn_pairs_same_participant_and_task):
        syn_pairs_same_participant_and_task = syn_pairs_same_participant_and_task[-rank:]
    print("syn_pairs_same_participant_and_task: " + str(syn_pairs_same_participant_and_task))
    #percent_same_participant_and_task = ((len(syn_pairs_same_participant_and_task) / len(participant_synergy_pairs)) * 100) if len(participant_synergy_pairs) != 0 else 'nan'
    #print("percent_same_participant_and_task: " + str(percent_same_participant_and_task))

def scalar_prod_similarity_uniform():
    datasets = np.array(['../data/referenceData_121120.csv', '../data/matchData_121120.csv'])

    # Find Hs for all the datasets
    H_ref, H_match, W_ref, W_match = find_all_Hs(datasets)
    H_ref = normalize(H_ref, norm='l1')
    H_match = normalize(H_match, norm='l1')
    print("H_refs: " + str(H_ref))
    print("H_matches: " + str(H_match))
    if type(H_ref) == list or type(H_match) == list: # still [], the initialized value
        return

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 2)
    """
    x_axis_names = ['Bicep', 'Tricep lateral',
                    'Anterior deltoid', 'Medial deltoid',
                    'Posterior deltoid', 'Pectoralis major',
                    'Lower trapezius', 'Middle trapezius']
    """
    x_axis_names = ['Bicep', 'Tri. lat.',
                    'Ant. delt.', 'Med. delt.',
                    'Post. delt.', 'Pect. major',
                    'Lower trap.', 'Mid. trap.']
    x = np.arange(0,8)
    #H_matches_values = list(H_matches.values())
    for syn in range(0, H_match.shape[0]):
        axs[int(syn/2), syn % 2].bar(x, H_match[syn,:])
        axs[int(syn/2), syn % 2].set_xticks(np.arange(0,8))
        axs[int(syn/2), syn % 2].set_xticklabels(x_axis_names, rotation=45)
        axs[int(syn/2), syn % 2].set_title('Row ' + str(syn))
        axs[int(syn/2), syn % 2].set_xlabel('EMG')
        axs[int(syn/2), syn % 2].set_ylabel('H Value')
    #plt.subplots_adjust(bottom=0.1, right=0.1, top=0.9)
    plt.tight_layout()
    plt.savefig('H_match for uniform distribution')
    plt.show()

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 2)
    x = np.arange(0, 8)
    #H_refs_values = list(H_refs.values())
    for syn in range(0, H_ref.shape[0]):
        axs[int(syn / 2), syn % 2].bar(x, H_ref[syn, :])
        axs[int(syn / 2), syn % 2].set_xticks(np.arange(0, 8))
        axs[int(syn / 2), syn % 2].set_xticklabels(x_axis_names, rotation=45)
        axs[int(syn / 2), syn % 2].set_title('Row ' + str(syn))
        axs[int(syn / 2), syn % 2].set_xlabel('EMG')
        axs[int(syn / 2), syn % 2].set_ylabel('H Value')
    # plt.subplots_adjust(bottom=0.1, right=0.1, top=0.9)
    plt.tight_layout()
    plt.savefig('H_ref for uniform distribution')
    plt.show()


    # Generate the random matrices, using a uniform distribution with points ranging from mean-stdev to mean+stdev
    num_emgs = 8
    #H_refs_concat= np.concatenate(list(H_refs.values())) # concatenate all H_refs into one matrix, then find their mean and stdev
    H_refs_emg_means = np.mean(H_ref, axis=0)
    #H_refs_emg_stdevs = np.std(H_refs_concat, axis=0)
    H_refs_emg_mins = np.min(H_ref, axis=0)
    H_refs_emg_maxes = np.max(H_ref, axis=0)

    H_ref_rand = np.zeros([1000,8])
    for emg_count in range(0, num_emgs):
        lower_bound = H_refs_emg_mins[emg_count]-0.2 if H_refs_emg_mins[emg_count]-0.2 > 0 else 0
        H_ref_rand[:,emg_count] = np.random.uniform(lower_bound, H_refs_emg_maxes[emg_count]+0.2, 1000)
    H_ref_rand = normalize(H_ref_rand, norm='l1')

    #H_matches_concat = np.concatenate(list(H_matches.values()))  # concatenate all H_refs into one matrix, then find their mean and stdev
    print("rank: " + str(np.shape(H_match)[0]))
    H_matches_emg_means = np.mean(H_match, axis=0)
    H_matches_emg_stdevs = np.std(H_match, axis=0)
    H_matches_emg_mins = np.min(H_match, axis=0)
    H_matches_emg_maxes = np.max(H_match, axis=0)
    H_matches_rand = np.zeros([1000, 8])

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, axs = plt.subplots(2, 4)
    for emg_count in range(0, num_emgs):
        lower_bound = H_matches_emg_mins[emg_count]-0.2 if H_matches_emg_mins[emg_count]-0.2 > 0 else 0
        H_matches_rand[:, emg_count] = np.random.uniform(lower_bound, H_matches_emg_maxes[emg_count]+0.2, 1000)
    H_matches_rand = normalize(H_matches_rand, norm='l1') # Normalize the random values
    
    # Plot the histogram of the random values
    for emg_count in range(0, num_emgs):
        axs[int(emg_count/4), emg_count % 4].hist(H_matches_rand[:, emg_count], list(np.array(np.arange(0,1,0.01))))
        title_str = 'μ≈' + str(round(H_matches_emg_means[emg_count], 2)) + ', σ≈' + str(round(H_matches_emg_stdevs[emg_count], 2))
        axs[int(emg_count/4), emg_count % 4].set_title(title_str)
        axs[int(emg_count / 4), emg_count % 4].set_xlabel('EMG H Value')
        axs[int(emg_count / 4), emg_count % 4].set_ylabel('Frequency')

    plt.savefig('H_matches_rand histogram (uniform distrib)')
    plt.show()


    #Plot 2 random Hs
    #plt.figure()
    fig = plt.figure(figsize = (10, 5))
    #ax1 = f1.add_subplot(111)
    x_axis_labels = ['Bicep', 'Tricep lateral',
                    'Anterior deltoid', 'Medial deltoid',
                    'Posterior deltoid', 'Pectoralis major',
                    'Lower trapezius', 'Middle trapezius']
    x = np.arange(0,8)
    plt.bar(x, H_matches_rand[500,:])
    plt.xlabel('Muscle')
    plt.ylabel('H Value')
    plt.xticks(x, x_axis_labels)
    plt.title('H_matches_rand Row 500')
    plt.savefig('H_matches_rand Row 500')
    plt.show()


    # Calculate scalar products and store them in the dictionary scalar_prods
    # key is tuple of format (href, href row, hmatch, hmatch row)
    # value is scalar product of href row (1 x 8) and hmatch row ( 1 x 8)
    scalar_prods = {}
    for i in range(0, H_ref.shape[0]):
        for j in range(0, H_match.shape[0]):
            #scalar_prods[(href, i, hmatch, j)] = np.dot(H_refs[href][i, :], H_matches[hmatch][j, :])
            scalar_prods[("H_ref", i, "H_match", j)] = np.dot(H_ref[i,:], H_match[j,:])

    # Add scalar products of H_ref_rand and H_match_rand to the dictionary
    # key is of form ("H_ref_rand", H_ref row, "H_match_rand", H_match row)
    for i in range(0, H_ref_rand.shape[0]):
        for j in range(0, H_matches_rand.shape[0]):
            scalar_prods[("H_ref_rand", i, "H_match_rand", j)] = np.dot(H_ref_rand[i,:], H_matches_rand[j,:])

    # Sort scalar_prods in ascending order by scalar product and identify the 95th percentile
    sorted_scalar_prods = sorted(scalar_prods.items(), key = lambda kv:(kv[1], kv[0]))
    percentile95_index = int(0.95*len(sorted_scalar_prods))
    keys_percentile95_and_above = list(sorted_scalar_prods)[percentile95_index:]
    participant_synergy_pairs = [pair for pair in keys_percentile95_and_above if pair[0][0] != "H_ref_rand"] # participant data, not random data
    print("participant_synergy_pairs: " + str(participant_synergy_pairs))
    #syn_pairs_same_participant_and_task = [pair for pair in participant_synergy_pairs if pair[0][0] == pair[0][2]] # if both correspond to same participant and same task
    # choose rank number of synergy pairs with the highest scalar product if the syn_pairs_same_participant_and_task >= rank
    rank = np.shape(H_match)[0]

    """
    matches_dict = {} # key: H_ref row. value: (H_match row, scalar product)
    for i in range(0,rank):
        matches_dict[i] = [(pair[0][3], pair[1]) for pair in participant_synergy_pairs if pair[0][1] == i][-1] # pick the match for a given row that appears last since it's in the highest percentile
        if matches_dict[i] == None:
            print("Match for row " + str(i) + ": " + "None")
        else:
            match_for_i = matches_dict[i][0]
            W_ref_means = np.mean(W_ref, axis=0)
            W_match_means = np.mean(W_match, axis=0)
            print("Match for row " + str(i) + ": " + str(match_for_i) + ", with H_ref: " + str(H_ref[i,:]) + ", W_ref: " + str(W_ref_means[i]) + ", H_match: " + str(H_match[match_for_i,:]) + ", W_match: " + str(W_match_means[match_for_i]) + ", and scalar product: " + str(matches_dict[i][1]))
    print("All synergy pairs: " + str(participant_synergy_pairs))
    """
    matches_dict = {} # key: H_ref row. value: H_match row
    H_refs_rows_matched = []
    H_matches_rows_matched = []
    W_ref_means = np.mean(W_ref, axis=0)
    W_match_means = np.mean(W_match, axis=0)
    for i in range(0, len(participant_synergy_pairs)): # iterate from last element (highest scalar product) to first element (lowest scalar product)
        current_H_ref_row = participant_synergy_pairs[len(participant_synergy_pairs) - 1 - i][0][1]
        current_H_match_row = participant_synergy_pairs[len(participant_synergy_pairs) - 1 - i][0][3]
        current_scalar_prod = participant_synergy_pairs[len(participant_synergy_pairs) - 1 - i][1]
        if current_H_ref_row not in H_refs_rows_matched and current_H_match_row not in H_matches_rows_matched:
            H_refs_rows_matched.append(current_H_ref_row)
            H_matches_rows_matched.append(current_H_match_row)
            matches_dict[current_H_ref_row] = current_H_match_row
            print("Match for H_ref row " + str(current_H_ref_row) + ": H_match row " + str(current_H_match_row) + ", with H_ref: " + str(np.round(H_ref[current_H_ref_row,:], 3)) + ", W_ref: " + str(np.round(W_ref_means[current_H_ref_row], 3)) + ", H_match: " + str(np.round(H_match[current_H_match_row,:], 3)) + ", W_match: " + str(np.round(W_match_means[current_H_match_row], 3)) + ", and scalar product: " + str(np.round(current_scalar_prod, 3)))



    #if rank < len(syn_pairs_same_participant_and_task):
    #    syn_pairs_same_participant_and_task = syn_pairs_same_participant_and_task[-rank:]
    #print("syn_pairs_same_participant_and_task: " + str(syn_pairs_same_participant_and_task))
    #print("syn_pairs_same_participant_and_task: " + str(syn_pairs_same_participant_and_task))
    #percent_same_participant_and_task = ((len(syn_pairs_same_participant_and_task) / len(participant_synergy_pairs)) * 100) if len(participant_synergy_pairs) != 0 else 'nan'
    #print("percent_same_participant_and_task: " + str(percent_same_participant_and_task))


if __name__ == "__main__":
    #scalar_prod_similarity_normal()
    #scalar_prod_similarity_truncnorm()
    scalar_prod_similarity_uniform()