import sys
import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

## read from a txt file with sepa as separation, return a ndarray
def read_ndarray(dirn, net, sepa):
    inputID = dirn + net + '.txt'
    M = pd.read_table(inputID, sep = sepa, header = None)
    M = M.as_matrix(columns = None)
    return M

## given DTI matrix, generate 10-fold CV for 1:1, 1:10, 1:all test dataset (with the same balanced training dataset)
def generate_CV(Y, output_path, seed, no_set = set(range(10)), is_dense = False):
    np.random.seed(seed)
    print('Use random seed', seed, 'to generate train and test datasets.')
    print('Y shape', np.shape(Y))

    whole_positive_index = np.argwhere(Y == 1)
    whole_negative_index = np.argwhere(Y == 0)
    print('# positive samples', len(whole_positive_index))
    print('# non-positive samples', len(whole_negative_index))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # positive random shuffle
    whole_positive_index = np.random.permutation(whole_positive_index)
    num_positive = whole_positive_index.shape[0]
    # positive 10 folds [each fold, 9 train, 1 test]
    # negative random shuffle
    whole_negative_index = np.random.permutation(whole_negative_index)
    # negative sample 1:1 positive, 10 folds [each fold, 9 train, 1 test]
    negative_index_1 = whole_negative_index[: num_positive, :]
    # negative sample another 1:9 positive, 10 folds [only used in 1:10 experiments, each fold, 1 test]
    negative_index_9_extra = whole_negative_index[num_positive : (num_positive*10), :]
    # negative sample all other, 10 folds / all [only used in all-neg experiments, each fold, 1 test]
    negative_index_all_extra = whole_negative_index[num_positive :, :]
    
    kf = StratifiedKFold([1 for i in range(num_positive)], n_folds=10, shuffle=False)
    for no, indexes in enumerate(kf):
        if not no in no_set:
            continue
        train_index, test_index = indexes
        if is_dense:
            train_dense = np.zeros(Y.shape, int)
            for itrain in train_index:
                train_dense[whole_positive_index[itrain, 0], whole_positive_index[itrain, 1]] = 1
            np.savetxt(output_path + 'fold' + str(no) + '_train_dense.txt', train_dense)
        else:
            with open(output_path + 'fold' + str(no) + '_train.txt', 'w') as f:
                for itrain in train_index:
                    f.write(str(whole_positive_index[itrain, 0]) + ' ')
                    f.write(str(whole_positive_index[itrain, 1]) + ' ')
                    f.write('1\n')
                    f.write(str(negative_index_1[itrain, 0]) + ' ')
                    f.write(str(negative_index_1[itrain, 1]) + ' ')
                    f.write('0\n')
        with open(output_path + 'fold' + str(no) + '_test1basic.txt', 'w') as f: # 1:1, 1:10, 1:all, these test data are used
            for itest in test_index:
                f.write(str(whole_positive_index[itest, 0]) + ' ')
                f.write(str(whole_positive_index[itest, 1]) + ' ')
                f.write('1\n')
                f.write(str(negative_index_1[itest, 0]) + ' ')
                f.write(str(negative_index_1[itest, 1]) + ' ')
                f.write('0\n')
                
    kf9 = StratifiedKFold([1 for i in range(negative_index_9_extra.shape[0])], n_folds=10, shuffle=False)
    for no, indexes in enumerate(kf9):
        if not no in no_set:
            continue
        train_index, test_index = indexes
        with open(output_path + 'fold' + str(no) + '_test9extra.txt', 'w') as f: # 1:10, these test data are used
            for itest in test_index:
                f.write(str(negative_index_9_extra[itest, 0]) + ' ')
                f.write(str(negative_index_9_extra[itest, 1]) + ' ')
                f.write('0\n')
                
    kfall = StratifiedKFold([1 for i in range(negative_index_all_extra.shape[0])], n_folds=10, shuffle=False)
    for no, indexes in enumerate(kfall):
        if not no in no_set:
            continue
        train_index, test_index = indexes
        with open(output_path + 'fold' + str(no) + '_testallextra.txt', 'w') as f: # 1:all (case 1), these test data are used
            for itest in test_index:
                f.write(str(negative_index_all_extra[itest, 0]) + ' ')
                f.write(str(negative_index_all_extra[itest, 1]) + ' ')
                f.write('0\n')
                
if __name__ == "__main__":
    suffix = ''
    seed = int(sys.argv[1])
    no = int(sys.argv[2])
    is_dense = (sys.argv[3] == 'dense')
    dir_before_prep = 'prep/DTI_data_before_prep/'
    dir_after_prep = 'prep/DTI_data_after_prep/'
    if not os.path.exists(dir_after_prep):
        os.makedirs(dir_after_prep)
    output_dir = dir_after_prep + 'trial' + str(seed) + suffix + '/'
    Y = read_ndarray(dir_before_prep, 'mat_drug_protein' + suffix, ' ')
    generate_CV(Y, output_dir, seed, no_set = set([no]), is_dense = is_dense)
    print('The files in mydata/DTI_full/trial1/ are a copy version of the files from prep/DTI_data_after_prep/trial1/ if you run python prep/DTI_valid.py 1 0 dense')