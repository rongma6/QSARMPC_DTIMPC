import pandas as pd
import numpy as np
import os

ilist = ['METAB', 'HIVINT', 'CB1', 'DPP4', 'HIVPROT', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN', '3A4', 'LOGD']

dir_before_prep = 'prep/ci500747n_si_002/'
dir_after_prep = 'prep/QSAR_data_after_prep/'
if not os.path.exists(dir_before_prep):
    print('Warning: please download the datasets from https://pubs.acs.org/doi/abs/10.1021/ci500747n (ci500747n_si_002.zip in the Supporting Information section) and unzip the data file under prep/ before running prep/QSAR_prep.py')
    assert False
if not os.path.exists(dir_after_prep):
    os.makedirs(dir_after_prep)

for i in ilist:    
    fn_train = dir_before_prep + i + '_training_disguised.csv'
    fn_test = dir_before_prep + i + '_test_disguised.csv'
    data_train = pd.read_csv(fn_train)
    data_test = pd.read_csv(fn_test)
    num_train = (data_train.values).shape[0]
    num_test = (data_test.values).shape[0]
    num = num_train + num_test
    data = pd.concat([data_train, data_test], ignore_index = True, sort = True)
    data = data.fillna(0)
    X_train = data.values[:num_train, 1:-1]
    y_train = data.values[:num_train, 0]
    X_train = np.array(X_train, dtype = 'int')
    y_train = np.array(y_train, dtype = 'float')
    X_test = data.values[num_train:, 1:-1]
    y_test = data.values[num_train:, 0]
    y_test = np.array(y_test, dtype = 'float')
    X_test = np.array(X_test, dtype = 'int')
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    np.save(dir_after_prep + i + '_Xtrain.npy', X_train)
    np.save(dir_after_prep + i + '_Xtest.npy', X_test)
    np.save(dir_after_prep + i + '_ytrain.npy', y_train)
    np.save(dir_after_prep + i + '_ytest.npy', y_test)
    print(i, X_train.shape, X_test.shape)

print('The files in mydata/QSAR_full/ are a copy version of the METAB dataset from prep/QSAR_data_after_prep/')