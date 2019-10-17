
import privpy as pp
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd() + '/privpy_library/privpy_lib')

import pnumpy as pnp
globals()['pnp'] = pnp

def R2(x, y):
    if (np.size(x) != np.size(y)):
        print('Warning: input vectors for R2 must be same length!')
        return -100.0
    else:
        avx = pnp.mean(x)
        avy = pnp.mean(y)
        num = pnp.sum((x - avx) * (y - avy))
        num = num * num
        denom = pnp.sum((x - avx) * (x - avx)) * pnp.sum((y - avy) * (y - avy))
        res = num * pp.reciprocal(denom)
        return res

def maxabsscaler(x):

    batch = 20
    max_abs = pp.farr([pnp.max(pnp.abs(x[i:i+batch,:]), axis=0) for i in range(0, x.shape[0], batch)])
    if (len(max_abs) == 1):
        max_abs = pp.farr(max_abs)
    else:
        max_abs = pnp.vstack(pp.farr(max_abs))

    max_abs = pnp.max(max_abs, axis = 0)
    m_tmp = pnp.ravel(max_abs)
    
    #max_abs[max_abs == 0.0] = 1.0
    for i in range(len(m_tmp)):
        flag_zero = (m_tmp[i] < 1e-8)
        m_tmp[i] = m_tmp[i]*(1-flag_zero) + 1.0*flag_zero

    max_abs = pnp.reshape(m_tmp, (1, -1))

    return max_abs

def scale(x, max_abs):
    #x = x / np.dot(np.ones((x.shape[0], 1)), max_abs)
    x_ones = pnp.ones((x.shape[0], 1))    
    sca = pnp.dot(x_ones, max_abs)
    tar_shape = x.shape
    #x = x * pnp.reciprocal(sca)
    batch_size = 20
    tmp_x = [(x[i:i+batch_size, :] * pp.reciprocal(sca[i:i+batch_size, :])) for i in range(0, x.shape[0], batch_size)]
    
    if(len(tmp_x) == 1):
        x = pp.farr(tmp_x)
        x = pnp.reshape(x, tar_shape)
    else:
        x = pnp.vstack(tmp_x)
        x = pnp.reshape(x, tar_shape)

    print("!!!!!!!!!1", x.shape)
    return x

def inverse_scale(x, max_abs):
    x_ones = pnp.ones((x.shape[0], 1))
    x = x * pnp.dot(x_ones, max_abs)
    return x

def relu(x):
    return (pnp.abs(x) + x) / 2.0

def relu_derivative(x):
    return (x > 0)

def add_layer(n_in, n_out, activation, p):
    weight = 0.1 * np.random.random((n_in, n_out)) - 0.05
    weight = pp.farr(weight)
    v = pnp.zeros((n_in, n_out))
    bias = pnp.zeros((1, n_out))
    bias_v = pnp.zeros((1, n_out))
    return {'weight': weight, 'bias': bias, 'activation': activation, 'p': p, 'v': v, 'bias_v': bias_v}

def predict(network, inputs, hypers):
    n_layer = len(network)
    batch_size = hypers[6]
    up = inputs.shape[0] - batch_size + 1
    for s in range(0, up, batch_size):
        temp = pp.farr(inputs[s: s + batch_size, :])
        for i in range(n_layer):
            layer = network[i]
            temp = pnp.dot(temp, layer['weight']) + pnp.dot(pnp.ones((temp.shape[0], 1)), layer['bias'])
            if layer['activation'] == 'relu':
                temp = relu(temp)
            else: # 'linear'
                pass
        if s == 0:
            pred = temp
        else:
            pred = pnp.concatenate((pred, temp), axis = 0)
    if up <= 0 or pred.shape[0] < inputs.shape[0]:
        if up <= 0:
            temp = inputs
        else:
            temp = inputs[pred.shape[0]:, :]
        for i in range(n_layer):
            layer = network[i]
            temp = pnp.dot(temp, layer['weight']) + pnp.dot(pnp.ones((temp.shape[0], 1)), layer['bias'])
            if layer['activation'] == 'relu':
                temp = relu(temp)
            else: # 'linear'
                pass
        if up <= 0:
            pred = temp
        else:
            pred = pnp.concatenate((pred, temp), axis = 0)
    return pred

def forward_dropout(network, inputs):
    n_layer = len(network)
    for i in range(n_layer):
        layer = network[i]
        print("-----in------------")
        print("input shape: ", inputs.shape)
        print("weigt: ", layer['weight'].shape)
        outputs_before_act = pnp.dot(inputs, layer['weight']) + pnp.dot(pnp.ones((inputs.shape[0], 1)), layer['bias'])
        layer['inputs'] = inputs
        layer['outputs_before_act'] = outputs_before_act
        if layer['activation'] == 'relu':
            outputs_after_act = relu(outputs_before_act)
            nrow = outputs_after_act.shape[0]
            ncol = outputs_after_act.shape[1]
            mask = (np.random.rand(1, ncol) > layer['p']) / (1 - layer['p'])
            layer['mask'] = mask
            outputs_after_act *= np.dot(np.ones((nrow, 1)), mask)
        else: # 'linear'
            pass
        inputs = outputs_after_act

def backward_dropout(network, expected):
    n_layer = len(network)
    for i in reversed(range(n_layer)):
        layer = network[i]
        if i == n_layer - 1: # linear, outputs_before_act == outputs
            exp_size = pnp.ravel(expected).shape[0]
            layer['delta'] = (layer['outputs_before_act'] - expected) / (exp_size + 0.0)
            layer['gradient'] = pnp.dot(pnp.transpose(layer['inputs']), layer['delta'])
            layer['bias_gradient'] = pnp.dot(pnp.ones((1, layer['inputs'].shape[0])), layer['delta'])
        else:
            nrow = layer['outputs_before_act'].shape[0]
            next = network[i + 1]
            layer['delta'] = pnp.dot(next['delta'], pnp.transpose(next['weight'])) * relu_derivative(layer['outputs_before_act']) * pnp.dot(pnp.ones((nrow, 1)), layer['mask'])
            layer['gradient'] = pnp.dot(pnp.transpose(layer['inputs']), layer['delta'])
            layer['bias_gradient'] = pnp.dot(pnp.ones((1, layer['inputs'].shape[0])), layer['delta'])


def update_weights_dropout(network, lr, momentum, early):

    n_layer = len(network)
    for i in reversed(range(n_layer)):
        layer = network[i]
        layer['v'] = momentum * layer['v'] - lr * layer['gradient']
        layer['weight'] += layer['v'] * (1 - early)
        layer['bias_v'] = momentum * layer['bias_v'] - lr * layer['bias_gradient']
        layer['bias'] += layer['bias_v'] * (1 - early)

def nn(X_train, y_train, X_valid, y_valid, X_test, y_test, seed, hypers):


    #neural network, on the raw activity, with normalization

    
    np.random.seed(seed)
    num_features = X_train.shape[1]
    
    num_l1 = hypers[0]
    num_l2 = hypers[1]
    
    print("==============", hypers)
    
    hidden1 = add_layer(num_features, num_l1, 'relu', hypers[2])
    hidden2 = add_layer(num_l1, num_l2, 'relu', hypers[3])
    output = add_layer(num_l2, 1, 'linear', None)
    network = [hidden1, hidden2, output]
    
    n_epoch = hypers[5]
    lr = hypers[4]
    momentum = 0.5
    batch_size = hypers[6]
    patience = 5
    val_best = 1e9
    nogain = 0
    early = 0
    
    num_train = X_train.shape[0]
    up = num_train - batch_size + 1
    
    # n_epoch, lr, momentum, batch_size, patience, num_train, up is public
    # val_best, nogain, early should be private <-
    
    if num_train < batch_size: # full-batch training
        print('Warning: too small dataset!')
        for epoch in range(n_epoch):
            forward_dropout(network, X_train)
            backward_dropout(network, y_train)
            update_weights_dropout(network, lr, momentum, early)
    else: # mini-batch training
        train_perm = np.array(range(num_train))
        for epoch in range(n_epoch):
            new_perm = np.random.permutation(num_train)
            # new_perm can be public
            train_perm = train_perm[new_perm]
            for s in range(0, up, batch_size):
                # s is public
                X_batch = X_train[train_perm[s: s + batch_size], :]
                y_batch = y_train[train_perm[s: s + batch_size], :]
                forward_dropout(network, X_batch)
                backward_dropout(network, y_batch)
                update_weights_dropout(network, lr, momentum, early)
            
            error = pnp.mean(pnp.square(predict(network, X_valid, hypers) - y_valid))
            # error should be private <-
            
            #print 'epoch', epoch, 'valid loss', error
            # ignore the error - nan
            if error >= val_best:
                nogain += 1
                if nogain > patience:
                    #print 'Early stop at epoch', epoch
                    #break
                    early = 1
            else:
                nogain = 0
                val_best = error
   
    return predict(network, X_test, hypers), predict(network, X_valid, hypers)

         
def nns(X_train, y_train, X_valid, y_valid0, X_test, y_test0, norm_y, nets_num, hypers, seed):

    y_valid = y_valid0
    y_test = y_test0

    seed = (seed - 1) * nets_num + 1
    # seed is public
    print('random seed range (inclusive)', seed, nets_num - 1 + seed)
    for inet in range(nets_num):
        # inet+seed - random seed
        iy_pre, iy_val = nn(X_train, y_train, X_valid, y_valid, X_test, y_test, inet + seed, hypers)
        if inet == 0:
            y_test_pre = iy_pre
            y_valid_pre = iy_val
        else:
            y_test_pre = pnp.hstack([y_test_pre, iy_pre])
            y_valid_pre = pnp.hstack([y_valid_pre, iy_val])
        print(y_test_pre.shape, y_valid_pre.shape)
    
    y_test_pre = pnp.mean(y_test_pre, axis = 1)
    # keepdim
    y_test_pre = pnp.reshape(y_test_pre, (y_test_pre.shape[0], 1))
    y_test = inverse_scale(y_test, norm_y)
    y_test_pre = inverse_scale(y_test_pre, norm_y)
    
    y_valid_pre = pnp.mean(y_valid_pre, axis = 1)
    # keepdim
    y_valid_pre = pnp.reshape(y_valid_pre, (y_valid_pre.shape[0], 1))
    y_valid = inverse_scale(y_valid, norm_y)
    y_valid_pre = inverse_scale(y_valid_pre, norm_y)
    return R2(y_test, y_test_pre), y_test_pre

def run(config):

    data_set = config[1][0]
    data_dir = config[1][1]

    ilist = [data_set]

    hyper = [
        int(config[1][2]),
        int(config[1][3]),
        float(config[1][4]),
        float(config[1][5]),
        float(config[1][6]),
        int(config[1][7]),   # hyper[5] - n_epoch
        int(config[1][8])    # hyper[6] - batch_size
    ]

    ensemble = int(config[1][9])
    seed = int(config[1][10])
    # ilist and hyper are public
    
    for i in ilist:
        X_train = pp.farr(np.load(data_dir + i + "_Xtrain.npy"))
        X_test = pp.farr(np.load(data_dir + i + "_Xtest.npy"))
        y_train = pp.farr(np.load(data_dir + i + "_ytrain.npy"))
        y_test = pp.farr(np.load(data_dir + i + "_ytest.npy"))

        ## X_train, X_test, y_train, y_test, all should be secretly shared <-
        
        print('**************', i, X_train.shape, X_test.shape, y_train.shape, y_test.shape, '***************')
        
        ## require MPC version <-
        
        # val, num_train, ran, random seed, index_perm, those five are public
        val = float(config[1][11])
        num_train = X_train.shape[0]
        ran = int(config[1][12])
        np.random.seed(seed)
        index_perm = np.random.permutation(num_train)
        
        X_train_c = X_train[index_perm,:] #shuffled train data
        y_train_c = y_train[index_perm,:] #keep corresponding
        X_test_c = X_test
        y_test_c = y_test
        
        maxabs_x = maxabsscaler(X_train_c)
        X_train_c = scale(X_train_c, maxabs_x)
        X_test_c = scale(X_test_c, maxabs_x)
        maxabs_y = maxabsscaler(y_train_c)
        y_train_c = scale(y_train_c, maxabs_y)
        y_test_c = scale(y_test_c, maxabs_y)

        # split train set and validation set
        valid_num = int(val * (X_train_c.shape[0]))
        X_valid_c = X_train_c[:valid_num, :]
        y_valid_c = y_train_c[:valid_num, :]
        X_train_c = X_train_c[valid_num:, :]
        y_train_c = y_train_c[valid_num:, :]
        
        # train and test, ensemble number (i.e., 8) is public
        r2test, y_test_pred = nns(X_train_c, y_train_c, X_valid_c, y_valid_c, X_test_c, y_test_c, maxabs_y, ensemble, hyper, ran)
        #if np.isnan(r2test) or np.isinf(r2test):
        #    r2test = 0.0
        
        print('important log')
        print('seed:', ran)
        print('hyper:', hyper)
        print('dataset:', i)

        return r2test, y_test_pred

def load_config(CONFIG_PATH):
    '''
    :param CONFIG_PATH: file path of the config
    :return: config information as pd.DataFrame
    '''
    config = pd.read_csv(CONFIG_PATH, sep=' ', header=None)
    return config


globals()['R2'] = R2
globals()['maxabsscaler'] = maxabsscaler
globals()['scale'] = scale
globals()['inverse_scale'] = inverse_scale
globals()['relu'] = relu
globals()['relu_derivative'] = relu_derivative
globals()['add_layer'] = add_layer
globals()['predict'] = predict
globals()['forward_dropout'] = forward_dropout
globals()['backward_dropout'] = backward_dropout
globals()['nn'] = nn
globals()['update_weights_dropout'] = update_weights_dropout
globals()['run'] = run
globals()['load_config'] = load_config


# PATH - which dataset to train
PATH_CONF = "conf/QSAR.conf"

config = load_config(PATH_CONF)
r2_test, y_test_pred = run(config)

pp.reveal(y_test_pred, 'res1')
pp.reveal(r2_test, 'res2')
