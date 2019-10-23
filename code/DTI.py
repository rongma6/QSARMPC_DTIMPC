import numpy as np
import sys,os
sys.path.append(os.getcwd() + '/privpy_library/privpy_lib')
from time import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import privpy as pp
import pnumpy as pnp

globals()['np'] = np
globals()['sys'] = sys
globals()['roc_auc_score'] = roc_auc_score
globals()['average_precision_score'] = average_precision_score
globals()['pp'] = pp
globals()['pnp'] = pnp
globals()['time'] = time
globals()['os'] = os

def myreciprocal(arr, replace=1e8):
    if isinstance(arr, (pp.SFixed, pp.SInt)):
        return pp.reciprocal(arr)
    flag = pnp.abs(arr)<1e-8
    reci = pp.reciprocal(arr)
    reci = flag*pnp.ones(arr.shape)*replace+(1-flag)*reci
    return reci

## read from a txt file with sepa as separation, return a ndarray, no need for MPC version
def read_ndarray(dirn, net, sepa, dtype = 'float64'):
    inputID = dirn + net + '.txt'
    print(inputID)
    M = np.loadtxt(inputID, delimiter = sepa, dtype = dtype)
    return pp.farr(M)
    
## part DTI with labels to DTIs and labels, can be done locally, no need for MPC version, but the returned arrays should be considered as private inputs <-
def part_DTI_label(X):
    x1 = X[:, :2]
    x2 = X[:, 2:]
    print('##'*10,'parti label')
    return x1, x2

##################################################################
##########          calculation for drug features       ##########
##################################################################

## calculate drug similarity matrices, require MPC version <-
def jaccard_sim_po(mat):
    intersect = pnp.dot(mat, pnp.transpose(mat))
    union = pnp.sum(mat, axis = 1)
    union = pnp.reshape(pnp.tile(union, len(mat)), (len(mat), len(mat)))
    union = union + pnp.transpose(union) - intersect + 0.0
    sim = intersect * myreciprocal(union,0.0)
    for i in range(sim.shape[0]):
        sim[i, i] = pp.sfixed(1.0)
    return sim
    
## pair-row dice similarity for the given matrix, require MPC version <-
def dice_sim_matrix_po(X):
    sumX = pnp.sum(X, axis = 1)
    sumX = pnp.reshape(pnp.tile(sumX, len(X)), (len(X), len(X)))
    sumX = sumX + pnp.transpose(sumX)
    cmpX = pnp.zeros((len(X), len(X)))
    for i in range(len(X)):
        cmpX[i] = pnp.sum(X * pnp.reshape(pnp.tile(X[i], len(X)), (len(X), len(X[0]))), axis = 1)
    result = 2 * cmpX * myreciprocal(sumX * sumX, 0.0)
    result = sumX * result
    return result
	
## RWR for drugs, require MPC version <-
def RWR_po(A, maxiter, restartProb):
    n = len(A)
    # normalize the adjacency matrix
    A = A + 0.0
    tmp_var = pnp.sum(A,axis=0)
    tmp_var = myreciprocal(tmp_var,0.0)
    tmp_var = pnp.tile(tmp_var,(A.shape[0],1))
    P = A * tmp_var
    # Personalized PageRank
    restart = pnp.eye(n) * restartProb
    Q = pnp.eye(n)
    for i in range(maxiter):
        Q = (1 - restartProb) * pnp.dot(P, Q) + restart
    return Q

## DCA for drugs, require MPC version <-
def DCA_po(networks, dim, rsp, maxiter, pmiter, log_iter):
    def log_po(x, times):
        tmp = x - 1
        sgn = 1
        result = 0
        for k in range(times):
            result += 1. / (k+1) * sgn * tmp
            tmp *= x - 1
            sgn *= -1
        return result
    
    def power(A, v, it):
        u = v
        for i in range(it):
            v = pnp.dot(A, u)
            l = pnp.dot(pnp.transpose(v), u) * myreciprocal(pnp.dot(pnp.transpose(u), u))
            u = v * myreciprocal(l.flatten()[0])
        return u, l
    
    def hhmul(v, w):
        return v - 2 * pnp.transpose(w).dot(v).flatten()[0]* w
    
    def hhupdate(A, w):
        wA = 2 * pnp.dot(w, pnp.dot(pnp.transpose(w), A))
        wAw = 2 * pnp.dot(pnp.dot(wA, w), pnp.transpose(w))
        A = A - wA - pnp.transpose(wA) + wAw
        return A[1:, 1:]
    
    def pmPCA_po(A, dim, it):
        results = []
        ws = []
        ls = []
        for i in range(dim):
            v = pnp.ones((A.shape[0], 1))
            v, l = power(A, v, it)
            # Prepare a vector w
            w = pnp.zeros(v.shape)
            w[0] = pnp.norm(v)
            w += v
            w = w * myreciprocal(pnp.norm(w))
            # Reduce the matrix dimension
            A = hhupdate(A, w)
            # Reconstruct the eigenvector of original matrix from the current one
            for wp in ws:
                v = pnp.concatenate((pp.farr([[0]]), v))
                v = hhmul(v, wp)
            v = v * myreciprocal(pnp.norm(v))
            results.append(v)
            ws.insert(0, w)
            ls.append(pp.sfixed(l.flatten()[0]))
        return pnp.concatenate(results, axis=1), pp.farr(ls)
    
    P = pp.farr([])
    for net in networks:
        tQ = RWR_po(net, maxiter, rsp)
        if P.shape[0] == 0:
            P = pnp.zeros((tQ.shape[0], 0))
        # concatenate network
        P = pnp.hstack((P, tQ))
    alpha = 0.01
    P = log_po(P + alpha, log_iter) - pnp.log(alpha) # 0 < p <ln(n+1)
    P = pnp.dot(P, pnp.transpose(P)) # 0 < p < n * ln^2(n+1)
    vecs, lambdas = pmPCA_po(P, dim, pmiter)
    sigd = pnp.dot(pnp.eye(dim), pnp.diag(lambdas))
    sigd_sqsq = pnp.sqrt(pnp.sqrt(sigd))
    flag = pnp.abs(sigd)<1e-6
    sigd_sqsq = flag*pnp.zeros(sigd.shape)+(1-flag)*sigd_sqsq
    X = pnp.dot(vecs, sigd_sqsq)
    return X

##########################################################
##########         functions for IMC            ##########
##########################################################

## IMC with known drug-protein interactions, drug features and protein features, require MPC version <-
def IMC_po(Y, D, P, k, lamb, maxiter, gciter):
    '''
    D: real matrix: 708 * (350 or lower) (private)
    P: real matrix: 1512 * (800 or lower) (public)
    Y: bin  matrix: 708 * 1512 (private)
    '''
     
    def multi_dot(arr):
        ans = arr[-1]
        for ii in range(len(arr)-2,-1,-1):
            ans = pp.farr(arr[ii]).dot(ans)
        return ans
    
    ## require MPC version <-
    def grad_full(X, W, H, Y, l):
        '''
        The symbol in this function is consistent with the symbol in paper Large-scale Multi-label Learning with Missing Labels
    
        X: real matrix: 708 * (350 or lower) (private)/ 1512 * (800 or lower) (public)
        W: real matrix: (350/800 or lower) * 125 (private)
        H: real matrix: 708/1512 * 125 (private)
        Y: bin  matrix: 708 * 1512 (private, dense)
        l: (lamb) 1
    
        A = X * W * H^T
        D = A - Y
        ans = X^T * D * H
        '''
        ans = multi_dot([pnp.transpose(X), X, W, pnp.transpose(H), H]) - multi_dot([pnp.transpose(X), Y, H]) + l * W
        return ans
    
    def hess_full(X, W, S, H, Y, l):
        '''
        Only works under square loss function
        '''
        ans = multi_dot([pnp.transpose(X), X, S, pnp.transpose(H), H]) + l * S
        return ans
    
    def fdot(A, B):
        # flatten dot. Regard A and B as long vector
        A = pp.farr(A)
        B = pp.farr(B)
        A = A * (1.0/A.shape[0])
        B = B * (1.0/B.shape[0])
        return pnp.sum(A * B)
    
    def GC(X, W, H, Y, l, iters):
        grad_solver = lambda W: grad_full(X, W, H, Y, l)
        hess_solver = lambda W, S: hess_full(X, W, S, H, Y, l)
        R = - grad_solver(W) + 0.0 
        D = R
        oldR2 = fdot(R, R)
        for t in range(iters):
            hessD = hess_solver(W, D)
            a = oldR2 * pp.reciprocal(fdot(D, hessD)+1e-8)
            W += a * D
            R -= a * hessD
            newR2 = fdot(R, R)
            b = newR2 * pp.reciprocal(oldR2+1e-8)
            D = R + b * D
            oldR2 = newR2
        return W
    
    W = pnp.eye(D.shape[1],k)*0.3  
    H = pnp.eye(P.shape[1],k)*0.3 
    
    updateW = lambda W, H, it: GC(D, W, pnp.dot(P, H), Y, lamb, it)
    updateH = lambda W, H, it: GC(P, H, pnp.dot(D, W), pnp.transpose(Y), lamb, it)
    for i in range(maxiter):
        W = updateW(W, H, gciter)
        H = updateH(W, H, gciter)

        if True:  # log
            Yhat = multi_dot([D, W, pnp.transpose(H), pnp.transpose(P)])
            loss = pnp.norm(Y - Yhat)
    Yhat = multi_dot((D, W, pnp.transpose(H), pnp.transpose(P)))
    return Yhat

def train(data_dir, seed_IMC, DTItrain, maxiterpr, maxiterd, restartProb, dim_drug, dim_prot, imc_k, imc_iter, log_iter, pmiter, explicit, gciter, lamb):
    ### input and output dir name for compact feature learning
    dir_inter = data_dir + 'data_prep/'
    dir_DTIMPC = data_dir + 'data_luo/' 
    ### get drug networks, require MPC version <-
    # structure
    dpsM = np.load(dir_inter + 'finger_rdkit_' + str(explicit) + '.npy') 
    dpsM = pp.farr(dpsM)
    drugSim = dice_sim_matrix_po(dpsM)
    drugNets = [drugSim]
    # interactions / associations
    drugNetworks = ['mat_drug_disease']
    for idrug in drugNetworks:
        idr = read_ndarray(dir_DTIMPC, idrug, ' ', 'int32')
        idrSim = jaccard_sim_po(idr)
        drugNets.append(idrSim)
    #DCA
    protein_feature = np.load(dir_inter + 'public_protein_feature_' + str(dim_prot) + '.npy')
    drug_feature = DCA_po(drugNets, dim_drug, restartProb, maxiterd, pmiter, log_iter)
    np.random.seed(seed_IMC) # <--- random seed to initialize IMC
    Re = IMC_po(DTItrain, drug_feature, protein_feature, k=imc_k, lamb=lamb, maxiter=imc_iter, gciter=gciter)
    return Re

def evaluate(Re, DTItest):
    DTItest = pp.back2plain(DTItest)
    DTItest = DTItest.astype(int)
    testID, testy = part_DTI_label(DTItest)
    pred = [pp.back2plain(Re[i[0]][i[1]]) for i in testID]
    pred = np.reshape(pred,(-1, 1))
    rocs = roc_auc_score(testy, pred)  
    auprs = average_precision_score(testy, pred) 
    return rocs, auprs
    return 0,0

def demo(hyper_param):
    seed = hyper_param['seed']
    suffix = ''
    data_dir = hyper_param['data_dir']
    no = hyper_param['no']
    seed = hyper_param['seed']
    dir_trial = data_dir + 'trial' + str(seed) + suffix + '/'
    metrics = []
    DTItrain = read_ndarray(dir_trial, 'fold' + str(no) + '_train_dense', ' ', 'int32')
    DTItrain = pp.farr(DTItrain)
    Re = train(data_dir, no + seed, DTItrain, hyper_param['maxiterpr'], hyper_param['maxiterd'], 
               hyper_param['restartProb'], hyper_param['dim_drug'], hyper_param['dim_prot'], 
               hyper_param['imc_k'], hyper_param['imc_iter'], hyper_param['log_iter'], hyper_param['pmiter'], 
               hyper_param['explicit'], hyper_param['gciter'], hyper_param['lamb'])

    test1 = read_ndarray(dir_trial, 'fold' + str(no) + '_test1basic', ' ', 'int32')
    auroc, aupr = evaluate(Re, test1)
    metrics.append(auroc)
    metrics.append(aupr)
    
    test9 = read_ndarray(dir_trial, 'fold' + str(no) + '_test9extra', ' ', 'int32')
    test_10 = pnp.vstack((test1, test9))
    auroc, aupr = evaluate(Re, test_10)
    metrics.append(auroc)
    metrics.append(aupr)
     
    testall = read_ndarray(dir_trial, 'fold' + str(no) + '_testallextra', ' ', 'int32')
    test_all = pnp.vstack((test1, testall))
    auroc, aupr = evaluate(Re, test_all)
    metrics.append(auroc)
    metrics.append(aupr)
    Re = pp.back2plain(Re)
    metrics = np.array(metrics)
    return Re, metrics

def load_conf(PATH):
    with open(PATH,'r') as f:
        hyper_param = {}
        for line in f.readlines():
            c  = line.strip().split(' ')
            if c[0] in ['restartProb', 'lamb']:
                hyper_param[c[0]] = float(c[1])
            elif c[0] == 'data_dir':
                hyper_param[c[0]] = c[1]
            else:
                hyper_param[c[0]] = int(c[1])
    return hyper_param

PATH_CONF = "conf/DTI.conf"

hyper_param = load_conf(PATH_CONF)
Re, metrics = demo(hyper_param) 

if role == 'sa':
    np.savetxt("result/Re.txt",Re)
    np.savetxt("result/metrics.txt",metrics)

