import numpy as np
import pandas as pd
import scipy.linalg as la
import sys
import os

## read from a txt file with sepa as separation, return a ndarray, no need for MPC version
def read_ndarray(dirn, net, sepa):
    inputID = dirn + net + '.txt'
    M = pd.read_table(inputID, sep = sepa, header = None)
    M = M.as_matrix(columns = None)
    return M

## calculate protein similarity matrices in plaintext, no need for MPC version
def jaccard_sim_pub(mat):
    intersect = np.dot(mat, mat.T)
    union = np.sum(mat, axis = 1)
    union = np.reshape(np.tile(union, len(mat)), (len(mat), len(mat)))
    union = union + union.T - intersect + 0.0
    sim = intersect / union
    sim = np.nan_to_num(sim)
    for i in range(sim.shape[0]):
        sim[i, i] = 1.0
    return sim

## RWR for proteins in plaintext, no need for MPC version
def RWR_pub(A, maxiter, restartProb):
    n = len(A)
    # normalize the adjacency matrix
    A = A + 0.0
    P = A / A.sum(axis = 0)
    # Personalized PageRank
    restart = np.eye(n)
    Q = np.eye(n)
    for i in range(maxiter):
        Q_new = (1 - restartProb) * np.dot(P, Q) + restart * restartProb
        delta = np.linalg.norm((Q - Q_new))
        Q = Q_new
        if delta < 1e-6:
            break
    return Q
    
## DCA for proteins in plaintext, no need for MPC version
def DCA_pub(networks, dim, rsp, maxiter):
    P = np.array([])
    for net in networks:
        tQ = RWR_pub(net, maxiter, rsp)
        if P.shape[0] == 0:
            P = np.zeros((tQ.shape[0], 0))
        P = np.hstack((P, tQ))
    nnode = len(P)
    alpha = 1. / nnode
    P = np.log(P + alpha) - np.log(alpha)
    P = np.dot(P, P.T)
    # use SVD to decompose matrix
    U, Sigma, VT = la.svd(P, lapack_driver='gesvd', full_matrices=True)
    sigd = np.dot(np.eye(dim), np.diag(Sigma[:dim]))
    Ud = U[:, :dim]
    # get context-feature matrix, since we use P*PT to get square matrix, we need to sqrt twice
    X = np.dot(Ud, np.sqrt(np.sqrt(sigd)))
    return X

def compute_prot_fea(dim_prot, maxiterpr, restartProb, input_dir):
    ## get protein networks
    # sequence
    proteinSim = read_ndarray(input_dir, 'Similarity_Matrix_Proteins', ' ') / 100.0
    proteinNets = [proteinSim]
    # disease associations
    proteinNetworks = ['mat_protein_disease']
    for iprotein in proteinNetworks:
        iprot = read_ndarray(input_dir, iprotein, ' ')
        iprotSim = jaccard_sim_pub(iprot)
        proteinNets.append(iprotSim)
    ## DCA
    protein_feature = DCA_pub(proteinNets, dim_prot, restartProb, maxiterpr)
    return protein_feature

if __name__ == "__main__":
    prot_dim = int(sys.argv[1])
    maxiterpr = int(sys.argv[2])
    restartProb = float(sys.argv[3])
    dir_before_prep = 'prep/DTI_data_before_prep/'
    dir_after_prep = 'prep/DTI_data_after_prep/'
    if not os.path.exists(dir_after_prep):
        os.makedirs(dir_after_prep)
    output_dir = dir_after_prep + 'data_prep/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prot_fea = compute_prot_fea(prot_dim, maxiterpr, restartProb, dir_before_prep)
    np.save(output_dir + 'public_protein_feature_' + str(prot_dim) + '.npy', prot_fea)
    
    print('The file mydata/DTI_full/data_prep/public_protein_feature_800.npy is a copy version of the file from prep/DTI_data_after_prep/data_prep/public_protein_feature_800.npy if you run python prep/DTI_public.py 800 20 0.5')