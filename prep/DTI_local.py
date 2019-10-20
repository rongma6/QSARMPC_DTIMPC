import sys
import os
import numpy as np
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem

## convert smiles to fingerprint with length of [explicit], using rdkit to get bit vector
def smiles2fpRdkit(smiles, explicit = 128):
    if len(smiles) == 0:
        fp = '\n'
        print('Warning by RM: empty smiles!')
        arr = np.zeors((1, explicit), dtype = int)
    else:
        ms = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(ms, 2, nBits = explicit)
        arr = np.zeros((1, ), dtype = int)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles2fp(infn, explicit):
    fps = []
    with open(infn, 'r') as infileobj:
        for line in infileobj:
            smiles = line.strip('\n').strip('\r')
            if len(smiles) == 0:
                fp = '\n'
                print('Error', smiles, len(fps))
            else:
                fp = smiles2fpRdkit(smiles, explicit)
            fps.append(fp)
    return fps

if __name__ == "__main__":
    explicit = int(sys.argv[1])
    dir_before_prep = 'prep/DTI_data_before_prep/'
    dir_after_prep = 'prep/DTI_data_after_prep/'
    if not os.path.exists(dir_after_prep):
        os.makedirs(dir_after_prep)
    output_dir = dir_after_prep + 'data_prep/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dps = smiles2fp(dir_before_prep + 'drug_smiles.txt', explicit)
    dpsnp = dps[0].reshape(1, -1)
    for idp in dps[1:]:
        dpsnp = np.vstack((dpsnp, idp.reshape(1, -1)))
    np.save(output_dir + 'finger_rdkit_' + str(explicit) + '.npy', dpsnp)
    
    print('The file mydata/DTI_full/data_prep/finger_rdkit_1024.npy is a copy version of the file from prep/DTI_data_after_prep/data_prep/finger_rdkit_1024.npy if you run python prep/DTI_local.py 1024')