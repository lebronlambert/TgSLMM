__author__ = 'Haohan Wang'

import sys

sys.path.append('../')

import numpy as np

from utility.dataLoader import loadRealData
from model.sLMM import sLMM


def run(model, cat, reg_min, reg_max):

    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5

    snps, Y, Kva, Kve = loadRealData(cat)
    discoverNum = 50*Y.shape[1]
    B = []

    mode = 'lmm'

    K = np.dot(snps, snps.T)

    lam = 1e-6  # Useless in this case

    slmm_model = sLMM(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode,
                      numintervals=numintervals,
                      model=model, lam=lam, cv_flag=True, reg_min=reg_min, reg_max=reg_max)
    beta_model_lmm = slmm_model.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    B.append(beta_model_lmm)

    if cat == 'g':
        fileHead = '../result/real/group/'
    else:
        fileHead = '../result/real/tree/'

    np.save(fileHead + 'beta_' + cat, B)

if __name__ == '__main__':
    m = sys.argv[1]
    # reg_min = float(sys.argv[1])
    # reg_max = float(sys.argv[2])
    reg_min = float('1e-15')
    reg_max = float('1')

    if m == 't':
        model = 'tree'
    else:
        model = 'group'
    cat = 'alz'
    run(model, cat, reg_min, reg_max)