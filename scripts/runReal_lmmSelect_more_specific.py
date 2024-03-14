__author__ = 'Xiang Liu'

import sys

sys.path.append('../')

import numpy as np

from model.lmm_select import lmm_select
from utility.dataLoader import loadRealData


def runlmmselect(X, Y,dense):

    print "begin"
    clf = lmm_select()
    # print X.shape[1],Y.shape[1]
    betaM = np.zeros((X.shape[1], Y.shape[1]))
    betaM2 = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        print "step: ", i
        temp = np.zeros((X.shape[1],))
        for j in range(X.shape[1]):
            x=np.array(X[:,j])
            x=x.reshape(x.shape[0],1)
            K=np.dot(x,x.T)
            betaM[j, i] = clf.fit(X=x, y=Y[:, i], K=K, Kva=None, Kve=None, mode='linear')
            temp[j] = betaM[j,i]

        s = np.argsort(temp)[0:int(X.shape[1] *dense)]
        s = list(s)
        print s
        X2 = X[:, s]
        K2 = np.dot(X2, X2.T)

        betaM2[:, i] = clf.fit2(X=X, y=Y[:, i], K=K2, Kva=None, Kve=None, mode='lmm')


    return betaM2


def run(cat,dense):

    snps, Y, Kva, Kve = loadRealData(cat)

    B = []

    print "run lmm select for at ready"

    beta_at_lmm_select=runlmmselect(snps,Y,dense)


    B.append(beta_at_lmm_select)
    print "run lmm select for at done"

    fileHead = '../result/real/tree/'

    np.save(fileHead + 'beta_lmm_select_more_specific_'+cat , B)

if __name__ == '__main__':
    cat = 'at'
    dense=0.1
    run(cat,dense)

