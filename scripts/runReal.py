__author__ = 'Haohan Wang'


import sys

sys.path.append('../')

import numpy as np
from model.BOLTLMM import BOLTLMM
from model.lmm_select import lmm_select
from model.LMMCaseControlAscertainment import LMMCaseControl
from model.sLMM import sLMM
from utility.dataLoader import loadRealData


def run(cat,model,lam=0.2):

    cv_flag=True
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5

    snps, Y, Kva, Kve = loadRealData(cat)

    discoverNum = 50
    discoverNum=Y.shape[1]*discoverNum
    B = []

    mode = 'lmm'
    if model=='tree':
        fileHead = '../result/real/tree/'
    else:
        fileHead = '../result/real/group/'

    lam = lam #useless

    K = np.dot(snps, snps.T)

    print lam
    if cat=='at':
        slmm_model = sLMM(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode,
                          numintervals=numintervals,
                          model=model, lam=lam,cv_flag =cv_flag )
        beta_model_lmm = slmm_model.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve,flag=True)
        B.append(beta_model_lmm)

        slmm_lasso = sLMM(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode,
                          numintervals=numintervals,
                          model='lasso', lam=lam,cv_flag =cv_flag)
        beta_lasso_lmm = slmm_lasso.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve,flag=True)
        B.append(beta_lasso_lmm)


        slmm_linear = sLMM(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='linear',
                           numintervals=numintervals,
                           model=model, lam=lam,cv_flag =cv_flag)
        beta_model_linear = slmm_linear.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
        B.append(beta_model_linear)

        lmm_mcp = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='linear',
                       numintervals=numintervals,
                       model="mcp", discovernum=discoverNum, cv_flag=cv_flag)
        beta_mcp = lmm_mcp.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
        B.append(beta_mcp)

        lmm_scad = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='linear',
                        numintervals=numintervals,
                        model='scad', discovernum=discoverNum, cv_flag=cv_flag)
        beta_scad = lmm_scad.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
        B.append(beta_scad)

        slmm_linear = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='linear',
                           numintervals=numintervals,
                           model="lasso", discovernum=discoverNum, cv_flag=cv_flag)
        beta_lasso = slmm_linear.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
        B.append(beta_lasso)

        clf = BOLTLMM()
        betaM = np.zeros((snps.shape[1], Y.shape[1]))
        for i in range(Y.shape[1]):
            temp = clf.train(snps, Y[:, i])
            temp = temp.reshape(temp.shape[0], )
            betaM[:, i] = temp
        B.append(betaM)
        print "bolt"

        clf = LMMCaseControl()
        betaM = np.zeros((snps.shape[1], Y.shape[1]))
        for i in range(Y.shape[1]):
            # print "step================>  ", i
            clf.fit(X=snps, y=Y[:, i], K=K, Kva=None, Kve=None, mode='lmm')
            betaM[:, i] = clf.getBeta()
        B.append(betaM)
        print "case"

        dense=0.1
        clf = lmm_select()
        betaM = np.zeros((snps.shape[1], Y.shape[1]))
        for i in range(Y.shape[1]):
            betaM[:, i] = clf.fit(X=snps, y=Y[:, i], K=K, Kva=None, Kve=None, mode='lmm')
            # print "step: ", i
        temp = np.zeros((snps.shape[1],))
        for i in range(snps.shape[1]):
            temp[i] = betaM[i, :].sum()
        s = np.argsort(temp)[0:int(snps.shape[1] * dense)]
        s = list(s)
        X2 = snps[:, s]
        K2 = np.dot(X2, X2.T)
        for i in range(Y.shape[1]):
            betaM[:, i] = clf.fit2(X=snps, y=Y[:, i], K=K2, Kva=None, Kve=None, mode='lmm')
            # print "step================>  ", i
        B.append(betaM)
        print "select"


    else:#alz mice
        slmm_model = sLMM(discoverNum=discoverNum, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode,
                          numintervals=numintervals,
                          model=model, lam=lam)
        beta_model_lmm = slmm_model.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve, flag=True)
        B.append(beta_model_lmm)

    if model=='tree':
        np.save(fileHead +cat, B)
    else:
        np.save(fileHead +cat, B)

if __name__ == '__main__':
    cat = 'at'#sys.argv[1]
    model= 'tree'#sys.argv[2]
    lam=1e-2
    run(cat,model,lam)

# for plot first evaluationR->vis_real
# cv_flag=True
# discoverNum = 50 * y.shape[1]
#mkdir some necessary files
#BTW in fact the results of BOLT_LMM LTMLM LMM_Select do not change any more~ you could just use the former result.
