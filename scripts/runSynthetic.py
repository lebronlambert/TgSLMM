__author__ = 'Haohan Wang'

import sys

sys.path.append('../')

import numpy as np
import time
from model.sLMM import sLMM
from model.BOLTLMM import BOLTLMM
from model.lmm_select import lmm_select
from utility.syntheticDataGeneration import generateData
from utility.roc import roc
from model.LMMCaseControlAscertainment import LMMCaseControl
from utility.helpingMethods import *


def lmm_train( X, Y, beta,s,c,d=0.1,regMin=1e-30, regMax=1.0, K=100 ,maxEigen=None,model="bolt" ):
    BETAs = []
    time_start = time.time()
    time_diffs = []
    score=0
    if model=="bolt":
        # for f2 in [0.5, 0.3, 0.1]:  # fix parameters
        #     for p in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:  # fix parameters
        # f2=0.3
        # p=0.1
        clf=BOLTLMM()
        betaM = np.zeros((X.shape[1], Y.shape[1]))
        print betaM.shape
        for i in range(Y.shape[1]):
            temp = clf.train(X, Y[:, i])
            temp = temp.reshape(temp.shape[0], )
            betaM[:, i] = temp
        score, fp_prc, tp_prc, fpr, tpr = roc(betaM, beta)
        # BETAs.append(betaM)
        BETAs.append(np.abs(betaM))
        print "bolt"
    elif model=="case":

        clf = LMMCaseControl()
        betaM = np.zeros((X.shape[1], Y.shape[1]))
        K = np.dot(X, X.T)
        for i in range(Y.shape[1]):
            print i
            clf.fit(X=X, y=Y[:, i], K=K, Kva=None, Kve=None, mode='lmm')
            betaM[:, i] = clf.getBeta()
        score, fp_prc, tp_prc, fpr, tpr = roc(betaM, beta)
        # BETAs.append(betaM)
        BETAs.append(np.abs(betaM))
        print "case"

    elif model=="select":
        clf = lmm_select()
        betaM = np.zeros((X.shape[1], Y.shape[1]))
        K = np.dot(X, X.T)
        for i in range(Y.shape[1]):
            betaM[:, i] = clf.fit(X=X, y=Y[:, i], K=K, Kva=None, Kve=None, mode='linear')
        temp = np.zeros((X.shape[1],))
        for i in range(X.shape[1]):
            temp[i] = betaM[i, :].sum()
        s = np.argsort(temp)[0:int(X.shape[1] * d)]
        s = list(s)
        s = sorted(s)
        X2 = X[:, s]

        K2 = np.dot(X2, X2.T)
        for i in range(Y.shape[1]):

            betaM[:, i] = clf.fit2(X=X, y=Y[:, i], K=K2, Kva=None, Kve=None, mode='lmm')

        score, fp_prc, tp_prc, fpr, tpr = roc(betaM, beta)

        BETAs.append(np.abs(betaM))
        print "select"


    time_end = time.time()
    time_diff = [time_end - time_start]
    print '... finished in %.2fs' % (time_diff[0])
    return BETAs,score

def run(seed, n, p, g, d, k, sigX, sigY,we,model,simulate=False):
    np.random.seed(seed)
    cv_flag =False
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5
    snps, Y, Kva, Kve, beta_true = generateData(n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, tree=True, test=True)
    K_=len(np.where(beta_true!= 0)[0])
    if simulate==True:
        beta_true2=beta_true/500.
    K = np.dot(snps, snps.T)

    B = []

    mode = 'lmm'
    slmm_model = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='lmm',
                      numintervals=numintervals,
                      model=model,discovernum=K_,cv_flag =cv_flag )
    beta_model_lmm = slmm_model.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    score_model_lmm,fp_prc1,tp_prc1,fpr1,tpr1 = roc(beta_model_lmm, beta_true)
    print score_model_lmm
    B.append(beta_model_lmm)


    slmm_linear = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='linear',
                       numintervals=numintervals,
                       model=model,discovernum=K_,cv_flag =cv_flag)
    beta_model_linear = slmm_linear.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    score_model_linear,fp_prc3,tp_prc3,fpr3,tpr3 = roc(beta_model_linear, beta_true)
    print score_model_linear
    B.append(beta_model_linear)


    slmm_lasso = sLMM(ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode='lmm',
                      numintervals=numintervals,
                      model='lasso',discovernum=K_,cv_flag =cv_flag)
    beta_lasso_lmm = slmm_lasso.train(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve)
    score_lasso_lmm ,fp_prc2,tp_prc2,fpr2,tpr2= roc(beta_lasso_lmm, beta_true)
    print score_lasso_lmm
    B.append(beta_lasso_lmm)







    #bolt_lmm
    beta_bolt_lmm,score = lmm_train(X=snps, Y=Y, beta=beta_true, s=sigX, c=sigY,model="bolt")
    print score
    B.append(beta_bolt_lmm)

    #lmm case
    beta_lmm_case, score = lmm_train(X=snps, Y=Y, beta=beta_true, s=sigX, c=sigY, model="case")
    print score
    B.append(beta_lmm_case)

    #lmm select
    beta_lmm_select, score = lmm_train(X=snps, Y=Y,d=d, beta=beta_true, s=sigX, c=sigY, model="select")
    print score
    B.append(beta_lmm_select)



    ##all plot code
    if simulate:
        if model == 'tree':
            fileHead = '../result/synthetic/tree/'
        else:
            fileHead = '../result/synthetic/group/'

        fileHead = fileHead + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_'+str(we)+'_' + str(seed) + '_'

        np.save(fileHead + 'X', snps)
        np.save(fileHead + 'Y', Y)
        np.save(fileHead + 'beta1', beta_true)
        np.save(fileHead + 'beta2', B)

        from matplotlib import pyplot as plt


        fig = plt.figure(dpi=500)
        ax=fig.add_subplot(1, 1, 1)
        im = ax.imshow(beta_true.T)
        ax.title.set_text('Simulated mapping linkage matrix')
        plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure/simulated_beta_true.png', dpi=300)
        #plt.show()


        ################
        fig = plt.figure(dpi=500)
        ax=fig.add_subplot(4, 1, 1)
        im = ax.imshow(beta_true2.T)
        ax.title.set_text('"ground-truth" Beta vector')
        ax1 = fig.add_subplot(4, 1, 2)
        im1 = ax1.imshow(beta_model_lmm.T)
        ax1.title.set_text('TgSLMM')
        ax2 = fig.add_subplot(4, 1, 3)
        im2 = ax2.imshow(beta_model_linear.T)
        ax2.title.set_text('Tree Lasso')
        ax3 = fig.add_subplot(4, 1, 4)
        im3 = ax3.imshow(beta_lasso_lmm.T)
        ax3.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure/beta_tree_slmm_lmmtree.png', dpi=500)
        #plt.show()


        ##abs
        fig = plt.figure(dpi=500)
        ax=fig.add_subplot(4, 1, 1)
        im = ax.imshow(beta_true2.T)
        ax.title.set_text('"ground-truth" Beta vector')
        ax1 = fig.add_subplot(4, 1, 2)
        im1 = ax1.imshow(abs(beta_model_lmm.T))
        ax1.title.set_text('TgSLMM')
        ax2 = fig.add_subplot(4, 1, 3)
        im2 = ax2.imshow(abs(beta_model_linear.T))
        ax2.title.set_text('Tree Lasoo')
        ax3 = fig.add_subplot(4, 1, 4)
        beta_lasso_lmm=beta_lasso_lmm
        im3 = ax3.imshow(abs(beta_lasso_lmm.T))
        ax3.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure/beta_abs_tree_slmm_lmmtree.png', dpi=500)
        #plt.show()



        y1=snps.dot(beta_model_lmm)
        #print y1.shape
        y2=snps.dot(beta_model_linear)
        #print y2.shape
        y3=snps.dot(beta_lasso_lmm)
        #print y3.shape
        fig = plt.figure(dpi=500,figsize=(20, 15))
        ax=fig.add_subplot(4, 1, 1)
        im = ax.imshow(Y.T)
        ax.title.set_text('"ground-truth" reponses')
        ax1 = fig.add_subplot(4, 1, 2)
        im1 = ax1.imshow(y1.T)
        ax1.title.set_text('TgSLMM')
        ax2 = fig.add_subplot(4, 1, 3)
        im2 = ax2.imshow(y2.T)
        ax2.title.set_text('Tree Lasoo')
        ax3 = fig.add_subplot(4, 1, 4)
        im3 = ax3.imshow(y3.T)
        ax3.title.set_text('sLMM')
        # plt.colorbar(im, orientation='horizontal')
        plt.savefig('./figure/y_tree_slmm_lmmtree2.png', dpi=500)
        #plt.show()


        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(beta_mcp.T)
        ax1.title.set_text('MCP')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(beta_scad.T)
        ax2.title.set_text('SCAD')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(beta_lasso.T)
        ax3.title.set_text('LASSO')
        plt.savefig('./figure_new/beta_mcp_scad_lasso.png', dpi=500)
        # plt.show()

        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(abs(beta_mcp).T)
        ax1.title.set_text('MCP')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(abs(beta_scad).T)
        ax2.title.set_text('SCAD')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(abs(beta_lasso).T)
        ax3.title.set_text('LASSO')
        plt.savefig('./figure_new/beta_abs_mcp_scad_lasso.png', dpi=500)
        # plt.show()

        y1 = snps.dot(beta_mcp)
        y2 = snps.dot(beta_scad)
        y3 = snps.dot(beta_lasso)
        fig = plt.figure(dpi=500, figsize=(20, 15))
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(y1.T)
        ax1.title.set_text('MCP')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(y2.T)
        ax2.title.set_text('SCAD')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(y3.T)
        ax3.title.set_text('LASSO')
        plt.savefig('./figure_new/y_mcp_scad_lasso.png', dpi=500)
        # plt.show()

        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(beta_bolt_lmm.T)
        ax1.title.set_text('BOLT')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(beta_lmm_case.T)
        ax2.title.set_text('CASE')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(beta_lmm_select.T)
        ax3.title.set_text('SELECT')
        plt.savefig('./figure_new/beta_bolt_case_select.png', dpi=500)
        # plt.show()

        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(abs(beta_bolt_lmm).T)
        ax1.title.set_text('BOLT')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(abs(beta_lmm_case).T)
        ax2.title.set_text('CASE')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(abs(beta_lmm_select).T)
        ax3.title.set_text('SELECT')
        plt.savefig('./figure_new/beta_abs_bolt_case_select.png', dpi=500)
        # plt.show()

        y1 = snps.dot(beta_bolt_lmm)
        y2 = snps.dot(beta_lmm_case)
        y3 = snps.dot(beta_lmm_select)
        fig = plt.figure(dpi=500, figsize=(20, 15))
        ax1 = fig.add_subplot(3, 1, 1)
        im1 = ax1.imshow(y1.T)
        ax1.title.set_text('BOLT')
        ax2 = fig.add_subplot(3, 1, 2)
        im2 = ax2.imshow(y2.T)
        ax2.title.set_text('CASE')
        ax3 = fig.add_subplot(3, 1, 3)
        im3 = ax3.imshow(y3.T)
        ax3.title.set_text('SELECT')
        plt.savefig('./figure_new/y_bolt_case_select.png', dpi=500)
        # plt.show()

    else:


        if model == 'tree':
            fileHead = '../result/synthetic/tree/'
        else:
            fileHead = '../result/synthetic/group/'

        fileHead = fileHead + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_'+str(we)+'_' + str(seed) + '_'

        np.save(fileHead + 'X', snps)
        np.save(fileHead + 'Y', Y)
        np.save(fileHead + 'beta1', beta_true)
        np.save(fileHead + 'beta2', B)

def run2(m,c,simulate=False):

    model='tree'
    if m == 't':
        model = 'tree'
    elif m == 'g':
        model = 'group'
    else:
        sys.exit()

    #plot
    if simulate:
        n = 250
        p = 500
        d = 0.05
        g = 10
        k = 50
        sigX = 0.001
        sigY = 1
        seed=0
        we=0.05
        run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)

    else:
        ##synthetic
        n = 1000
        p = 5000
        d = 0.05
        g = 10
        k = 50
        sigX = 0.001
        sigY = 1
        we=0.05
        for seed in [0,1,2,3,4]:
            print seed,"----",c
            if c == '0':
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'n':
                for n in [500,800,1500,2000]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'p':
                for p in [1000, 2000,8000,10000]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'g':
                for g in [2, 5, 20, 50]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we,model=model,simulate=simulate)
            if c == 'k':
                for k in [5, 10, 100, 200]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 's':
                for sigX in [0.0001, 0.0005,0.002, 0.005, 0.01]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'c':
                for sigY in [0.1,0.5, 5, 10]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c =='we':
                for we in [0.001,0.01, 0.1]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'd':
                for d in [0.03,0.06]: #[0.03,0.04,0.06]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)
            if c == 'd2':
                for d in [ 0.01,0.1, 0.5]:  #0.005 all and   all seed 3 not good
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY,we=we, model=model,simulate=simulate)



if __name__ == '__main__':
    # m = sys.argv[1]
    # c = sys.argv[2]
    ## seed=sys.argv[3]
    run2('t','0',simulate=True)
    # run2(m,c,seed)
    # run2('t','n')
    # run2('t','p')
    # run2('t','k')
    # run2('t','s')
    # run2('t','c')
    # run2('t','we')
    # run2('t','g')
    # run2('t','d')
    # run2('t','d2')



#CV_train for the cross_valaidation snum
#mkdir some  file
#print out for you to know which step you are going on now~
#evaluationR => vis_real



