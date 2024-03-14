__author__ = 'Haohan Wang'

import scipy.optimize as opt
import time

from Lasso import Lasso
from ProximalGradientDescent import ProximalGradientDescent
from GroupLasso import GFlasso
from TreeLasso import TreeLasso
from SCAD import SCAD
from MCP import MCP

from helpingMethods import *
def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / s
class sLMM:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, scale=0, mode='linear',
                 model='lasso', learningRate=1e-5, lam=5,discovernum=50, cv_flag=False, reg_min=1e-2, reg_max=1e5):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.discoverNum = discoverNum
        self.scale = scale
        self.mode = mode
        self.model = model  # 'lasso':regular lasso, 'tree':tree lasso, 'group':group lasso
        self.learningRate = learningRate
        self.lam = lam
        self.discovernum=discovernum
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.cv_flag=cv_flag
        self.lam=lam

    def train(self, X, K, Kva, Kve, y,flag=False):
        time_start = time.time()
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        X0 = np.ones(len(y)).reshape(len(y), 1)

        if self.mode != 'linear': # LMM
            S, U, ldelta0, monitor_nm = self.train_nullmodel(y, K, S=Kva, U=Kve)

            delta0 = scipy.exp(ldelta0)
            Sdi = 1. / (S + delta0)
            Sdi_sqrt = scipy.sqrt(Sdi)
            SUX = scipy.dot(U.T, X)
            SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
            SUy = scipy.dot(U.T, y)
            SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
            SUX0 = scipy.dot(U.T, X0)
            SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
            if flag==True:
                for i in range(0,y.shape[1]):
                    SUy[:,i]=normalize(SUy[:,i])
                SUX=normalize(SUX)
        else: # linear models
            SUX = X
            SUy = y
            ldelta0 = 0
            SUX0 = None

        #beta = self.runLasso(SUX, SUy, self.mode, maxEigen=np.max(Kva),lam_=self.lam)
        #beta=self.cv_train(X=SUX, Y=SUy,regMin=1e-30, regMax=1.0, K=self.discovernum,model=self.model,maxEigen=np.max(Kva))  #1400 alz    #2200 at
        if self.cv_flag:
            beta = self.cv_train(X=SUX, Y=SUy, regMin=self.reg_min, regMax=self.reg_max, K=self.discovernum, model=self.model,
                                 maxEigen=np.max(Kva))
        else:
            beta = self.runLasso(SUX, SUy, self.mode, maxEigen=np.max(Kva),lam_=self.lam)
        time_end = time.time()
        time_diff = time_end - time_start
        print '... finished in %.2fs' % (time_diff)
        return beta

    def cv_train(self,X, Y, regMin=1e-30, regMax=1.0, K=1000,model=None,maxEigen=None):
        filename = str(regMin) + '_'+str(regMax)+'_'+self.model+'.txt'
        f = open(filename, 'a')
        f.write('\nbegin with ' + str(regMin) + ' and ' + str(regMax))
        f.close()
        betaM = None
        iteration = 0
        patience = 100
        ss = []
        time_start = time.time()
        time_diffs = []
        minFactor = 0.5
        maxFactor = 2.0
        while regMin < regMax and iteration < patience:
            iteration += 1
            reg = np.exp((np.log(regMin)+np.log(regMax)) / 2.0)
            coef_=self.runLasso(X,Y,model,maxEigen,reg)
            k = len(np.where(coef_ != 0)[0])
            print "k:",k,"   lam:",reg
            ss.append((reg, k))
            if k < K*minFactor:   # Regularizer too strong
                regMax = reg
                if betaM is None:
                    betaM=coef_
            elif k > K*maxFactor: # Regularizer too weak
                regMin = reg
                if betaM is None:
                    betaM = coef_
            else:
                betaM = coef_
                break
            f = open(filename, 'a')
            f.write("\n# of nonzero: " + str(k) + "current lambda: " + str(reg))
            f.close()
        time_diffs.append(time.time() - time_start)
        return betaM



    def runLasso(self, X, Y, mode, maxEigen,lam_):
        if self.model == 'lasso':
            model = Lasso(lam=lam_, lr=self.learningRate)
            model.fit(X, Y)
            print "lasso"
            return model.getBeta()
        elif self.model == 'tree':
            pgd = ProximalGradientDescent(learningRate=self.learningRate, mode=mode)
            model = TreeLasso(lambda_=lam_, clusteringMethod='single', threhold=1.0, mu=0.1, maxEigen=maxEigen)
            model.setXY(X, Y)
            pgd.run(model, self.model)
            print "tree"
            return model.beta
        elif self.model == 'group':
            pgd = ProximalGradientDescent(learningRate=self.learningRate * 2e4,mode=mode)
            model = GFlasso(lambda_flasso=lam_, gamma_flasso=0.7, mau=0.1)
            # Set X, Y, correlation
            model.setXY(X, Y)
            graph_temp = np.cov(Y.T)
            graph = np.zeros((Y.shape[1], Y.shape[1]))
            for i in range(0, Y.shape[1]):
                for j in range(0, Y.shape[1]):
                    graph[i, j] = graph_temp[i, j] / (np.sqrt(graph_temp[i, i]) * (np.sqrt(graph_temp[j, j])))
                    if (graph[i, j] < 0.618):
                        graph[i, j] = 0
            model.corr_coff = graph
            pgd.run(model, self.model)
            print "group"
            return model.beta
        elif self.model=="mcp":
            clf=MCP()
            clf.setLambda(lam_)
            clf.setLearningRate(self.learningRate)
            clf.fit(X, Y)
            betaM = clf.getBeta()
            print "mcp"
            return betaM
        elif self.model=="scad":
            clf = MCP()
            clf.setLambda(lam_)
            clf.setLearningRate(self.learningRate)
            clf.fit(X, Y)
            betaM = clf.getBeta()
            print "scad"
            return betaM
        else:
            return None

    def hypothesisTest(self, UX, Uy, X, UX0, X0):
        [m, n] = X.shape
        p = []
        for i in range(n):
            if UX0 is not None:
                UXi = np.hstack([UX0 ,UX[:, i].reshape(m, 1)])
                XX = matrixMult(UXi.T, UXi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
                Uyr = Uy - matrixMult(UXi, beta)
                Q = np.dot( Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            else:
                Xi = np.hstack([X0 ,UX[:, i].reshape(m, 1)])
                XX = matrixMult(Xi.T, Xi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
                Uyr = Uy - matrixMult(Xi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
            if -1e10 < ts < 1e10:
                p.append(ps)
            else:
                p.append(1)
        return p

    def train_nullmodel(self, y, K, S=None, U=None):
        self.ldeltamin += self.scale
        self.ldeltamax += self.scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)
        # grid search
        nllgrid = scipy.ones(self.numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(self.numintervals + 1) / (self.numintervals * 1.0) * (self.ldeltamax - self.ldeltamin) + self.ldeltamin
        for i in scipy.arange(self.numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in scipy.arange(self.numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt


        return S, U, ldeltaopt_glob, None
