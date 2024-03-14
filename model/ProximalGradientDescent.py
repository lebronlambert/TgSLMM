__author__ = "Xiang Liu"

from utility.roc import roc
import numpy as np

class ProximalGradientDescent:
    def __init__(self, tolerance=0.000001, learningRate=0.001, learningRate2=0.001, prev_residue=9999999999999999.,
                 innerStep1=10, innerStep2=10, progress=0., maxIteration=1000, isRunning=False, shouldStop=False,mode='linear'):
        self.tolerance = tolerance
        self.learningRate = learningRate
        self.learningRate2 = learningRate2
        self.prev_residue = prev_residue
        self.innerStep1 = innerStep1
        self.innerStep2 = innerStep2
        self.progress = progress
        self.maxIteration = maxIteration
        self.isRunning = isRunning
        self.shouldStop = shouldStop
        self.mode=mode

    def stop(self):
        self.shouldStop = True

    def setUpRun(self):
        self.isRunning = True
        self.progress = 0.
        self.shouldStop = False

    def finishRun(self):
        self.isRunning = False
        self.progress = 1.

    def run(self, model, str):
        if self.mode == 'linear':
            self.learningRate = self.learningRate*800
        if str == "tree":
            model.hierarchicalClustering()
            self.learningRate = self.learningRate *1e5
            epoch = 0
            residue = model.cost()
            theta = 1.
            theta_new = 0.
            beta_prev = model.beta
            beta_curr = model.beta
            beta = model.beta
            beta_best = model.beta
            model.initGradientUpdate()
            # diff=self.tolerance*2
            diff = np.inf
            self.maxIteration=1000
            while (epoch < self.maxIteration and diff > self.tolerance and (not self.shouldStop)):
                epoch = epoch + 1
                # if epoch%100==0:
                #     print "times:",epoch
                self.progress = (0. + epoch) / self.maxIteration
                theta_new = 2. / (epoch + 2)
                grad = model.proximal_derivative()
                in_ = beta - 1 / model.getL() * grad
                beta_curr = model.proximal_operator(in_, self.learningRate)
                beta = beta_curr + (1 - theta) / theta * theta_new * (beta_curr - beta_prev)
                beta_prev = beta_curr
                theta = theta_new
                model.updateBeta(beta)
                residue = model.cost()
                diff = abs(self.prev_residue - residue)
                if (residue < self.prev_residue):
                    beta_best = beta
                    self.prev_residue = residue

            model.updateBeta(beta_best)

        elif str == "group":
            self.learningRate = self.learningRate*1e6
            epoch = 0
            residue = model.cost()
            theta = 1.
            theta_new = 0.
            beta_prev = model.beta
            beta_curr = model.beta
            beta = model.beta
            beta_best = model.beta
            diff = self.tolerance * 2
            while (epoch < self.maxIteration and diff > self.tolerance and not self.shouldStop):
                epoch = epoch + 1
                self.progress = (0. + epoch) / self.maxIteration
                theta_new = 2. / (epoch + 3.)
                grad = model.gradient()
                in_ = beta - 1. / model.getL() * grad
                beta_curr = model.proximal_operator(in_, self.learningRate)
                beta = beta_curr + (1 - theta) / theta * theta_new * (beta_curr - beta_prev)
                beta_prev = beta_curr
                theta = theta_new
                model.beta = beta
                residue = model.cost()
                diff = abs(self.prev_residue - residue)
                if (residue < self.prev_residue):
                    beta_best = beta
                    self.prev_residue = residue
            model.beta = beta_best

        else:
            pass
