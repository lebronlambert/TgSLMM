__author__ = 'Haohan Wang'

import numpy as np

from alzDataFunctions import *

modelNum = 9

def evaluate():
    snpNames = [line.strip().split('\t')[1] for line in open('../result/real/alz/marker.txt')]
    #print snpNames[0:100]
    #print len(snpNames)

    #weights = np.load('../result/real/weights_alz.npy')
    #weights=np.load('/home/miss-iris/Desktop/lmmn-g/2_beta_alz.npy')
    weights = np.load('/home/miss-iris/Desktop/lmmn-g/n_beta_alz.npy')
    print weights.shape
    i=0
    print len(np.where(weights[i] != 0)[0])
    label = getLabels_alz()
    # print label[0:100]
    ind_l = np.where(label!=0)[0].tolist()
    # print len(ind_l)
    # print weights.shape
    weight = weights[i][:,-1]
    n = len(np.where(weight!=0)[0])
    ind = np.argsort(np.abs(weight))[::-1][:99]
    # print np.argsort(np.abs(weight))
    # print np.argsort(np.abs(weight))[::-1]
    # print ind
    #print ind

    snps = []
    for i in ind.tolist():
        snps.append(snpNames[i])
        if i in ind_l:
            print i

    #f = open('../result/alz20.csv', 'w')
    f = open('/home/miss-iris/Desktop/lmmn-g/n_beta_alz.csv','w')
    f.writelines('Rank,SNP,Rank,SNP,Rank,SNP\n')
    m = len(snps)/3
    for i in range(m):
        f.writelines(str(i+1)+','+snps[i]+','+str(i+m+1)+','+snps[i+m]+','+str(i+m+m+1)+','+snps[i+m+m]+'\n')
    f.close()



if __name__ == '__main__':
    evaluate()
