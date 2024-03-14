import numpy as np

def loadSyntheticData(model):
    if model == 'tree':
        snps = np.load('../toyData/group_tree_X.npy')
        Y = np.load('../toyData/y_tree_test.npy')
        Kva = np.load('../toyData/group_tree_Kva.npy')
        Kve = np.load('../toyData/group_tree_Kve.npy')
        return snps, Y, Kva, Kve

    elif model == 'group':
        snps = np.load('../toyData/group_tree_X.npy')
        Y = np.load('../toyData/y_group_test.npy')
        Kva = np.load('../toyData/group_tree_Kva.npy')
        Kve = np.load('../toyData/group_tree_Kve.npy')
        return snps, Y, Kva, Kve

def loadRealData(dt):
    if dt == 'alz':
        path = '/home/haohanw/AlzData/'
        X = np.load(path + 'snps.npy')
        y = np.load(path + 'phenoMulti.npy')
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        return X, y, Kva, Kve
    elif dt == 'at':
        path = '../realData/ATData/'
        X = np.load(path + 'geno.npy')
        y = np.load(path + 'pheno.npy')
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        ind = [0, 1, 2, 3, 4, 5, 6, 39, 40,
               32, 33, 34, 73, 74, 75, 101, 102, 98, 99, 48, 49, 50, 51, 52, 71, 72,
               82,83, 84, 65, 66, 67, 79, 80, 81, 88, 89, 90, 7, 100, 104, 105, 106, 91]
        # ind = [100, 104, 105, 106]
        y = y[:, ind]
        return X, y, Kva, Kve
    elif dt == 'mice':
        ind = range(49, 58) + range(78, 97)
        path = '/home/haohanw/miceData/'
        X = np.load(path + 'snps.npy')
        y = np.load(path + 'pheno.npy')[:, ind]
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        return X, y, Kva, Kve


def loadRealData2(dt):
    if dt == 'alz':
        path = 'E:/lx/AdvancedLMM-master/realData/AlzData/'
        X = np.load('E:/lx/AdvancedLMM-master/AdvancedLMM-master/scripts/SUX.npy')
        # print "SUX"
        y = np.load('E:/lx/AdvancedLMM-master/AdvancedLMM-master/scripts/SUy.npy')
        # print "SUy"
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        return X, y, Kva, Kve
    elif dt == 'at':
        path = 'E:/lx/AdvancedLMM-master/realData/ATData/'
        X = np.load('E:/lx/AdvancedLMM-master/AdvancedLMM-master/scripts/SUXat.npy')
        y = np.load('E:/lx/AdvancedLMM-master/AdvancedLMM-master/scripts/SUyat.npy')
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        # ind = [82, 83, 84, 65, 66, 67, 79, 80, 81, 88, 89, 90, 7, 100, 104, 105, 106, 91, 0, 1, 2, 3, 4, 5, 6, 39, 40,
        #        32, 33, 34, 73, 74, 75, 101, 102, 98, 99, 48, 49, 50, 51, 52, 71, 72]
        # # ind = [100, 104, 105, 106]
        # # ind = [100]
        # y = y[:, ind]
        return X, y, Kva, Kve
    elif dt == 'mice':
        ind = range(49, 58) + range(78, 97)
        path = '/home/haohanw/miceData/'
        X = np.load(path + 'snps.npy')
        y = np.load(path + 'pheno.npy')[:, ind]
        Kva = np.load(path + 'Kva.npy')
        Kve = np.load(path + 'Kve.npy')
        return X, y, Kva, Kve

if __name__ == '__main__':
    # path='E:/lx/AdvancedLMM-master/AdvancedLMM-master/data/real/at/'
    # x=np.load(path+'causal_0.npy')
    # #path='E:/lx/AdvancedLMM-master/AdvancedLMM-master/data/real/at/'
    # y=np.load(path+'position.npy')
    # # x=float(x)
    # # y=float(y)
    # # np.savetxt(path+'causal_0.csv',x,delimiter=',')
    # # np.savetxt(path+'position.csv',y,delimiter=',')
    # # print x
    # # print " "
    # # print y.shape
    # print y[0]
    pass