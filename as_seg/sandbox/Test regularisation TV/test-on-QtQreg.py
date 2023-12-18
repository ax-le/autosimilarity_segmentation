import numpy as np
from matplotlib import pyplot as plt
from tv_kmeans import tv_kmeans_admm, tv_kmeans

# Testing come Together

# root at file


lamb = 5 # careful, lamb does not change in the figure name
mu = 2
k = 6

ranks = '12 24 24'

for num in ['1']:#,'25','26','84']:
    path = '../data/{}/factors_{}_pcp_chromas_96.npy'.format(ranks,num)
    data = np.load(path, allow_pickle=True)

    Q = data[2].T
    Qnorm = Q/np.sqrt(np.sum(Q**2, axis=0))
    QtQ = np.dot(Qnorm.T,Qnorm)

    # Adding temporal regularity
    reg = 0.5
    n, _ = QtQ.shape
    Offdiag = np.zeros([n,n])
    for i in range(n-1):
        Offdiag[i+1,i] = 1
        Offdiag[i,i+1] = 1
    plt.subplot(121)
    plt.imshow(QtQ)
    QtQreg = np.minimum(QtQ + reg*Offdiag, 1)
    plt.subplot(122)
    plt.imshow(QtQreg)
    plt.show()



    for i in range(4):
        plt.subplot(2,2,i+1)
        out_tv = tv_kmeans(QtQreg, k, lamb=lamb, mu=mu, itermax=100)  # TV regularized
        plt.plot(out_tv[1], linestyle = '-.')
        plt.plot(out_tv[2])
        plt.plot(out_tv[3], linestyle = '--')
        plt.legend(['TVsigma', 'TVz', 'NoTV'])

    #fname = '../plots/rwc{}_lambda3_mu{}_k{}_ranks_{}'.format(num,mu,k,ranks)
    plt.title('RWC'+num)
    #plt.savefig(fname)
    plt.show()
