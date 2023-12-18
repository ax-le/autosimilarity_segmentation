import numpy as np
from matplotlib import pyplot as plt
import segmentation
from tv_kmeans import tv_kmeans

# Testing come Together

# root at file


lamb = 15 # careful, lamb does not change in the figure name
mu = 5
k = 5

ranks = '12 24 24'

for num in ['1','25','26','84']:
    path = '../data/{}/factors_{}_pcp_chromas_96.npy'.format(ranks,num)
    data = np.load(path, allow_pickle=True)

    Q = data[2].T
    # Adding bleed to Q for temporal consistency?
    _, m = np.shape(Q)
    Qreg = np.copy(Q)
    reg = 4e-1
    for i in range(2,m-2): # to implement with matrix product
        # weighted local average
        Qreg[:,i] = (reg/2*(Q[:,i-2] + Q[:,i+2]) + reg*(Q[:,i-1]+Q[:,i+1]) + Q[:,i])/(1+3*reg)
    # Dealing with borders
    Qreg[:,0] = (reg/2*Q[:,2] + reg*Q[:,1] + Q[:,0])/(1+1.5*reg)
    Qreg[:,-1] = (reg/2*Q[:,-3] + reg*Q[:,-2] + Q[:,-1])/(1+1.5*reg)
    Qreg[:,1] = (reg/2*Q[:,3] + reg*(Q[:,0] + Q[:,2]) + Q[:,1])/(1+2.5*reg)
    Qreg[:,-2] = (reg/2*Q[:,-4] + reg*(Q[:,-3]+ Q[:,-1]) + Q[:,-2])/(1+2.5*reg)

    frontiers = segmentation.frontiers_several_runs_clustering(Qreg.T, k, lamb, mu, 50)

    # for the plot
    out_tv = tv_kmeans(Qreg, k, lamb=lamb, mu=mu, itermax=100)  # TV regularized

    print(frontiers)

    #fname = '../plots/Qreg_rwc{}_lambda{}_mu{}_k{}_ranks_{}'.format(num,lamb,mu,k,ranks)
    plt.plot(out_tv[1])
    #plt.title('RWC'+num)
    #plt.savefig(fname)
    plt.show()
