# A script to implement naively a possible piece-wise clustering with regularized K-means

import numpy as np
import condat_tv
from sklearn.cluster import KMeans

def tv_kmeans(M, k, lamb=1, mu=1, init='sklearn', itermax=100, custom_sigma = None):
    '''
    Computes naively a coherent k-means using TV regularization on the affectations. In practice, splitting is used.

    Improvements: use better splitting (ADMM?)

    Inputs:

    M : np array (n by m)
        data matrix, columns are the vectors to cluster.

    k : int
        number of clusters

    lamb : float
        TV regularization parameter

    mu : float
        split constraint parameter (strengh of \| z - sigma \|^2)

    init : string or (C, sigma, z) tuple (todo!!)
        initialization method. Set 'random' for random init, 'sklearn' for standard kmeans init with scikit-learn.

    Outputs:

    C : np array (n by k)
        centroids

    sigma : np.array (n)
        the affectation of each data point to the centroids

    z : np.array(n)
        split variable, should be close to sigma

    version 0 by J.E. Cohen, 10 June 2021
    '''

    m,n = M.shape

    if init=='random':
        #C = np.random.randn(m,k)
        sigma = np.random.randint(0, high=k, size=n)
        z = sigma.copy()
    elif init=='sklearn':
        model = KMeans(n_clusters=k).fit(M.T)
        sigma = model.labels_
        z = sigma.copy()
        C = np.transpose(model.cluster_centers_)
    elif init=='custom':
        if custom_sigma is None:
            raise NotImplementedError("custom_sigma must be set for custom init.")
        sigma = custom_sigma
        z = sigma.copy()
    else:
        raise NotImplementedError(f"Initialization type not understood: {init}")

    # Sorting the initialization to remove permutation ambiguity
    acum = []
    for i in range(n):
        if sigma[i] not in acum:
            acum.append(sigma[i])
    sigma_sort = sigma.copy()
    for l in range(len(acum)):
        sigma_sort[sigma==acum[l]] = l
    sigma = sigma_sort.copy()
    sigma_0 = sigma.copy()

    for iter in range(itermax):
        # calcul des centroids
        C = np.zeros((m,k))
        counts = np.zeros(k)
        for i in range(n):
            C[:,int(sigma[i])] += M[:,i]
            counts[int(sigma[i])] += 1
        # normalization
        C = C/np.maximum(counts,1)

        # calcul des affectations
        # computing all the costs (separable :) )
        costij = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                costij[i,j] = np.sum((M[:,i] - C[:,j])**2)/n + mu*(j - z[i])**2
        # Minimizing cost
        sigma = np.argmin(costij, axis=1)

        # calcul de z
        if mu > 0:
            z_old = z.copy()
            z = condat_tv.tv_denoise(sigma, lamb/2/mu)

        # Stopping if z does not change
        if np.linalg.norm(z-z_old)<1e-8:
            break
        

    return C, sigma, z, sigma_0



def tv_kmeans_admm(M, k, lamb=1, mu=1, init='sklearn', itermax=100):
    '''
    Computes naively a coherent k-means using TV regularization on the affectations. In practice, splitting is used. ADMM version

    Improvements: use better splitting (ADMM?)

    Inputs:

    M : np array (n by m)
        data matrix, columns are the vectors to cluster.

    k : int
        number of clusters

    lamb : float
        TV regularization parameter

    mu : float
        split constraint parameter (strengh of \| z - sigma + tau \|^2)

    init : string or (C, sigma, z) tuple (todo!!)
        initialization method. Set 'random' for random init, 'sklearn' for standard kmeans init with scikit-learn.

    Outputs:

    C : np array (n by k)
        centroids

    sigma : np.array (n)
        the affectation of each data point to the centroids

    z : np.array(n)
        split variable, should be close to sigma

    version 0 by J.E. Cohen, 10 June 2021
    '''

    m,n = M.shape

    if init=='random':
        #C = np.random.randn(m,k)
        sigma = np.random.randint(0, high=k, size=n)
        z = sigma.copy()
    elif init=='sklearn':
        model = KMeans(n_clusters=k).fit(M.T)
        sigma = model.labels_
        z = sigma.copy()
        C = np.transpose(model.cluster_centers_)

    # Sorting the initialization to remove permutation ambiguity
    acum = []
    for i in range(n):
        if sigma[i] not in acum:
            acum.append(sigma[i])
    sigma_sort = sigma.copy()
    for l in range(len(acum)):
        sigma_sort[sigma==acum[l]] = l
    sigma = sigma_sort.copy()
    sigma_0 = sigma.copy()

    # always start with tau=0
    tau = np.zeros(z.shape)

    for iter in range(itermax):
        # calcul des centroids
        C = np.zeros((m,k))
        counts = np.zeros(k)
        for i in range(n):
            C[:,sigma[i]] += M[:,i]
            counts[sigma[i]] += 1
        # normalization
        C = C/np.maximum(counts,1)

        # calcul des affectations
        # computing all the costs (separable :) )
        costij = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                costij[i,j] = np.sum((M[:,i] - C[:,j])**2)/n + mu*(j - z[i] + tau[i])**2
        # Minimizing cost
        sigma = np.argmin(costij, axis=1)

        # calcul de z
        if mu > 0:
            z_old = z.copy()
            z = condat_tv.tv_denoise(sigma + tau, lamb/mu)

        # Update tau by gradient ascent
        tau = tau + sigma - z

        # Stopping if z does not change
        if np.linalg.norm(z-z_old)<1e-8:
            break

    return C, sigma, z, sigma_0



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    A = np.random.randn(4,50)
    k = 3
    #out0 = tv_kmeans(A,k,lamb=1,mu=1)
    out = tv_kmeans_admm(A,k,lamb=0.2,mu=1)
    plt.figure()
    plt.plot(out[1])
    plt.plot(out[3], linestyle='--')
    #plt.plot(out0[1])
    plt.legend(['TVadmm', 'NoTV'])#, 'TVfuzzy'])
    print(out)
    plt.show()

    #plt.figure()
    #plt.scatter(A[0,:], A[1,:])
    #plt.scatter(out[0][0,:], out[0][1,:])
    #plt.scatter(out2[0][0,:], out2[0][1,:])
    #plt.show()
