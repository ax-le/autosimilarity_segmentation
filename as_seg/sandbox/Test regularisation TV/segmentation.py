import numpy as np
from tv_kmeans import tv_kmeans_admm, tv_kmeans

def frontiers_q_clustering_one_run(q_factor, nb_clusters = 6, lamb = 20, mu = 20):
    # Une passe de clustering et les forntières qui lui sont associées
    clusters = tv_kmeans(q_factor.T, nb_clusters, lamb=lamb, mu=mu, itermax=100, init="sklearn")[1]  # TV regularized
    frontiers = []
    for i in range(len(clusters) - 1):
        if clusters[i] != clusters[i+1]:
            frontiers.append(i+1) # i + 1 car segment ne comprend pas dernière mesure (car la mesure d'après commence à la frontière)
    frontiers.append(len(clusters) - 1) # On rajoute la dernière pour cloturer segmentation
    return frontiers

def frontiers_several_runs_clustering(q_factor, nb_clusters, lamb, mu, iter_clus):
    # On itère plusieurs fois le processus de clustering et on ne garde que les frontières qui apparaissent plus de la moitié du temps.
    frontier_proba = np.zeros(q_factor.shape[0])
    for i in range(iter_clus):
        frontiers_one_iteration = frontiers_q_clustering_one_run(q_factor)
        for front in frontiers_one_iteration:
            frontier_proba[front] += 1
    frontiers = []
    for index_proba in range(len(frontier_proba)):
        if frontier_proba[index_proba] >= iter_clus/2:
            frontiers.append(index_proba)
    return frontiers
