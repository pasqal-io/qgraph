from data_preprocessing import load_dataset
import ray
from utils import generate_signal_fourier, return_fourier_from_dataset, jensen_shannon
import os
import numpy as np
import networkx as nx
from mpi4py import MPI
import pickle
from time import time
import grakel
import pickle
from estimators import QuantumKernelEstimator
import argparse


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cores = comm.Get_size()


parser = argparse.ArgumentParser()

parser.add_argument("dataset", help="dataset you want to benchmark")
parser.add_argument("results_folder", help="folder to store the results")
parser.add_argument("-N", "--N_sample", help="number of samples to draw", type=int)
args = parser.parse_args()

def return_partial_distance_matrix(histograms, indices):
    js = np.zeros((len(indices), len(histograms)))
    for i in range(len(indices)):
        for j in range(np.min(indices)+i, len(histograms)):
            js[i,j] = jensen_shannon(histograms[np.min(indices)+i], histograms[j])
    return js



dataset = args.dataset
results_folder = args.results_folder

graphs, targets = load_dataset(dataset, max_node=16, min_node=0)
np.random.seed(40)

if args.N_sample>0:
    N_sample = args.N_sample
    sample = np.random.choice(len(graphs), size=N_sample, replace=False).astype(int)
else:
    N_sample=len(graphs)
    sample = np.arange(len(graphs)).astype(int)

print(N_sample)

graphs_sample = graphs[sample]
targets_sample = targets[sample]

graphs_sample = [nx.convert_node_labels_to_integers(G) for G in graphs_sample]
graphs_sample = np.array(graphs_sample, dtype=object)

indices = np.array_split(np.arange(N_sample).astype(int), cores)

if rank == 0:
    np.save(results_folder + '/' + 'targets.npy', targets_sample)


t0=time()
fourier_xy, fourier_ising = return_fourier_from_dataset(graphs_sample[indices[rank]])
t1=time()
print("Rank "+str(rank)+": ",t1-t0)

all_fourier_xy = comm.gather(fourier_xy, root=0)
all_fourier_ising = comm.gather(fourier_xy, root=0)
#all_indices = comm.gather(indices[rank], root=0)

if rank == 0:
    all_fourier_xy = np.concatenate(all_fourier_xy, axis=0)
    all_fourier_ising = np.concatenate(all_fourier_ising, axis=0)
    print(all_fourier_xy.shape)
    np.save(results_folder + '/' + 'all_fourier_xy.npy', all_fourier_xy)
    np.save(results_folder + '/' + 'all_fourier_ising.npy', all_fourier_ising)
    #send_xy = [all_fourier_xy] * cores
    #send_ising = [all_fourier_ising] * cores
    print(all_fourier_xy[0].shape)
else:
    all_fourier_xy = None
    all_fourier_ising = None


all_fourier_xy= comm.bcast(all_fourier_xy, root=0)
all_fourier_ising = comm.bcast(all_fourier_ising, root=0)


histograms_xy = []
histograms_ising = []

for i in range(N_sample):
    histograms_xy.append(np.histogram(all_fourier_xy[i,1], bins=500, weights=np.abs(all_fourier_xy[i,0])**2, density=True))
    histograms_ising.append(np.histogram(all_fourier_ising[i,1], bins=500, weights=np.abs(all_fourier_ising[i,0])**2, density=True))

partial_matrix_xy = return_partial_distance_matrix(histograms_xy, indices[rank])
partial_matrix_ising = return_partial_distance_matrix(histograms_ising, indices[rank])

all_matrices_xy = comm.gather(partial_matrix_xy, root=0)
all_matrices_ising = comm.gather(partial_matrix_ising, root=0)

scoring = {'accuracy': make_scorer(accuracy_score),
          'recall': make_scorer(recall_score, average='weighted'),
          'f1_score': make_scorer(f1_score, average='weighted')}


if rank == 0:
    js_xy = np.concatenate(all_matrices_xy, axis=0)
    js_ising = np.concatenate(all_matrices_ising, axis=0)
    js_xy = js_xy + np.transpose(js_xy)
    js_ising = js_ising + np.transpose(js_ising)
    print(js_xy.shape)
    np.save(results_folder + '/' + 'js_xy.npy', js_xy)
    np.save(results_folder + '/' + 'js_ising.npy', js_ising)
    p_list = np.linspace(0,1,11)
    mu_list = np.logspace(-3, 2, 6)
    C_list = np.logspace(-3, 4, 8)

    param_grid = {'p':p_list, 'mu':mu_list, 'C':C_list}
    X = np.concatenate([np.expand_dims(js_xy, 2), np.expand_dims(js_ising, 2)], axis=2)
    print(X.shape)
    estimator = QuantumKernelEstimator()
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=10, refit=False)

    result = grid_search.fit(X, targets_sample).cv_results_

    file_name = results_folder + "/result_quantum.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(result, f)



    # scores = np.zeros((len(p_list), len(mu_list), len(C_list), 10))
    # for k,p in enumerate(p_list):
    #     for i,mu in enumerate(mu_list):
    #         for j,C in enumerate(C_list):
    #             K = np.exp(-mu * (p*js_xy + (1-p)* js_ising))
    #             clf = svm.SVC(kernel='precomputed', C=C, random_state=76)
    #             s = cross_val_score(clf, K, targets_sample, cv=10, scoring=make_scorer(f1_score, average='weighted'))
    #             scores[k,i,j] = s
    print('Hyperparams quantum')
    # np.save(results_folder + "/all_scores_quantum.npy", scores)
    

    

if rank>0 and rank<=3:
    graphs_grakel = [grakel.Graph(nx.adjacency_matrix(graphs_sample[i])) for i in range(N_sample)]

    rw_kernel = grakel.RandomWalk(lamda=0.001)
    gs_kernel = grakel.GraphletSampling(k=6, sampling={'n_samples':100})
    lt_kernel = grakel.LovaszTheta()

    kernels = [rw_kernel, gs_kernel, lt_kernel]
    names = ['rw', 'gs', 'lt']

    K = kernels[rank-1].fit_transform(graphs_grakel)
    np.save(results_folder + '/K_' + names[rank-1] + ".npy", K)
    C_list = np.logspace(-3, 2, 6)
    print('Kernel '+names[rank-1]+' computed')
    param_grid = {'C':C_list}

    estimator = svm.SVC(kernel='precomputed', random_state=76)
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=10, refit=False)
    result = grid_search.fit(K, targets_sample).cv_results_
    
    # scores = np.zeros((1, len(C_list), 10))
    # for j,C in enumerate(C_list):
    #     clf = svm.SVC(kernel='precomputed', C=C, random_state=76)
    #     s = cross_val_score(clf, K, targets_sample, cv=10, scoring=make_scorer(f1_score, average='weighted'))
    #     scores[0,j] = s

    file_name = results_folder + "/all_scores_classical_" + names[rank-1] + ".pkl"


    with open(file_name, "wb") as f:
        pickle.dump(result, f)

    #np.save(results_folder+file_name, scores)
    print('Hyperparams '+ str(names[rank-1]))







                                                                     
