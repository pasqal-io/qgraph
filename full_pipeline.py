from data_preprocessing import load_dataset
from utils import return_fourier_from_dataset, jensen_shannon
import numpy as np
import networkx as nx
from mpi4py import MPI
from time import time
import grakel

from sklearn import svm
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cores = comm.Get_size()


def return_partial_distance_matrix(histograms, indices):
    js = np.zeros((len(indices), len(histograms)))
    for i in range(len(indices)):
        for j in range(np.min(indices)+i, len(histograms)):
            js[i, j] = jensen_shannon(histograms[np.min(indices)+i],
                                      histograms[j])
    return js


dataset = 'IMDB-MULTI'
results_folder = 'results/results_imdb'

graphs, targets = load_dataset(dataset, max_node=10, min_node=0)

N_sample = 50

np.random.seed(40)
sample = np.random.choice(len(graphs),
                          size=N_sample, replace=False).astype(int)


graphs_sample = graphs[sample]
targets_sample = targets[sample]

graphs_sample = [nx.convert_node_labels_to_integers(G) for G in graphs_sample]
graphs_sample = np.array(graphs_sample, dtype=object)

indices = np.array_split(np.arange(N_sample).astype(int), cores)


t0 = time()
fourier_xy, fourier_ising = return_fourier_from_dataset(graphs_sample[
                                                        indices[rank]])
t1 = time()

print("Rank "+str(rank)+": ", t1-t0)

all_fourier_xy = comm.gather(fourier_xy, root=0)
all_fourier_ising = comm.gather(fourier_xy, root=0)
# all_indices = comm.gather(indices[rank], root=0)

if rank == 0:
    all_fourier_xy = np.concatenate(all_fourier_xy, axis=0)
    all_fourier_ising = np.concatenate(all_fourier_ising, axis=0)
    print(all_fourier_xy.shape)
    np.save(results_folder + '/' + 'all_fourier_xy.npy', all_fourier_xy)
    np.save(results_folder + '/' + 'all_fourier_ising.npy', all_fourier_ising)
    # send_xy = [all_fourier_xy] * cores
    # send_ising = [all_fourier_ising] * cores
    print(all_fourier_xy[0].shape)
else:
    all_fourier_xy = None
    all_fourier_ising = None


all_fourier_xy = comm.bcast(all_fourier_xy, root=0)
all_fourier_ising = comm.bcast(all_fourier_ising, root=0)


histograms_xy = []
histograms_ising = []

for i in range(N_sample):
    histograms_xy.append(
        np.histogram(all_fourier_xy[i, 1],
                     bins=500,
                     weights=np.abs(all_fourier_xy[i, 0])**2,
                     density=True))
    histograms_ising.append(
        np.histogram(all_fourier_ising[i, 1],
                     bins=500,
                     weights=np.abs(all_fourier_ising[i, 0])**2, density=True))

partial_matrix_xy = return_partial_distance_matrix(
                                        histograms_xy, indices[rank])
partial_matrix_ising = return_partial_distance_matrix(
                                        histograms_ising, indices[rank])

all_matrices_xy = comm.gather(partial_matrix_xy, root=0)
all_matrices_ising = comm.gather(partial_matrix_ising, root=0)

if rank == 0:
    js_xy = np.concatenate(all_matrices_xy, axis=0)
    js_ising = np.concatenate(all_matrices_ising, axis=0)
    js_xy = js_xy + np.transpose(js_xy)
    js_ising = js_ising + np.transpose(js_ising)
    print(js_xy.shape)
    np.save(results_folder + '/' + 'js_xy.npy', js_xy)
    np.save(results_folder + '/' + 'js_ising.npy', js_ising)
    p_list = np.linspace(0, 1, 11)
    mu_list = np.logspace(-3, 2, 6)
    C_list = np.logspace(-3, 4, 8)

    scores = np.zeros((len(p_list), len(mu_list), len(C_list), 10))
    for k, p in enumerate(p_list):
        for i, mu in enumerate(mu_list):
            for j, C in enumerate(C_list):
                K = np.exp(-mu * (p*js_xy + (1-p) * js_ising))
                clf = svm.SVC(kernel='precomputed', C=C, random_state=76)
                s = cross_val_score(clf, K, targets_sample, cv=10,
                                    scoring=make_scorer(f1_score,
                                                        average='weighted'))
                scores[k, i, j] = s
    print('Hyperparams quantum')
    np.save(results_folder + "/all_scores_quantum.npy", scores)


if rank > 0 and rank <= 3:
    graphs_grakel = [grakel.Graph(
                        nx.adjacency_matrix(
                            graphs_sample[i])) for i in range(N_sample)]

    rw_kernel = grakel.RandomWalk(lamda=0.001)
    gs_kernel = grakel.GraphletSampling(k=6, sampling={'n_samples': 100})
    lt_kernel = grakel.LovaszTheta()

    kernels = [rw_kernel, gs_kernel, lt_kernel]
    names = ['rw', 'gs', 'lt']

    K = kernels[rank-1].fit_transform(graphs_grakel)
    C_list = np.logspace(-3, 2, 6)
    print('Kernel ' + names[rank-1] + ' computed')

    scores = np.zeros((1, len(C_list), 10))
    for j, C in enumerate(C_list):
        clf = svm.SVC(kernel='precomputed', C=C, random_state=76)
        s = cross_val_score(clf, K, targets_sample, cv=10,
                            scoring=make_scorer(f1_score, average='weighted'))
        scores[0, j] = s

    file_name = "/all_scores_classical" + "_" + names[rank-1] + ".npy"
    print('Hyperparams ' + str(names[rank-1]))

    np.save(results_folder+file_name, scores)
