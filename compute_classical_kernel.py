import numpy as np
import networkx as nx
import grakel
from data_preprocessing import load_dataset
from mpi4py import MPI
import pickle
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cores = comm.Get_size()



dataset = 'reddit_threads'
results_folder = 'results/results_reddit'

graphs, targets = load_dataset(dataset, max_node=16, min_node=0)

files = os.listdir(results_folder)

all_indices = []
for file_name in files:
    with open('results/'+file_name, 'rb') as f:
        data = pickle.load(f)
    all_indices.append(data['indices'])

all_indices = np.concatenate(all_indices, axis=0)

N = len(all_indices)

graphs_grakel = [grakel.Graph(nx.adjacency_matrix(graphs[all_indices[i]])) for i in range(N)]

n_jobs = 20

rw_kernel = grakel.RandomWalk(n_jobs=n_jobs, lamda=0.001)
gs_kernel = grakel.GraphletSampling(k=6, sampling={'n_sample':100})
lt_kernel = grakel.LovaszTheta(n_jobs=n_jobs)

kernels = [rw_kernel, gs_kernel, lt_kernel]
names = ['rw', 'gs', 'lt']

assert cores<=len(kernels)

K = kernels[rank].fit_transform(graphs_grakel)

name = 'results/classical_kernels/K_' + names[rank]

np.save(name, K)




