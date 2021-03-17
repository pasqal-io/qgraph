from data_preprocessing import load_dataset
import ray
from utils import generate_signal_fourier
import os
import numpy as np
import networkx as nx
from mpi4py import MPI
import pickle
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cores = comm.Get_size()
print("Rank " + str(rank) + ": CPU count: ", os.cpu_count())

dataset = 'IMDB-MULTI'

graphs, targets = load_dataset(dataset, max_node=16, min_node=0)

N_sample=len(graphs)

#np.random.seed(40)
#sample = np.random.choice(len(graphs), size=N_sample, replace=False)

#graphs_sample = graphs[sample]
#targets_sample = targets[sample]

graphs_sample = [nx.convert_node_labels_to_integers(G) for G in graphs_sample]
graphs_sample = np.array(graphs_sample, dtype=object)

#@ray.remote
def return_fourier_from_dataset(graph_list):
    fourier_signals_xy = np.zeros((2, len(graph_list), 1000))
    fourier_signals_ising = np.zeros((2, len(graph_list), 1000))

    for i,graph in enumerate(graph_list):
        fourier_signals_xy[0][i], fourier_signals_xy[1][i] = generate_signal_fourier(graph, N_sample=1000, hamiltonian='xy')
        fourier_signals_ising[0][i], fourier_signals_ising[1][i] = generate_signal_fourier(graph, N_sample=1000, hamiltonian='ising')

    return fourier_signals_xy, fourier_signals_ising


indices = np.array_split(np.arange(N_sample), cores)

indices_rank = np.array(indices[rank]).astype(int)

#ray.init()

t0=time()
#n_jobs = max(1,os.cpu_count()-2)
#n_jobs = 2
#batches = np.array_split(indices_rank, n_jobs)
results = []
print("Type of object: ", type(graphs_sample[0]))
# for i in range(n_jobs):
#     l = []
#     for j in batches[i]:
#         l.append(graphs_sample[j])
#     try:
#         results.append(return_fourier_from_dataset.remote(l))
#     except TypeError:
#         print(batches[i].astype(int).dtype)
#         print(batches[i])
#result = ray.get(results)
for j in indices_rank:
    l.append(graphs_sample[j])
try:
    results = return_fourier_from_dataset(l)
except TypeError:
    pass
 #   print(batches[i].astype(int).dtype)
    #print(batches[i])

t1=time()
print("Time "+str(rank)+": "+str(t1-t0))

# fourier_signals_xy_list = []
# fourier_signals_ising_list = []

# for i in range(len(result)):
#     fourier_signals_xy_list.append(results[i][0])
#     fourier_signals_ising_list.append(results[i][1])

fourier_signals_xy = results[0]#np.concatenate(fourier_signals_xy_list, axis=1)
fourier_signals_ising = results[1]#np.concatenate(fourier_signals_ising_list, axis=1)

#ray.shutdown()

result_to_save = {}
result_to_save['indices'] = indices_rank
result_to_save['xy'] = fourier_signals_xy
result_to_save['ising'] = fourier_signals_ising

file_name = "results/results_imdb/result_"+str(rank)+".pkl"

with open(file_name, "wb") as f:
    pickle.dump(result_to_save, f)
