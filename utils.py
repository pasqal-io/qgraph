import qutip
from quantum_routines import generate_empty_initial_state, generate_mixing_Ham, generate_detuning_Ham, generate_Ham_from_graph
from qutip import mcsolve, sesolve, sigmaz, sigmap, qeye, tensor, Options
from settings import init
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import ray
import numba

init()

def generate_signal_fourier(G, N_sample=1000, hamiltonian='xy', tf=100*math.pi):
    """
    Function to return the Fourier transform of the average number of excitation signal

    Arguments:
    ---------
    - G: networx.Graph, graph to analyze
    - N_sample: int, number of timesteps to compute the evolution
    - hamiltonian: str 'xy' or 'ising', type of hamiltonian to simulate
    - tf: float, total time of evolution

    Returns:
    --------
    - plap_fft: numpy.Ndarray, shape (N_sample,) values of the fourier spectra
    - freq_normalized: numpy.Ndarray, shape (N_sample,) values of the fequencies
    """

    assert hamiltonian in ['ising', 'xy']
    N_nodes = G.number_of_nodes()
    H_evol = generate_Ham_from_graph(G, type_h=hamiltonian)

    rotation_angle_single_exc = math.pi/4.
    tlist = np.linspace(0, rotation_angle_single_exc, 200)

    psi_0 = generate_empty_initial_state(N_nodes)
    H_m = generate_mixing_Ham(N_nodes)

    sz = sigmaz()
    si = qeye(2)
    sp = sigmap()
    eops = []
    sz_list = []
    sp_list=[]
    nz = (sz + 1) / 2
    for j in range(N_nodes):
        op_list = [si for _ in range(N_nodes)]
        op_list[j] = nz
        eops.append(tensor(op_list))
        op_list[j] = sz
        sz_list.append(tensor(op_list))
        op_list[j] = sp
        sp_list.append(tensor(op_list))

    opts = Options()
    opts.store_states = True
    result = sesolve(H_m, psi_0, tlist, e_ops=eops, options=opts)
    plap = result.expect
    final_state = result.states[-1]

    #tf = 100*math.pi
    tlist = np.linspace(0, tf, N_sample)

    observable =- 2*math.sin(2*rotation_angle_single_exc)*sum(spj for spj in sp_list) + math.cos(2*rotation_angle_single_exc)*sum(szj for szj in sz_list)

    observables = []
    for j in range(len(sz_list)):
        observables.append(math.cos(2*rotation_angle_single_exc)*sz_list[j] - (2*math.sin(2*rotation_angle_single_exc)*sp_list[j]))

    result = sesolve(H_evol, final_state, tlist, e_ops=[observable],options=opts)
    plap = result.expect
    
    signal = plap[0].real
    plap_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    freq_normalized = np.abs(freq * N_sample * 2) / (tf / np.pi)
    
    return plap_fft, freq_normalized


@numba.njit
def entropy(p):
    """
    Returns the entropy of a discrete distribution p

    Arguments:
    ---------
    - p: numpy.Ndarray dimension 1 non-negative floats summing to 1

    Returns:
    --------
    - float, value of the entropy
    """
    assert (p>=0).all()
    assert abs(np.sum(p)-1)<1e-6
    return -np.sum(p*np.log(p+1e-12))


@numba.njit
def jensen_shannon(hist1, hist2):
    '''
    Returns the Jensen Shannon divergence between two probabilities distribution represented as histograms.

    Arguments:
    ---------
    - hist1: tuple of numpy.Ndarray (density, bins), len(bins) = len(density) + 1. The integral of the density wrt bins sums to 1.
    - hist2: same format.
    
    Returns:
    --------
    - float, value of the Jensen Shannon divergence.
    
    '''
    
    bins = np.sort(np.unique(np.array(list(hist1[1]) + list(hist2[1]))))
    masses1 = []
    masses2 = []

    for i,b in enumerate(bins[1::]):
        if b<=hist1[1][0]:
            masses1.append(0.)
        elif b>hist1[1][-1]:
            masses1.append(0.)
        else:
            j=0
            while b>hist1[1][j]:
                j+=1
            masses1.append((b-bins[i]) * hist1[0][j-1])
            
        if b<=hist2[1][0]:
            masses2.append(0.)
        elif b>hist2[1][-1]:
            masses2.append(0.)
        else:
            j=0
            while b>hist2[1][j]:
                j+=1
            masses2.append((b-bins[i]) * hist2[0][j-1])
            
    masses1 = np.array(masses1)
    masses2 = np.array(masses2)
    masses12 = (masses1+masses2)/2
    
    return entropy(masses12) - (entropy(masses1) + entropy(masses2))/2


#@ray.remote
def return_fourier_from_dataset(graph_list):
    """
    Returns the fourier transform of evolution for a list of graphs for the hamiltonian ising and xy.

    Arguments:
    ---------
    - graph_list: list or numpy.Ndarray of networkx.Graph objects
    
    Returns:
    --------
    - fourier_signals_xy: numpy.Ndarray of shape (2, len(graph_list), 1000)
                        [0,i]: Fourier signal of graph i at 1000 points for hamiltonian XY
                        [1,i]: frequencies associated to graph i at 1000 points for hamiltonian XY
    - fourier_signals_ising: same for the Ising hamiltonian

    """
    fourier_signals_xy = np.zeros((2, len(graph_list), 1000))
    fourier_signals_ising = np.zeros((2, len(graph_list), 1000))

    for i,graph in enumerate(graph_list):
        fourier_signals_xy[0][i], fourier_signals_xy[1][i] = generate_signal_fourier(graph, N_sample=1000, hamiltonian='xy')
        fourier_signals_ising[0][i], fourier_signals_ising[1][i] = generate_signal_fourier(graph, N_sample=1000, hamiltonian='ising')
        
    return fourier_signals_xy, fourier_signals_ising



    



