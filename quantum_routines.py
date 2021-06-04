import qutip
from qutip import qeye, sigmam, sigmap, sigmay, sigmaz, tensor
import numpy as np
import settings


def generate_random_positions(N_atoms):
    return np.random.rand(N_atoms, 2)


def generate_mixing_Ham(N, coords=None):
    """Build the mixing Hamiltonian
    Parameters
    ----------
    N: number of nodes
    coords: list of lists
        coordinates of particles
    Returns
    -------
    h_m: qutip.Qobj()
        Mixing Hamiltonian, with possible non-zero detuning
    """
    si = qeye(2)
    sz = sigmaz()
    sy = sigmay()
    nz = (sz + 1) / 2

    sy_list = []
    nz_list = []

    for j in range(N):
        op_list = [si for _ in range(N)]

        op_list[j] = sy
        sy_list.append(tensor(op_list))

        op_list[j] = nz
        nz_list.append(tensor(op_list))

    # construct the hamiltonian
    h_m = 0

    # Laser field
    for j in range(N):
        h_m += -settings.omega * sy_list[j] + 0. * settings.delta * nz_list[j]

    return h_m


def generate_detuning_Ham(N, coords=None):
    """Build the detuning Hamiltonian
    Parameters
    ----------
    coords: list of lists
        coordinates of particles
    Returns
    -------
    h_m: qutip.Qobj()
        Detuning Hamiltonian
    """

    si = qeye(2)
    sz = sigmaz()
    nz = (sz + 1) / 2
    nz_list = []

    for j in range(N):
        op_list = [si for _ in range(N)]

        op_list[j] = nz
        nz_list.append(tensor(op_list))

    # construct the hamiltonian
    h_d = 0

    # Laser field
    for j in range(N):
        h_d += settings.delta * nz_list[j]

    return h_d


def generate_many_exc_mixing_Ham(N, nexc, coords=None):
    """Build the m-excitation ladder operator
    Parameters
    ----------
    coords: list of lists
        coordinates of particles
    Returns
    -------
    h_m: qutip.Qobj()
        Mixing Hamiltonian, with possible non-zero detuning
    """
    si = qeye(2)
    sp = sigmap()
    sp_list = []

    for j in range(N):
        op_list = [si for _ in range(N)]
        op_list[j] = sp
        sp_list.append(tensor(op_list))

    h_m = 0
    for j in range(N):
        h_m += sp_list[j]
    h_m = h_m ** nexc
    return h_m / np.amax(h_m)


def generate_Ham_from_graph(graph, type_h='xy', process_node=None,
                            process_edge=None):
    """Given a connectivity graph, build the Hamiltonian, Ising or XY.
    Parameters
    ----------
    graph: networkx.Graph(), nodes numeroted from 0 to N_nodes
    type_h: str, type of hamiltonian 'xy' or 'ising'
    process_node: function, function to convert the node attribute into a
    numerical value, add diagonal term to the hamiltonian
    process_edge: funciton, function to convert the edge attribute into a
    numerical value, add weight to the hamiltonian

    Returns
    -------
    H: qutip.Qobj()
        Hamiltonian for the configuration
    """
    assert type_h in ['ising', 'xy']

    N = graph.number_of_nodes()

    si = qeye(2)
    sp = sigmap()
    sm = sigmam()

    sz = sigmaz()

    sp_list = []
    sz_list = []
    sm_list = []

    for j in range(N):
        op_list = [si for _ in range(N)]

        op_list[j] = sp
        sp_list.append(tensor(op_list))

        op_list[j] = sm
        sm_list.append(tensor(op_list))

        op_list[j] = 0.5 * (sz + si)
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    for edge in graph.edges.data():
        edge_weight = 1
        if len(edge[2]) > 0:
            if process_edge is not None:
                edge_weight = process_edge(edge[2]['attr'])
        if type_h == 'ising':
            H += edge_weight * sz_list[edge[0]] * sz_list[edge[1]]
        elif type_h == 'xy':
            H += edge_weight * (sp_list[edge[0]] * sm_list[edge[1]]
                  + sm_list[edge[0]] * sp_list[edge[1]])
    return H


def generate_empty_initial_state(N):
    """Generates the empty initial wavefunction
    Parameters
    ----------
    N: number of nodes
    coords: list of lists
        coordinates of particles
    Returns
    -------
    psi_0: qutip.Qobj()
        Initial wavefunction
    """
    ei = '1'
    for tt in range(N - 1):
        ei += '1'
    psi_0 = qutip.ket(ei)
    return psi_0
