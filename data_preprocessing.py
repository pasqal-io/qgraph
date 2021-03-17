import numpy as np
import networkx as nx
#import grakel
import os
from os import path


#folder where the datasets are located
datasets_path = 'datasets'


def load_dataset(name, min_node=0, max_node=12):

    '''
    Loads the dataset with the corresponding name. Creates an array of the graph with nodes and edge attributes, and an array of targets.
    Details about the file formats here: https://chrsmrrs.github.io/datasets/docs/format/
    The datasets can be downloaded here: https://chrsmrrs.github.io/datasets/docs/datasets/

    Arguments:
    ---------
    - name: str, name of the dataset
    - min_node: int, eliminate all the graphs with a number of nodes below the value passed
    - max_node: int, eliminate all the graphs with a number of nodes above the value passed

    Returns:
    --------
    - graph_filtered: numpy.Ndarray of networkx.Graph objects, all nodes attributes and edge attributes are stored in the key 'attr'
    - targets_filtered: numpy.Ndarray of floats, discrete values for classification, continuous ones for regression
    '''
    
    directory = datasets_path + '/' + name + '/' + name + '_'
    is_node_attr = False
    is_edge_attr = False
    
    with open(directory + 'graph_indicator.txt') as file:
        all_nodes = np.array(file.read().splitlines()).astype(int)
    
    with open(directory + 'A.txt') as file:
        all_edges = np.array(file.read().splitlines())

    if path.exists(directory + 'graph_labels.txt'):
        targets = np.loadtxt(directory + 'graph_labels.txt', delimiter=',')

    if path.exists(directory + 'graph_attributes.txt'):
        targets = np.loadtxt(directory + 'graph_attributes.txt', delimiter=',')

    if path.exists(directory + 'edge_attributes.txt'):
        edge_attr = np.loadtxt(directory + 'edge_attributes.txt', delimiter=',')
        is_edge_attr = True

    if path.exists(directory + 'node_attributes.txt'):
        node_attr = np.loadtxt(directory + 'node_attributes.txt', delimiter=',')
        is_node_attr = True

    l = []
    for edge in all_edges:
        #print(edge)
        edge = edge.replace(' ', '')
        l.append((int(edge.split(',')[0]), int(edge.split(',')[1])))
    #l = [(int(edge.split(', ')[0]), int(edge.split(', ')[1])) for edge in all_edges]
    all_edges = np.array(l)

    all_graphs = [nx.Graph() for i in range(np.max(all_nodes))]

    for i,graph_id in enumerate(all_nodes):
        if is_node_attr:
            all_graphs[graph_id-1].add_node(i+1, attr=node_attr[i])
        else:
            all_graphs[graph_id-1].add_node(i+1)

    for i,edge in enumerate(all_edges):
        node_id = edge[0]-1
        graph_id = all_nodes[node_id]
        if is_edge_attr:
            all_graphs[graph_id-1].add_edge(*edge, attr=edge_attr[i])
        else:
            all_graphs[graph_id-1].add_edge(*edge)

    all_graphs = np.array(all_graphs, dtype=object)

    
    graph_filtered_id = np.array([i for i,graph in enumerate(all_graphs) if (graph.number_of_nodes()<=max_node) & (graph.number_of_nodes()>=min_node)]).astype(int)

    graph_filtered = all_graphs[graph_filtered_id]
    targets_filtered = targets[graph_filtered_id]

    return graph_filtered, targets_filtered

    