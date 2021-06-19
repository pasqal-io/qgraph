import networkx as nx
from tqdm.auto import tqdm
import numpy as np
import planarity
import dgl
from qutip import sigmaz, sigmap, qeye, tensor, sigmam
from utils import return_list_of_states, return_energy_distribution, return_js_square_matrix, return_js_dist_matrix, merge_energies, Memoizer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary

BOLD = "\033[1m"
STYLE_END = "\033[0m"

def load_planar_graphs(n):
    """ Load database of planar graphs """
    # n=9 takes 2s already
    # Data from https://hog.grinvin.org/Planar#triangulations
    return nx.read_graph6(f'datasets/planar_conn.{n}.g6')
    
def build_graph(edges):
    """ Builds a networkx graph from edges list """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def observable(graph):
    N = graph.number_of_nodes()

    si = qeye(2)
    sp = sigmap()
    sm = sigmam()

    sz = sigmaz()

    sp_list = []
    sz_list = []
    sm_list = []
    sn_list = []

    for j in range(N):
        op_list = [si for _ in range(N)]

        op_list[j] = sp
        sp_list.append(tensor(op_list))

        op_list[j] = sm
        sm_list.append(tensor(op_list))

        op_list[j] = sz
        sz_list.append(tensor(op_list))

        op_list[j] = 0.5 * (sz + si)
        sn_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    for node in graph.nodes.data():
        # node_weight = graph.degree[node[0]]
        node_weight = 0
        H += node_weight * sz_list[node[0]]

    for edge in graph.edges.data():
        edge_weight = 1
        H += edge_weight * sz_list[edge[0]] * sz_list[edge[1]]

    return H

def generate_graphs(nb_nodes: list, N_samples: list, generator='uniform_edges', verbose=False):
    """ 
    Generates N_samples_list[i] graphs with nb_nodes[i] nodes.
    Also computes the planarity test for these graphs.
    """
    graphs, targets = [], []
    for n, N in tqdm(zip(nb_nodes, N_samples), disable=not verbose):
        if generator == 'uniform_edges':
            possible_edges = np.array([(a, b) for a in range(n) for b in range(a)])
            N_edges_max = len(possible_edges)
            if verbose: print(f"Maximum edge number is {N_edges_max}")

        graphs_loop, targets_loop = [], []
        for _ in range(N):
            if generator == 'uniform_edges':
                taken_edges = np.random.choice(
                    range(len(possible_edges)),
                    # random number of edges
                    size=np.random.randint(
                        n - 1,  # can be not connected
                        N_edges_max + 1  # fully connected graph
                    ),
                    replace=False
                )
                graph_edges = possible_edges[taken_edges]
            elif generator == 'binomial':
                for _ in range(10):
                    graph_edges = list(nx.generators.random_graphs.binomial_graph(n, np.random.random(), seed=0).edges)
                    if len(graph_edges) > 0: break
                if len(graph_edges) == 0:
                    graph_edges = [0, 1]
            else:
                raise ValueError('Unexpected value for generator argument of generate_graphs')
            
            graphs_loop.append(build_graph(graph_edges))
            targets_loop.append(planarity.is_planar(graph_edges))

        graphs_loop = [nx.convert_node_labels_to_integers(G) for G in graphs_loop]
        # graphs = np.array(graphs, dtype=object)
        graphs.extend(graphs_loop)
        targets.extend(targets_loop)
    
        if verbose: print(f"Generated graphs with {n:3d} nodes are at {sum(targets_loop) / len(targets_loop):6.1%} planar in average")

    if verbose: print(f"Generated graphs are at {sum(targets) / len(targets):6.1%} planar in average")

    return graphs, targets

def get_trained_model(times, pulses, graphs, targets, verbose=True):
    # 1. Generate final states
    states = return_list_of_states(graphs, times, pulses, evol='ising', verbose=verbose)

    # 2. Compute probability distribution, their distance matrix and the kernel
    observables_memoizer = Memoizer(observable)
    energies_masses, energies = return_energy_distribution(graphs, states, observables_memoizer.get_observable, return_energies=True, verbose=verbose)
    matrix = return_js_square_matrix(energies_masses)
    K = np.exp(-matrix)

    # 3. Fit the model
    # 3.a Meta-parameter
    C_list = np.logspace(-3, 3, 7)
    param_grid = {'C': C_list}

    # 3.b Scoring
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average='weighted')
    }

    # 3.c Cross-validation
    skf = RepeatedStratifiedKFold(10, 10, random_state=47)
    estimator = svm.SVC(kernel='precomputed', random_state=76)
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=skf, refit='accuracy', n_jobs=-1)

    # 3.d Training
    result = grid_search.fit(K, targets).cv_results_

    if verbose:
        i_best = np.argmax(result['mean_test_accuracy'])
        print(f"Accuracy: {result['mean_test_accuracy'][i_best]:.3%} ± {result['std_test_accuracy'][i_best]:.3%} for C={C_list[i_best]}")
        plt.plot(C_list, result['mean_test_accuracy'])
        plt.ylabel('Mean test accuracy')
        plt.xlabel('C')
        plt.xscale('log') 
        plt.title('Results of cross-validation on C')
        plt.show()

    return grid_search, energies_masses, energies

def predict(model, times, pulses, test_graphs, test_targets, energies_masses, energies, verbose=True):
    # 1. Evolve test graphs and compute energy and probability distributions for them
    test_states = return_list_of_states(test_graphs, times, pulses, evol='ising', verbose=verbose)
    observables_memoizer = Memoizer(observable)
    test_energies_masses, test_energies = return_energy_distribution(test_graphs, test_states, observables_memoizer.get_observable, return_energies=True, verbose=True)
    energies_masses, _energies_masses = merge_energies(energies_masses, energies, test_energies_masses, test_energies)

    # 2. Compute distances and kernel
    matrix = return_js_dist_matrix(_energies_masses, energies_masses, verbose=True)
    K = np.exp(-matrix)

    # Make model predict
    return test_targets, model.predict(K), model.score(K, test_targets)

def has_only_one_class(y_true):
    return len(np.unique(y_true)) == 1

def empty_score():
    return {'precision': 1,
             'recall': 1,
             'f1-score': 1,
             'support': 0}

def analyse_pred(y_true, y_pred, score, metric='f1-score', verbose=True):
    if has_only_one_class(y_true):
        if verbose: print('Warning: only one class.')
        return empty_score(), empty_score()
    if verbose:
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=['Non-planar', 'Planar']))
    scores = classification_report(y_true, y_pred, target_names=['Non-planar', 'Planar'], output_dict=True)
    if verbose: print(f"\n\t### Score: {score:.2%} ({metric} for planar: {scores['Planar'][metric]:.3f} & non-planar: {scores['Non-planar'][metric]:.3f})")
    return scores['Planar'], scores['Non-planar']

def test_suite(times, pulses, train_ns, train_nbs, generator, verbose=True, seed=None, metric = 'f1-score',
                test_ns=[], test_nbs=[], test_big=True, test_ramping_max_n=10):
    assert metric in ['precision', 'recall', 'f1-score']
    if seed is not None: np.random.seed(seed)

    if verbose: print(f"\n\t{BOLD}# 1. Generate train and test datasets{STYLE_END}\n")
    graphs, targets = generate_graphs(train_ns, train_nbs, generator=generator, verbose=verbose)
    
    if verbose: print(f"\n\t{BOLD}# 2. Train model{STYLE_END}\n")
    model, energies_masses, energies = get_trained_model(times, pulses, graphs, targets, verbose=verbose)
    if verbose: print(model)

    if verbose: print(f"\n\t{BOLD}# 3. Smoke-test model{STYLE_END}\n")
    y_true, y_pred, score = predict(model, times, pulses, graphs, targets, energies_masses, energies, verbose=verbose)
    analyse_pred(y_true, y_pred, score, metric=metric, verbose=verbose)
    
    if verbose: print(f"\n\t{BOLD}# 4. Investigate capabilities{STYLE_END}\n")
    if len(test_ns) > 0 and len(test_ns) == len(test_nbs):
        if verbose: print(f"\n\t{BOLD}# 4. Investigate generalisation capabilities{STYLE_END}\n")
        n, N = 20, 10
        if verbose: print(f"\n\t#   - Test set {test_ns} nodes in numbers {test_nbs}\n")
        test_graphs, test_targets = generate_graphs(test_ns, test_nbs, generator=generator)
        analyse_pred(*predict(model, times, pulses, test_graphs, test_targets, energies_masses, energies, verbose=verbose), metric=metric, verbose=verbose)


    if test_big:
        n, N = 20, 10
        if verbose: print(f"\n\t#   - Generalisation: {N} graphs of {n} nodes\n")
        test_graphs, test_targets = generate_graphs([n], [N], generator=generator)
        analyse_pred(*predict(model, times, pulses, test_graphs, test_targets, energies_masses, energies, verbose=verbose), metric=metric, verbose=verbose)

    scores = []
    if test_ramping_max_n is not None:
        N = 1000
        if verbose: print(f"\n\t#   - Generalisation: Ramping up number of nodes\n")
        ns = range(3, test_ramping_max_n)
        for n in ns:
            if verbose: print(f"\n\t### {n} NODES ###")
            test_graphs, test_targets = generate_graphs([n], [N], generator='binomial')
            y_true, y_pred, score = predict(model, times, pulses, test_graphs, test_targets, energies_masses, energies, verbose=verbose)
            score_p, score_np = analyse_pred(y_true, y_pred, score, metric=metric, verbose=verbose)
            scores.append((score, score_p[metric], score_np[metric]))
        
        # Plot scores
        pd.DataFrame(scores, columns=['Global accuracy', f'Planar {metric}', f'Non-planar {metric}'], index=ns).plot()
        plt.ylabel('Scores')
        plt.xlabel('Number of nodes')
        plt.legend()
        plt.show()

    return scores


### DGL
def generate_graphs_dgl(nb_nodes: list, N_samples: list, generator='uniform_edges', verbose=False):
    graphs, targets = generate_graphs(nb_nodes, N_samples, generator=generator, verbose=verbose)
    graphs = list(map(dgl.from_networkx, graphs))
    targets = list(map(int, targets))
    return graphs, targets

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_trained_model_dgl(graphs, targets, batch_size=32, epochs=100, verbose=False):
    trainset = list(zip(graphs, targets))
    # Use PyTorch's DataLoader and the collate function defined before.
    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Create model
    num_classes = 2
    model = Classifier(1, 256, num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        if verbose: print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
    
    if verbose:
        plt.title('Cross-entropy averaged over mini-batches')
        plt.xlabel('Batch')
        plt.ylabel('Cross-entropy')
        plt.plot(epoch_losses)
        plt.show()
    
    return model

def predict_dgl(model, graphs, targets, verbose=False):
    model.eval()
    test_X, test_Y = graphs, targets
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    if verbose: print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    argmax_score = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    if verbose: print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        argmax_score * 100
    ))
    return test_Y, argmax_Y, argmax_score

def test_suite_dgl(train_ns, train_nbs, generator, verbose=True, seed=None):
    metric = 'precision'  # in precision, recall, f1-score
    if seed is not None: np.random.seed(seed)

    if verbose: print(f"\n\t{BOLD}# 1. Generate train and test datasets{STYLE_END}\n")
    graphs, targets = generate_graphs_dgl(train_ns, train_nbs, generator=generator, verbose=verbose)

    if verbose: print(f"\n\t{BOLD}# 2. Train model{STYLE_END}\n")
    model = get_trained_model_dgl(graphs, targets, batch_size=32, epochs=100, verbose=verbose)
    if verbose: print(model)  # ToDo: print summary

    if verbose: print(f"\n\t{BOLD}# 3. Smoke-test model{STYLE_END}\n")
    y_true, y_pred, score = predict_dgl(model, graphs, targets, verbose=verbose)
    analyse_pred(y_true, y_pred, score, metric=metric, verbose=verbose)
    
    if verbose: print(f"\n\t{BOLD}# 4. Investigate generalisation capabilities{STYLE_END}\n")
    n, N = 18, 10
    if verbose: print(f"\n\t#   a. {N} graphs of {n} nodes\n")
    test_graphs, test_targets = generate_graphs_dgl([n], [N], generator=generator)
    analyse_pred(*predict_dgl(model, test_graphs, test_targets, verbose=verbose), metric=metric, verbose=verbose)

    N = 1000
    if verbose: print(f"\n\t#   b. Ramping up number of nodes\n")
    scores = []
    ns = range(3, 101)
    for n in ns:
        if verbose: print(f"\n\t### {n} NODES ###")
        test_graphs, test_targets = generate_graphs_dgl([n], [N], generator='binomial', verbose=verbose)
        y_true, y_pred, score = predict_dgl(model, test_graphs, test_targets, verbose=verbose)
        score_p, score_np = analyse_pred(y_true, y_pred, score, metric=metric, verbose=verbose)
        scores.append((score, score_p[metric], score_np[metric]))
    
    # Plot scores
    pd.DataFrame(scores, columns=['Global accuracy', f'Planar {metric}', f'Non-planar {metric}'], index=ns).plot()
    plt.ylabel('Scores')
    plt.xlabel('Number of nodes')
    plt.legend()
    plt.show()

    return scores