"""
from a continuous adjacency matrix to a discrete directed acyclic graph, represented by a discrete adjacency matrix.
"""

import torch
import random
import numpy as np

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def softmax(probs):
    """
    Args:
        probs: np.ndarray, shape (n, ), probabilities.
    Returns:
        probs: np.ndarray, shape (n, ), probabilities after softmax.
    zero terms are not considered in the softmax operation.
    """
    probs = np.array(probs)
    probs = np.exp(probs)
    for i in range(len(probs)):
        if probs[i] == np.exp(0): # 0 means 0, no edge
            probs[i] = 0
    probs = probs / np.sum(probs)
    assert sum(probs) >= 0.999 and sum(probs) <= 1.001
    return probs

# probs = [0.18, 0.23, 0, 0.28]
# print(softmax(probs))

def top_p_sampling_selection(probs, top_p_threshold):
    """
    Args:
        probs: np.ndarray, shape (n, ), probabilities.
        top_p_threshold: float, threshold for top-p sampling-based operations.
    Returns:
        selected_indice: int, selected indice.
    sum up probs from high to low until reach top_p_threshold, renormalize probs with softmax, sample from probs and return the selected index.
    """
    assert sum(probs) >= 0.999 and sum(probs) <= 1.001
    prob_index_lists = list(zip(probs, range(len(probs))))
    prob_index_lists.sort(reverse = True)
    cum_prob = 0
    selected_indice = []
    for prob, index in prob_index_lists:
        cum_prob += prob
        selected_indice.append(index)
        if cum_prob > top_p_threshold:
            break
    selected_probs = [probs[i] for i in selected_indice]
    selected_probs = np.array(selected_probs)
    selected_probs = softmax(selected_probs)
    selected_index = np.random.choice(selected_indice, p = selected_probs)
    return selected_index

# probs = [0.18, 0.23, 0.31, 0.28]
# top_p_threshold = 0.9
# print(top_p_sampling_selection(probs, top_p_threshold))    

def graph_decode(adjacency_matrix, top_p_threshold = 0): # 0 for deterministic by default
    """
    Args:
        adjacency_matrix: np.ndarray, shape (n, n), continuous adjacency matrix.
        top_p_threshold: float, threshold for top-p sampling-based operations.
    Returns:
        discrete_adjacency_matrix: np.ndarray, shape (n, n), discrete adjacency matrix.
    """
    n = adjacency_matrix.shape[0]
    discrete_adjacency_matrix = np.zeros((n, n))
    remaining_nodes = list(range(n))
    existing_nodes = []

    # diagnoals must be zeros
    assert np.all(np.diag(adjacency_matrix) == 0)
    
    # select end point
    out_degrees = np.sum(adjacency_matrix, axis = 1)
    out_degrees = np.array([1 / value for value in out_degrees])
    out_degrees = softmax(out_degrees)
    end_point = top_p_sampling_selection(out_degrees, top_p_threshold)
    existing_nodes.append(end_point)
    remaining_nodes.remove(end_point)

    # iteratively selecting and adding one point
    while len(remaining_nodes) > 0:
        out_degrees = np.sum(adjacency_matrix, axis = 1)
        # existing node to 0
        for node in existing_nodes:
            out_degrees[node] = 0
        out_degrees = softmax(out_degrees)
        selected_node = top_p_sampling_selection(out_degrees, top_p_threshold)
        # select an existing node to connect to
        out_degree_to_existing_nodes = adjacency_matrix[selected_node]
        for node in remaining_nodes:
            out_degree_to_existing_nodes[node] = 0
        out_degree_to_existing_nodes = softmax(out_degree_to_existing_nodes)
        selected_existing_node = top_p_sampling_selection(out_degree_to_existing_nodes, top_p_threshold)
        # update the state
        discrete_adjacency_matrix[selected_node, selected_existing_node] = 1
        existing_nodes.append(selected_node)
        # print(selected_node, remaining_nodes)
        remaining_nodes.remove(selected_node)

    # the one with no out degrees is the end point
    # the one(s) with no in degrees is the start point
    return discrete_adjacency_matrix

def topological_sort(discrete_adjacency_matrix):
    """
    Args:
        discrete_adjacency_matrix: np.ndarray, shape (n, n), discrete adjacency matrix.
    Returns:
        topological_order: list, topological order of the graph.
    """
    n = discrete_adjacency_matrix.shape[0]
    in_degrees = np.sum(discrete_adjacency_matrix, axis = 0)
    out_degrees = np.sum(discrete_adjacency_matrix, axis = 1)
    topological_order = []
    while len(topological_order) < n:
        for node in range(n):
            if in_degrees[node] == 0:
                topological_order.append(node)
                in_degrees[node] = -1
                for i in range(n):
                    if discrete_adjacency_matrix[node, i] == 1:
                        in_degrees[i] -= 1
    return topological_order

# adjacency_matrix = np.array(
#     [
#         [0,8,3,6],
#         [2,0,4,7],
#         [4,3,0,5],
#         [4,5,2,0]
#     ]
# )

# result = graph_decode(adjacency_matrix, 1)
# print(result)

# topological_order = topological_sort(result)
# print(topological_order)

# next: organize the data structure in each search/: model and graph folders
# next: inference on a discrete DAG to get a utility function

# adjacency_matrix = np.load('search/test_{0.3}_{0.4}_{0.2}_{0.1}_{0.7}_a100-16-4-bk-2/all_time_best/graph/adjacency.npy')
# discrete_adjacency_matrix = graph_decode(adjacency_matrix, 0)
# print(discrete_adjacency_matrix)