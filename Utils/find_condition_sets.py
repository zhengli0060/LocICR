import itertools
from typing import Any, Dict, List, Set, Tuple, Union
from networkx import find_cliques
import numpy as np
import networkx as nx
from Utils.PAG_edge import PAGEdge
from Utils.Type_CR import CausalRelation

# find_circle_collider_path, find_ne_mc, find_non_ancestral_collider_path, is_collider_path

def is_collider_path(pag_matrix, path):
    if len(path) == 2:
        return False
    elif len(path) == 3:
        return pag_matrix[path[0], path[1]] == PAGEdge.ARROW and pag_matrix[path[2], path[1]] == PAGEdge.ARROW
    elif len(path) > 3:
        for i in range(1,len(path)-1):   # [0,1,2,3,4]  1,2,3
            # Check if the current node is a collider i.e. i-1 -> i <- i+1
            if not (pag_matrix[path[i-1], path[i]] == PAGEdge.ARROW and pag_matrix[path[i+1], path[i]] == PAGEdge.ARROW):
                return False
        return True
    else:
        raise ValueError('Path length must be greater than or equal to 2')


def is_non_ancestral_collider_path(pag_matrix, path):
    
    if len(path) == 2:
        # <-*
        return pag_matrix[path[1], path[0]] == PAGEdge.ARROW
    elif len(path) >= 3:
        # <-><-*
        if pag_matrix[path[1], path[0]] == PAGEdge.ARROW and is_collider_path(pag_matrix, path):
            return True
        else:
            return False
    else:
        raise ValueError('Path length must be greater than or equal to 2')


def is_circle_collider_path(pag_matrix, path):
    
    if len(path) == 2:
        # o-*
        return pag_matrix[path[1], path[0]] == PAGEdge.CIRCLE
    elif len(path) >= 3:
        # o-><-*
        if pag_matrix[path[1], path[0]] == PAGEdge.CIRCLE and is_collider_path(pag_matrix, path):
            return True
        else:
            return False
    else:
        raise ValueError('Path length must be greater than or equal to 2')
    

def find_non_ancestral_collider_path(pag_matrix, target, node, G, cutoff=None):
    """
    Find all non-ancestral collider paths from target to node
    
    Parameters:
        pag_matrix: PAG adjacency matrix
        target: source node
        node: destination node
        G: undirected graph
        cutoff: optional maximum path length limit, None means no limit
    
    Returns:
        list: all paths that satisfy the non-ancestral collider path criteria
    """
    if not nx.has_path(G, target, node):
        return []
    
    non_an_collider_paths = []
    for path in nx.all_simple_paths(G, source=target, target=node, cutoff=cutoff):
        if len(path) > 2:
            if not (pag_matrix[path[1], path[0]] == PAGEdge.ARROW and pag_matrix[path[0], path[1]] == PAGEdge.ARROW):
                continue
            if not pag_matrix[path[-1], path[-2]] == PAGEdge.ARROW:
                continue
                
        if is_non_ancestral_collider_path(pag_matrix, path):
            non_an_collider_paths.append(path)
            
    return non_an_collider_paths

def find_circle_collider_path(pag_matrix, target, node, G, cutoff=None):
    """
    Find all circle collider paths from target to node
    
    Parameters:
        pag_matrix: PAG adjacency matrix
        target: source node
        node: destination node
        G: undirected graph
        cutoff: optional maximum path length limit, None means no limit
    
    Returns:
        list: all paths that satisfy the circle collider path criteria
    """
    if not nx.has_path(G, target, node):
        return []
    
    circle_collider_paths = []
    for path in nx.all_simple_paths(G, source=target, target=node, cutoff=cutoff):
        if len(path) > 2:
            
            if not pag_matrix[path[1], path[0]] == PAGEdge.CIRCLE:
                continue

            if not (pag_matrix[path[-1], path[-2]] == PAGEdge.ARROW and pag_matrix[path[-3], path[-2]] == PAGEdge.ARROW):
                continue
                
        if is_circle_collider_path(pag_matrix, path):
            circle_collider_paths.append(path)
            
    return circle_collider_paths



def find_ne_mc(pag_matrix:np.ndarray, target:int):
    
    # find all undirected neighbors of target
    undirected_neighbors = np.where((pag_matrix[target, :] != PAGEdge.NONE) & (pag_matrix[:, target] == PAGEdge.CIRCLE))[0]
    
    # create a subgraph, the subgraph should not contain any directed edges
    subg = pag_matrix[np.ix_(undirected_neighbors, undirected_neighbors)]
    subg = np.sign(subg + subg.T)

    # create an undirected graph and keep the original index
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(undirected_neighbors)
    for u, v in itertools.combinations(range(len(undirected_neighbors)), 2):
        if subg[u, v] == 1:
            undirected_graph.add_edge(undirected_neighbors[u], undirected_neighbors[v])
    
    # find all maximal cliques
    max_cliques = list(find_cliques(undirected_graph))
    
    return max_cliques


