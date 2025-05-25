from itertools import combinations, product
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union
import networkx as nx
from Utils.find_condition_sets import find_circle_collider_path, find_ne_mc, find_non_ancestral_collider_path, is_collider_path
from Utils.MB_TC import MB_TC
from Utils.CI_test import CI_Test 
from Utils.PAG_edge import PAGEdge
from Utils.Type_CR import CausalRelation
from Utils.orient_rules import orient_rules


class LocalCausalLearner:
    """Local learning P_MB """
    
    def   __init__(self,
                 data: np.ndarray,
                 target: int,
                 alpha: float = 0.05,
                 max_k: int = 3,
                 verbose: bool = False):
        """
        Initialize the local causal learner
        
        Args:
            data: Observation data matrix (samples n variables)
            target: Index of the target_X variable
            alpha: Significance level
            max_k: Maximum size of the conditioning set
        """
        self.data = data
        self.target = target
        self.alpha = alpha
        self.max_k = max_k
        self.verbose = verbose
        self.n_samples, self.n_vars = data.shape
        self.all_test = 0
        self.adj_citest_num = 0
        self.mb_citest_num = 0
        self.pag_matrix = np.zeros((self.n_vars, self.n_vars), dtype=int)

        # Separation set dictionary: values are either a separation set (Set[int]) or "adj" indicating an edge exists
        self.sepset: Dict[Tuple[int, int], Union[Set[int], str]] = {}
        
        self.mb_set: Dict[int, Set[int]] = {}
        self.adj_set: Dict[int, Set[int]] = {}
        self.not_adj_in_mb_set: Dict[int, Set[int]] = {}
        self._ci_cache: Dict[tuple[int, int, frozenset], bool] = {}  

        self.waitlist = [target]
        self.donelist = []
        self.suffStat = {'C': np.corrcoef(self.data, rowvar=False), 'n': self.n_samples}


    def _add_edge(self, i: int, j: int) -> None:
        """Add an edge (indicating a direct relationship between two variables)"""
        self.sepset[(i, j)] = "adj"
        self.sepset[(j, i)] = "adj"
        if self.pag_matrix[i, j] == PAGEdge.NONE and self.pag_matrix[j, i] == PAGEdge.NONE:
            self.pag_matrix[i, j] = PAGEdge.CIRCLE
            self.pag_matrix[j, i] = PAGEdge.CIRCLE
        
    def _add_separation(self, i: int, j: int, sep: Set[int]) -> None:
        """Add a separation set"""
        self.sepset[(i, j)] = sep
        self.sepset[(j, i)] = sep
        if self.pag_matrix[i, j] != PAGEdge.NONE and self.pag_matrix[j, i] != PAGEdge.NONE:
            self.pag_matrix[i, j] = PAGEdge.NONE
            self.pag_matrix[j, i] = PAGEdge.NONE

    def _add_arrow_in_edge(self, i: int, j: int) -> None:
        """Add an arrow edge"""
        if self.pag_matrix[i, j] != PAGEdge.ARROW:
            self.pag_matrix[i, j] = PAGEdge.ARROW
        if self.pag_matrix[j, i] == PAGEdge.NONE:
            self.pag_matrix[j, i] = PAGEdge.CIRCLE

    def is_adj(self, i: int, j: int) -> bool:
        """Check if there is an edge between two variables"""
        return (i, j) in self.sepset and self.sepset[(i, j)] == "adj"
    
    def is_not_adj(self, i: int, j: int) -> bool:
        """Check if there is no edge between two variables"""
        return (i, j) in self.sepset and self.sepset[(i, j)] != "adj"
    
    def is_V_struct(self, i: int, j: int, k: int) -> bool:
        """Check if there is a V structure"""
        if self.is_adj(i, j) and self.is_adj(j, k) and self.is_not_adj(i, k):
            if self.pag_matrix[i, j] == PAGEdge.ARROW and self.pag_matrix[k, j] == PAGEdge.ARROW:
                return True
        return False
    
    def get_sepset(self, i: int, j: int) -> Union[Set[int], str, bool]:
        """Retrieve the separating set between two variables; return None if an edge exists"""
        if (i, j) not in self.sepset:
            return False
        sep = self.sepset[(i, j)]
        return "adj" if sep == "adj" else sep

    def _learn_markov_blanket(self, node:int) -> None:
        if node in self.mb_set:
            return
        # if self.oracle_CI_test is not None:
        #     mb, n_test = MB_TC(self.data, node, self.alpha, self.oracle_CI_test)
        # else:
        mb, n_test = MB_TC(self.data, node, self.alpha)
        self.mb_set[node] = set(mb)       
        self.mb_citest_num += n_test
        not_mb = set(range(self.n_vars)) - set(mb)
        for v in not_mb:
            self._add_separation(node, v, set(mb))

    def _get_target_mb_set(self) -> Set[int]:
        """Get the Markov blanket of the target variable"""
        return self.mb_set[self.target]
    
    def test_independence(self,a: int, b: int, condition_set: Union[Set[int], Tuple[int]]) -> bool:
        """Test if a and b are independent given condition_set"""
        # ensure a < b, so the cache key is consistent
        if a > b:
            a, b = b, a
        key = (a, b, frozenset(condition_set))
        
        result = self._ci_cache.get(key)
        if result is None:
            CI, _ = CI_Test(a, b, list(condition_set), self.suffStat, self.alpha)
            self.adj_citest_num += 1
            self._ci_cache[key] = CI
            return CI
        return result



    def _learn_mb_structure(self, node: int) -> None:
        """Learn the Local MB structure of the target node and update the global PAG matrix"""
        if self.verbose:
            print(f"learning mb structure of {node}")
        mb_plus = self.mb_set[node] | {node}
        if node not in self.adj_set:
            self.adj_set[node] = set()
        if node not in self.not_adj_in_mb_set:
            self.not_adj_in_mb_set[node] = set()
        
        # if len(mb_plus) > 15:
        #     warnings.warn(f"mb_plus[{node}] is too large: {len(mb_plus)}")
        
        complete_graph = nx.Graph()
        complete_graph.add_nodes_from(mb_plus)
        complete_graph.add_edges_from(combinations(mb_plus, 2))
        
        # initialize the local PAG matrix
        local_pag = np.full((self.n_vars, self.n_vars), PAGEdge.NONE)
        
        sep_size = 0
        while sep_size <= self.max_k:
            can_continue = False  # 
            for a, b in complete_graph.edges():
                if self.is_not_adj(a, b):
                    complete_graph.remove_edge(a, b)
                    continue
                if self.is_adj(a, b):
                    continue
                if len(mb_plus) > 10:
                    candidate_set_1 = set(complete_graph.neighbors(a)) | set(complete_graph.neighbors(b)) - {a, b}
                    if b not in self.mb_set:
                        self._learn_markov_blanket(b)
                    candidate_set_2 = set(complete_graph.neighbors(a)) & self.mb_set[b] - {a, b}
                    candidate_set_3 = set(complete_graph.neighbors(a)) - {a, b}
                    candidate_set_4 = set(complete_graph.neighbors(b)) - {a, b}
                    candidate_set = min(candidate_set_1, candidate_set_2, candidate_set_3, candidate_set_4, key=len)
                else:
                    candidate_set_1 = set(complete_graph.neighbors(a)) | set(complete_graph.neighbors(b)) - {a, b}
                    if b not in self.mb_set:
                        self._learn_markov_blanket(b)
                    candidate_set_2 = set(complete_graph.neighbors(a)) & self.mb_set[b] - {a, b}
                    candidate_set = min(candidate_set_1, candidate_set_2, key=len)
                if len(candidate_set) >= sep_size:
                    can_continue = True  # whether continue to find removed edges
                    for sepset in combinations(candidate_set, sep_size):
                        if self.test_independence(a, b, set(sepset)):
                            self._add_separation(a, b, set(sepset))
                            complete_graph.remove_edge(a, b)
                            break
            if not can_continue:
                break  
            sep_size += 1
        for a, b in complete_graph.edges():
            local_pag[a, b] = PAGEdge.CIRCLE
            local_pag[b, a] = PAGEdge.CIRCLE
        
        def is_adj(a:int,b:int) -> bool:
            return local_pag[a, b] != PAGEdge.NONE and local_pag[b, a] != PAGEdge.NONE
        
        adjacency = {(a, b): is_adj(a, b) for a in mb_plus for b in mb_plus if a != b}

        # update adj_set and not_adj_in_mb_set of node
        for a in self.mb_set[node]:
            if adjacency.get((a, node), False):
                self._add_edge(node, a)
                self.adj_set[node].add(a)
            else:
                self.not_adj_in_mb_set[node].add(a)
        if self.verbose:
            print(f"node:{node},adj_set:{self.adj_set[node]},non_adj_in_mb_set:{self.not_adj_in_mb_set[node]}")
        # V structure orientation
        def confirm_v_structure(a: int, b: int, c: int) -> bool:
            """Confirm whether the three vertices form a V structure a->b<-c"""

            if adjacency.get((a, c), False) or (b in self.sepset.get((a, c), [])):
                return False

            if not adjacency.get((a, b), False) or not adjacency.get((b, c), False):
                return False

            local_pag[a, b] = PAGEdge.ARROW
            local_pag[c, b] = PAGEdge.ARROW
            local_pag[b, a] = local_pag[b, a] if local_pag[b, a] != PAGEdge.NONE else PAGEdge.CIRCLE
            local_pag[b, c] = local_pag[b, c] if local_pag[b, c] != PAGEdge.NONE else PAGEdge.CIRCLE
            return True

        # find correct V structures in the local PAG
        V_structures = []
        for b in mb_plus:
            for a, c in combinations(mb_plus - {b}, 2):
                if confirm_v_structure(a, b, c):
                    V_structures.append((a, b, c))
                    if self.verbose:
                        print(f"find {a} -> {b} <- {c}")


        candidate_collider = []
        for a, b, c in V_structures:
            if node in {a, b, c}:
                self._add_arrow_in_edge(a, b)
                self._add_arrow_in_edge(c, b)
            else:
                candidate_collider.append((a, b, c))

        collider_path = []
        for a, b, c in candidate_collider:
            for path in nx.all_simple_paths(complete_graph, source=node, target=c):
                if len(path) > 3 and path[-2] == b and path[-3] == a and is_collider_path(local_pag, path):
                    collider_path.append(path)
            for path in nx.all_simple_paths(complete_graph, source=node, target=a):
                if len(path) > 3 and path[-2] == b and path[-3] == c and is_collider_path(local_pag, path):
                    collider_path.append(path)

        # update the global PAG matrix
        for path in collider_path:
            for i in range(len(path) - 2):
                a, b, c = path[i], path[i + 1], path[i + 2]
                self._add_arrow_in_edge(a, b)
                self._add_arrow_in_edge(c, b)

        if self.verbose:
            print(f"learning mb structure of {node} done")
 
    def _mb_strcuture_learner(self) -> np.ndarray:
        
        while (self.waitlist):
            node = self.waitlist.pop(0)
            self._learn_markov_blanket(node)
            self._learn_mb_structure(node)
            self._orient_edges()
            self.donelist.append(node)
            for n in self.adj_set[node]:
                if n not in self.donelist and n not in self.waitlist:
                    self.waitlist.append(n)
            if all(v in self.donelist for v in self.adj_set[self.target]):
                if self.verbose:
                    print(f'-----testing--rules---')
                if self._stop_rule_one() or self._stop_rule_three():
                    if self.verbose:
                        print(f'-----testing--rules--get-')
                    break
                else:
                    if self.verbose:
                        print(f'-----testing--rules--not-get--')
   
        return self.pag_matrix
    
    def _stop_rule_one(self) -> bool:
        # stop one 
        mb_plus_set = self.mb_set[self.target] | {self.target}
        for a, b in combinations(mb_plus_set, 2):
            if self.get_sepset(a, b) == False:
                return False
            if (self.is_adj(a, b) and 
                (self.pag_matrix[a, b] == PAGEdge.CIRCLE or 
                 self.pag_matrix[b, a] == PAGEdge.CIRCLE)):
                return False
        return True
    
    def _stop_rule_three(self) -> bool:
        def stop_condition_three(T, done=None, maxdepth=5, depth=1):
            if depth > maxdepth:
                return True   
            if done is None:
                done = set()  
            done.add(T)
            
            adj_T = np.where(self.pag_matrix[T, :] != PAGEdge.NONE)[0]
            adj_T = np.setdiff1d(adj_T, list(done))
            
            if not adj_T.size:
                return T in self.donelist

            for adj in adj_T:
                if self.pag_matrix[T, adj] != PAGEdge.ARROW:  
                    if not stop_condition_three(adj,  done, maxdepth, depth + 1):
                        return False

            return True

        mb_plus_set = self.mb_set[self.target] | {self.target}
        for a, b in combinations(mb_plus_set, 2):
            if self.get_sepset(a, b) == False:
                return False

        # Check if each CIRCLE satisfies stop_condition_three
        for a, b in combinations(mb_plus_set, 2):
            if (self.is_not_adj(a, b) or 
            (self.is_adj(a, b) and 
             self.pag_matrix[a, b] != PAGEdge.CIRCLE and 
             self.pag_matrix[b, a] != PAGEdge.CIRCLE)):
                continue
            if self.pag_matrix[a, b] == PAGEdge.CIRCLE and self.pag_matrix[b, a] == PAGEdge.ARROW:
                if not stop_condition_three(b,done={a}):
                    return False
            elif self.pag_matrix[a, b] == PAGEdge.ARROW and self.pag_matrix[b, a] == PAGEdge.CIRCLE:
                if not stop_condition_three(a,done={b}):
                    return False
            else:
                if not stop_condition_three(a):
                    return False
        return True
    
    
    def _orient_edges(self) -> None:
        all_sepset = np.empty((self.n_vars, self.n_vars), dtype=object)
        find_edge = np.full((self.n_vars, self.n_vars), False, dtype=bool)
        for (i, j), sep in self.sepset.items():
            find_edge[i, j] = True
            find_edge[j, i] = True  
            if sep != 'adj':
                all_sepset[i, j] = sep
                all_sepset[j, i] = sep  
        self.pag_matrix = orient_rules(self.pag_matrix,all_sepset,find_edge)

    
    def _find_alpha_set(self) -> List[int]:
        #   NDE_MB(target_X)
        alpha_sepset = set(np.where(self.pag_matrix[:,self.target] == PAGEdge.ARROW)[0])
        candidate_set = self.mb_set[self.target] - self.adj_set[self.target]
        updated = True
        
        while updated:
            updated = False
            for V in list(candidate_set):
                for size in range(min(len(alpha_sepset) + 1, self.max_k + 1)):
                    found_independent = False
                    for Z in combinations(alpha_sepset, size):
                        if self.test_independence(self.target, V, set(Z)):
                            alpha_sepset.add(V)
                            candidate_set.remove(V)
                            updated = True
                            found_independent = True
                            break
                    if found_independent:
                        break
        
        return list(alpha_sepset)

    def _is_invariant_non_ancestor(self, v, alpha_sepset) -> bool:
        if v in alpha_sepset:
            return True
        else:
            return False
        
    def _find_condition_sets(self) -> Tuple[Set[int], Set[int], Dict[frozenset, set]]:
        """
        Find the augmented parents, augmented neighbors, and augmented neighbors sets for a target vertex.

        Returns:
            Tuple containing:
            - augmented parents of the target vertex
            - augmented neighbors of the target vertex
            - augmented neighbors sets for each maximal clique
        """
        # find NDE_MB(target_X)
        alpha_sepset = self._find_alpha_set()
        pag_matrix = np.zeros_like(self.pag_matrix)  
        target = self.target
        mb_plus = list(self.mb_set[self.target] | {self.target})
        # P_MB <- P over MB_PLUS
        for i in mb_plus:
            for j in mb_plus:
                pag_matrix[i, j] = self.pag_matrix[i, j]
        

        num = pag_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(num))
        edges = [(i, j) for i in range(num) for j in range(i+1, num) if pag_matrix[i, j] != 0]
        G.add_edges_from(edges)

        
        augmented_parents = set()
        can_pa_star = self.mb_set[self.target]
        for node in can_pa_star:
            if node == target:
                continue
            non_an_col_paths = find_non_ancestral_collider_path(pag_matrix, target, node, G)
            if not non_an_col_paths:
                continue
            for path in non_an_col_paths:
                if all(self._is_invariant_non_ancestor(v,alpha_sepset) for v in path[2:]):
                    augmented_parents.add(node)
                    break


        augmented_neighbors = set()
        all_poss_ne = self.mb_set[self.target] - augmented_parents
        for node in all_poss_ne:
            if node == target:
                continue
            circle_collider_paths = find_circle_collider_path(pag_matrix, target, node, G)
            if not circle_collider_paths:
                continue
            for path in circle_collider_paths:
                if all(self._is_invariant_non_ancestor(v,alpha_sepset) for v in path[2:]):
                    augmented_neighbors.add(node)
                    break

        
        maximal_cliques = find_ne_mc(pag_matrix, target)
        augmented_neighbors_M_sets = {}
        for clique in maximal_cliques:
            clique_set = frozenset(clique)
            augmented_neighbors_set = set(clique)
            for node in all_poss_ne:
                if (node == target) or (node in augmented_neighbors_set):
                    continue
                circle_collider_paths = find_circle_collider_path(pag_matrix, target, node, G)
                if not circle_collider_paths:
                    continue
                for path in circle_collider_paths:
                    if path[1] not in clique_set:
                        continue
                    if all(self._is_invariant_non_ancestor(v,alpha_sepset) for v in path[2:]):
                        augmented_neighbors_set.add(node)
                        break
            augmented_neighbors_M_sets[clique_set] = augmented_neighbors_set

        return augmented_parents, augmented_neighbors, augmented_neighbors_M_sets
    
    def _causal_relation_identify(self,target_Y:int) -> Union[CausalRelation,int]:
        if target_Y == self.target:
            raise ValueError('target_Y is the same as the target_X')
        def is_node_in_sets(target_Y, augmented_parents, augmented_neighbors, augmented_neighbors_M_sets):
            if target_Y in augmented_parents:
                return True
            if target_Y in augmented_neighbors:
                return True
            for neighbors_set in augmented_neighbors_M_sets.values():
                if target_Y in neighbors_set:
                    return True
            return False
        _ = self._mb_strcuture_learner()

        
        if target_Y in self.adj_set[self.target]:
            if self.pag_matrix[target_Y,self.target] == PAGEdge.ARROW:
                return CausalRelation.invariant_non_ancestor
            elif self.pag_matrix[target_Y,self.target] == PAGEdge.TAIL and self.pag_matrix[self.target,target_Y] == PAGEdge.ARROW:
                return CausalRelation.explicit_invariant_ancestor
            else:
                return CausalRelation.possible_ancestor
        else:
            augmented_parents, augmented_neighbors, augmented_neighbors_M_sets = self._find_condition_sets()
            if is_node_in_sets(target_Y, augmented_parents, augmented_neighbors, augmented_neighbors_M_sets):
                return CausalRelation.invariant_non_ancestor
            else:
                if self.test_independence(target_Y,self.target,augmented_parents):
                    return CausalRelation.invariant_non_ancestor
                elif not self.test_independence(target_Y,self.target,augmented_parents| augmented_neighbors):
                    return CausalRelation.explicit_invariant_ancestor
                else:
                    ne_star_M_values = list(augmented_neighbors_M_sets.values())
                    for ne_star_M in ne_star_M_values:
                        if self.test_independence(target_Y,self.target,augmented_parents|ne_star_M):
                            return CausalRelation.possible_ancestor
                    return CausalRelation.implicit_invariant_ancestor

    def _get_citest_num(self) -> int:
        return self.adj_citest_num + self.mb_citest_num
    

    

    
