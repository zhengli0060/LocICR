from dataclasses import dataclass

@dataclass
class CausalRelation:
    invariant_non_ancestor = 0
    possible_ancestor = 1
    explicit_invariant_ancestor = 2
    implicit_invariant_ancestor = 3