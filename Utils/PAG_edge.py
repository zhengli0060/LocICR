from dataclasses import dataclass

@dataclass
class PAGEdge:
    """a mark in a PAG"""
    TAIL = 3      # -
    ARROW = 2      # >
    CIRCLE = 1     # o
    NONE = 0       # No edge