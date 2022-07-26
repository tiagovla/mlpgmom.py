import numpy as np
import scipy.spatial
from mlpgmom.geogen import circle, square


class Nodes(np.ndarray):
    @property
    def x(self):
        return self[:, 0]

    @property
    def y(self):
        return self[:, 1]


class Mesh2D:
    def __init__(self, nodes: np.ndarray, bc: np.ndarray, conn: np.ndarray):
        self.nodes = nodes.view(Nodes)
        self.bc = bc
        self.conn = conn

    @classmethod
    def from_shape(cls, radius, delh, type: str = "circle"):
        if type == "square":
            nodes, bc = square(radius, delh)
        elif type == "circle":
            nodes, bc = circle(radius, delh)
        else:
            raise Exception("Shape not supported.")
        conn = scipy.spatial.Delaunay(nodes).simplices
        return cls(nodes, bc, conn)

    @property
    def bc_nodes(self):
        return self.nodes[self.bc, :].view(Nodes)
