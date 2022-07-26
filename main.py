from mlpgmom import mesh
from mlpgmom.constants import LIGHT_SPEED, MU_0, EPSILON_0
import numpy as np
from matplotlib import pyplot as plt


def main():
    eps_r: float = 2
    mu_r: float = 1
    freq: float = 300e6  # Hz
    radius: float = 1  # m
    delh = 1 / 10
    n: int = 168  # number of elements - 1
    ng_obs: int = 3
    ng_src: int = 4

    omega: float = 2 * np.pi * freq  # rads
    lmbda_0: float = LIGHT_SPEED / freq
    k_0: float = 2 * np.pi / lmbda_0  # m−1
    eta_0: float = np.sqrt(MU_0 / EPSILON_0)

    m = mesh.Mesh2D.from_shape(radius, delh, "circle")

    bc_size = m.bc_nodes.shape[0]
    current = np.arange(bc_size)
    next = np.mod(current + 1, bc_size)
    bc_diff = m.bc_nodes[next] - 1 * m.bc_nodes[current]
    bc_delta = np.linalg.norm(bc_diff, axis=1)
    bc_t = bc_diff * bc_delta[:, np.newaxis]
    bc_n = np.stack((bc_t[:, 1], -bc_t[:, 0]), axis=1)
    bc_z = 1

    zz = np.zeros((n-1, n-1), dtype=complex)
    ve = np.zeros((n-1, 1), dtype=complex)

    for seg_obs in range(n - 1):
        for seg_src in range(n - 1):
            ...



    _, ax = plt.subplots(constrained_layout=True)
    ax.scatter(*m.nodes.T)
    ax.triplot(*m.nodes.T, m.conn)
    ax.scatter(*m.bc_nodes.T)
    ax.quiver(m.bc_nodes.x, m.bc_nodes.y, bc_t[:, 0], bc_t[:, 1])
    ax.quiver(m.bc_nodes.x, m.bc_nodes.y, bc_n[:, 0], bc_n[:, 1])


if __name__ == "__main__":
    main()
    # plt.show()
