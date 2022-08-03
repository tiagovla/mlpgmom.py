import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from mlpgmom.constants import EPSILON_0, MU_0
from mlpgmom.mesh import Mesh2D
from mlpgmom.mom import mom_tm
from mlpgmom import mie

np.set_printoptions(4)
np.set_printoptions(linewidth=150)


def main():
    freq = 300e6
    omega = 2 * np.pi * freq
    eps_r = 2
    mu_r = 1
    radius = 1
    delh = 1 / 15
    ng_obs = 3
    ng_src = 4

    k_0 = omega * np.sqrt(MU_0 * EPSILON_0)
    eta_0 = np.sqrt(MU_0 / EPSILON_0)

    k_d = np.sqrt(eps_r * mu_r) * k_0
    eta_d = np.sqrt(mu_r / eps_r) * eta_0

    m = Mesh2D.from_shape(radius, delh, "circle")
    geo_out = m.mom_geo(outwards=True)
    geo_in = m.mom_geo(outwards=False)

    zz0, zt0, tz0, tt0, ve, vh = mom_tm(geo_out, k_0, eta_0, ng_obs, ng_src)
    zz1, zt1, tz1, tt1, _, _ = mom_tm(geo_in, k_d, eta_d, ng_obs, ng_src)

    z = np.vstack((np.hstack((zz0 + zz1, zt0 + zt1)), np.hstack((tz0 + tz1, tt0 + tt1))))
    v = np.vstack((ve, vh))

    b = np.linalg.solve(z, v)

    theta = np.linspace(0, 2 * np.pi, geo_out.size, endpoint=False)

    _, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)  # type: ignore
    size = zz0.shape[0]

    ax1.plot(theta * 180 / np.pi, np.abs(b[:size]))
    ax2.plot(theta * 180 / np.pi, np.abs(b[size:]))
    ax1.set(xlabel=r"Angle", ylabel="i_j")
    ax2.set(xlabel=r"Angle", ylabel="i_m")
    ax1.grid(True)
    ax2.grid(True)

    i_j, i_m = mie.current(m.bc_nodes, 30, radius, k_0, eps_r, mu_r, "TM")
    ax1.plot(theta * 180 / np.pi, np.abs(i_j))
    ax2.plot(theta * 180 / np.pi, np.abs(i_m))

if __name__ == "__main__":
    main()
    plt.show()
