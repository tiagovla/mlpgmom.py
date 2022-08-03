from mlpgmom import mesh
from mlpgmom.constants import LIGHT_SPEED, MU_0, EPSILON_0
import numpy as np
from matplotlib import pyplot as plt
import scipy.special
import sys

np.set_printoptions(4)
np.set_printoptions(linewidth=60000)


def j_analytical(phi, n, a, eta_0, k_0):
    r = 2 / (np.pi * eta_0 * k_0 * a)
    sum = 0
    for n in range(n):
        cn = 1 if n == 0 else 2
        num = 1j**n * cn * np.cos(n * phi)
        den = scipy.special.hankel2(n, k_0 * a)
        sum += num / den
    return r * sum


def main():
    eps_r: float = 2
    mu_r: float = 1
    freq: float = 300e6  # Hz
    radius: float = 1  # m
    delh = 1 / 25
    ng_obs: int = 3
    ng_src: int = 4
    phi_i: float = 0  # rads

    omega: float = 2 * np.pi * freq  # rads
    lmbda_0: float = LIGHT_SPEED / freq
    k_0: float = 2 * np.pi / lmbda_0  # m−1
    eta_0: float = np.sqrt(MU_0 / EPSILON_0)

    gp_obs, gw_obs = scipy.special.roots_legendre(ng_obs)
    gp_src, gw_src = scipy.special.roots_legendre(ng_src)

    tri_obs = np.stack(((gp_obs + 1) / 2, (1 - gp_obs) / 2), axis=1)
    tri_src = np.stack(((gp_src + 1) / 2, (1 - gp_src) / 2), axis=1)

    m = mesh.Mesh2D.from_shape(radius, delh, "circle")
    n = m.bc_nodes.shape[0]

    bc_size = m.bc_nodes.shape[0]
    current = np.arange(bc_size)
    next = np.mod(current + 1, bc_size)
    bc_diff = m.bc_nodes[next] - 1 * m.bc_nodes[current]
    bc_delta = np.linalg.norm(bc_diff, axis=1)
    bc_t = bc_diff / bc_delta[:, np.newaxis]
    bc_n = np.stack((bc_t[:, 1], -bc_t[:, 0]), axis=1)
    bc_z = 1

    x = m.bc_nodes
    _, ax = plt.subplots(constrained_layout=True)
    ax.scatter(*m.nodes.T)

    zz = np.zeros((n, n), dtype=complex)
    zt = np.zeros((n, n), dtype=complex)
    tz = np.zeros((n, n), dtype=complex)
    tt = np.zeros((n, n), dtype=complex)
    ve = np.zeros((n, 1), dtype=complex)
    vh = np.zeros((n, 1), dtype=complex)

    cos_phi_i, sin_phi_i = np.cos(phi_i), np.sin(phi_i)

    for seg_obs in range(n):
        obs_now, obs_next = current[seg_obs], next[seg_obs]

        xo_s, xo_e = x[obs_now], x[obs_next]
        wp = gw_obs * bc_delta[seg_obs] / 2
        xp = 0.5 * (np.outer(gp_obs, xo_e - xo_s) + xo_e + xo_s)
        fp = wp[:, np.newaxis] * tri_obs

        incident_field = np.exp(1j * k_0 * (xp[:, 0] * cos_phi_i + xp[:, 1] * sin_phi_i))
        indexes_obs = ((obs_now, obs_next), (0, 0))
        indexes_self = ((obs_now, obs_now), (obs_next, obs_next)), ((obs_now, obs_next), (obs_now, obs_next))
        ve[indexes_obs] += fp.T @ incident_field
        vh[indexes_obs] += fp.T @ incident_field * (np.dot(bc_t[seg_obs], [-sin_phi_i, cos_phi_i]))

        self_term = 0.5 * bc_z * tri_obs.T @ fp / (-1j * k_0 / 4)
        zt[indexes_self] += self_term
        tz[indexes_self] += self_term

        for seg_src in range(n):
            src_now, src_next = current[seg_src], next[seg_src]

            indexes = ((obs_now, obs_now), (obs_next, obs_next)), (
                (src_now, src_next),
                (src_now, src_next),
            )

            xs_s, xs_e = x[src_now], x[src_next]
            wq = gw_src * bc_delta[seg_src] / 2
            xq = 0.5 * (np.outer(gp_src, xs_e - xs_s) + xs_e + xs_s)

            Rho = xp - xq[:, np.newaxis]
            rho = np.linalg.norm(Rho, axis=2)
            Rho = Rho / rho[:, :, np.newaxis]

            prod_zt = np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_src, 1], bc_t[seg_src, 0]]))
            prod_tz = np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_obs, 1], bc_t[seg_obs, 0]]))
            prod_tt = np.dot(bc_t[seg_src], bc_t[seg_obs])

            hankel_02 = scipy.special.hankel2(0, k_0 * rho)
            hankel_12 = scipy.special.hankel2(1, k_0 * rho)

            fq = wq[:, np.newaxis] * tri_src
            zz[indexes] += fq.T @ hankel_02 @ fp
            zt[indexes] += fq.T @ (prod_zt * hankel_12) @ fp
            tz[indexes] += fq.T @ (prod_tz * hankel_12) @ fp
            tt[indexes] += (k_0 * eta_0 / 4) * fq.T @ (prod_tt * hankel_02) @ fp
            tt[indexes] += (1 / (k_0**2 * bc_delta[seg_obs] * bc_delta[seg_src])) * (
                fq.T @ hankel_02 @ fp
            )  # * np.array([[1, -1], [-1, 1]])

    zz *= k_0 * eta_0 / 4
    zt *= -1j * k_0 / 4
    tz *= -1j * k_0 / 4
    tt *= 1 / (eta_0**2)

    b = np.linalg.solve(zz, ve)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    _, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)  # type: ignore
    ax1.plot(theta, np.real(b))
    ax2.plot(theta, np.imag(b))

    j_a = np.zeros_like(theta, dtype=complex)
    for idx, t in enumerate(theta):
        j_a[idx] = j_analytical(t, 40, 1, 120 * np.pi, k_0)
    ax1.plot(theta, np.real(j_a), color="red")
    ax2.plot(theta, np.imag(j_a), color="red")

    # ax.triplot(*m.nodes.T, m.conn)
    # ax.scatter(*m.bc_nodes.T)
    # ax.quiver(m.bc_nodes.x, m.bc_nodes.y, bc_t[:, 0], bc_t[:, 1])
    # ax.quiver(m.bc_nodes.x, m.bc_nodes.y, bc_n[:, 0], bc_n[:, 1])


if __name__ == "__main__":
    main()
    plt.show()
