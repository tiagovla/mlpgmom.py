import numpy as np
import scipy.special


def mom_tm(
    geo,
    k_0: float,
    eta_0: float,
    ng_obs: int,
    ng_src: int,
    phi_i: float = 0,
):
    if ng_obs + ng_src % 2 == 1:
        raise Exception("Gauss points should not be both even or both odd.")

    nn = geo.nodes.shape[0]
    zz = np.zeros((nn, nn), dtype=complex)
    zt = np.zeros((nn, nn), dtype=complex)
    tz = np.zeros((nn, nn), dtype=complex)
    tt = np.zeros((nn, nn), dtype=complex)
    ve = np.zeros((nn, 1), dtype=complex)
    vh = np.zeros((nn, 1), dtype=complex)

    gp_obs, gw_obs = scipy.special.roots_legendre(ng_obs)
    gp_src, gw_src = scipy.special.roots_legendre(ng_src)

    print(gp_obs)
    print(gw_obs)

    tri_obs = np.stack(((gp_obs + 1) / 2, (1 - gp_obs) / 2), axis=1)
    tri_src = np.stack(((gp_src + 1) / 2, (1 - gp_src) / 2), axis=1)

    current = geo.current
    next = geo.next
    bc_delta = geo.delta
    bc_t = geo.t
    bc_n = geo.n
    bc_z = geo.z

    x = geo.nodes

    cos_phi_i, sin_phi_i = np.cos(phi_i), np.sin(phi_i)

    for seg_obs in range(nn):
        obs_now, obs_next = current[seg_obs], next[seg_obs]

        xo_s, xo_e = x[obs_now], x[obs_next]
        wp = gw_obs * bc_delta[seg_obs] / 2
        xp = 0.5 * (np.outer(gp_obs, xo_e - xo_s) + xo_e + xo_s)
        fp = wp[:, np.newaxis] * tri_obs

        incident_field = np.exp(-1j * k_0 * (xp[:, 0] * cos_phi_i + xp[:, 1] * sin_phi_i), dtype=np.complex64)
        indexes_obs = ((obs_now, obs_next), (0, 0))
        indexes_self = ((obs_now, obs_now), (obs_next, obs_next)), ((obs_now, obs_next), (obs_now, obs_next))
        ve[indexes_obs] += fp.T @ incident_field
        vh[indexes_obs] -= (1/eta_0)*fp.T @ incident_field * (np.dot(bc_t[seg_obs], [-sin_phi_i, cos_phi_i]))

        self_term = 0.5 * bc_z * tri_obs.T @ fp / (-1j * k_0 / 4)
        zt[indexes_self] += self_term
        tz[indexes_self] += self_term

        for seg_src in range(nn):
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

            prod_zt = bc_z * np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_src, 1], bc_t[seg_src, 0]]))
            prod_tz = bc_z * np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_obs, 1], bc_t[seg_obs, 0]]))
            prod_tt = np.dot(bc_t[seg_src], bc_t[seg_obs])

            hankel_02 = scipy.special.hankel2(0, k_0 * rho)
            hankel_12 = scipy.special.hankel2(1, k_0 * rho)

            fq = wq[:, np.newaxis] * tri_src
            zz[indexes] += fq.T @ hankel_02 @ fp
            zt[indexes] += fq.T @ (prod_zt * hankel_12) @ fp
            tz[indexes] += fq.T @ (prod_tz * hankel_12) @ fp
            # tt[indexes] += (k_0 * eta_0 / 4) * fq.T @ (prod_tt * hankel_02) @ fp
            # tt[indexes] +=(k_0 * eta_0 / 4)* (1 / (k_0**2 * bc_delta[seg_obs] * bc_delta[seg_src])) * (fq.T @ hankel_02 @ fp) * np.array([[1, -0], [-0, 0]])
            tt[indexes] += (1 / (k_0**2 * bc_delta[seg_obs] * bc_delta[seg_src])) * (k_0 * eta_0 / 4)* (wq[:, np.newaxis].T @ hankel_02 @ wp[:, np.newaxis]) * np.array([[-1, 1], [1, -1]])

    zz *= k_0 * eta_0 / 4
    zt *= -1j * k_0 / 4
    tz *= -1j * k_0 / 4
    tt /= eta_0**2
    print(eta_0)

    return zz, zt, tz, tt, ve, vh
