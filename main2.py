import matplotlib.pyplot as plt
import numpy as np

from mlpgmom import mesh
from mlpgmom.constants import EPSILON_0, LIGHT_SPEED as C, MU_0



def main():
    freq = 300e6
    omega = 2 * np.pi * freq
    eps_r = 2
    mu_r = 1
    radius = 1
    delh = 1 / 10


    m = mesh.Mesh2D.from_shape(radius, delh, "circle")
    print(m)





if __name__ == "__main__":
    main()

# def tm_mom(geo, k, eta, ng_obs, ng_src) -> tuple(np.ndarray, ...):
#
#     zz = np.zeros((n, n), dtype=complex)
#     zt = np.zeros((n, n), dtype=complex)
#     tz = np.zeros((n, n), dtype=complex)
#     tt = np.zeros((n, n), dtype=complex)
#     ve = np.zeros((n, 1), dtype=complex)
#     vh = np.zeros((n, 1), dtype=complex)
#
#     cos_phi_i, sin_phi_i = np.cos(phi_i), np.sin(phi_i)
#
#     for seg_obs in range(n):
#         obs_now, obs_next = current[seg_obs], next[seg_obs]
#
#         xo_s, xo_e = x[obs_now], x[obs_next]
#         wp = gw_obs * bc_delta[seg_obs] / 2
#         xp = 0.5 * (np.outer(gp_obs, xo_e - xo_s) + xo_e + xo_s)
#         fp = wp[:, np.newaxis] * tri_obs
#
#         incident_field = np.exp(1j * k_0 * (xp[:, 0] * cos_phi_i + xp[:, 1] * sin_phi_i))
#         indexes_obs = ((obs_now, obs_next), (0, 0))
#         indexes_self = ((obs_now, obs_now), (obs_next, obs_next)), ((obs_now, obs_next), (obs_now, obs_next))
#         ve[indexes_obs] += fp.T @ incident_field
#         vh[indexes_obs] += fp.T @ incident_field * (np.dot(bc_t[seg_obs], [-sin_phi_i, cos_phi_i]))
#
#         self_term = 0.5 * bc_z * tri_obs.T @ fp / (-1j * k_0 / 4)
#         zt[indexes_self] += self_term
#         tz[indexes_self] += self_term
#
#         for seg_src in range(n):
#             src_now, src_next = current[seg_src], next[seg_src]
#
#             indexes = ((obs_now, obs_now), (obs_next, obs_next)), (
#                 (src_now, src_next),
#                 (src_now, src_next),
#             )
#
#             xs_s, xs_e = x[src_now], x[src_next]
#             wq = gw_src * bc_delta[seg_src] / 2
#             xq = 0.5 * (np.outer(gp_src, xs_e - xs_s) + xs_e + xs_s)
#
#             Rho = xp - xq[:, np.newaxis]
#             rho = np.linalg.norm(Rho, axis=2)
#             Rho = Rho / rho[:, :, np.newaxis]
#
#             prod_zt = np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_src, 1], bc_t[seg_src, 0]]))
#             prod_tz = np.einsum("ijk,k->ij", Rho, np.array([-bc_t[seg_obs, 1], bc_t[seg_obs, 0]]))
#             prod_tt = np.dot(bc_t[seg_src], bc_t[seg_obs])
#
#             hankel_02 = scipy.special.hankel2(0, k_0 * rho)
#             hankel_12 = scipy.special.hankel2(1, k_0 * rho)
#
#             fq = wq[:, np.newaxis] * tri_src
#             zz[indexes] += fq.T @ hankel_02 @ fp
#             zt[indexes] += fq.T @ (prod_zt * hankel_12) @ fp
#             tz[indexes] += fq.T @ (prod_tz * hankel_12) @ fp
#             tt[indexes] += (k_0 * eta_0 / 4) * fq.T @ (prod_tt * hankel_02) @ fp
#             tt[indexes] += (1 / (k_0**2 * bc_delta[seg_obs] * bc_delta[seg_src])) * (
#                 fq.T @ hankel_02 @ fp
#             )  # * np.array([[1, -1], [-1, 1]])
#
#     zz *= k_0 * eta_0 / 4
#     zt *= -1j * k_0 / 4
#     tz *= -1j * k_0 / 4
#     tt *= 1 / (eta_0**2)
