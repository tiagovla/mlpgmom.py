import numpy as np
import mlpgmom.constants as cte
from scipy.special import hankel2, jv
from mlpgmom.utils import hankel2_prime, jv_prime


def coeffs(n, a: float, k_0: float, eps_r: float, mu_r: float) -> tuple[np.ndarray, ...]:
    k_d = np.sqrt(eps_r * mu_r) * k_0
    eta_r = np.sqrt(mu_r / eps_r)
    jv_prime_kd = jv_prime(n, k_d * a)
    jv_kd = jv(n, k_d * a)
    jv_k0 = jv(n, k_0 * a)
    hankel2_k0 = hankel2(n, k_0 * a)

    n_1 = jv_k0 * jv_prime_kd
    n_2 = jv_prime(n, k_0 * a) * jv_kd
    d_1 = hankel2_k0 * jv_prime_kd
    d_2 = hankel2_prime(n, k_0 * a) * jv_kd

    an = -(n_1 - eta_r * n_2) / (d_1 - eta_r * d_2)
    cn = (jv_k0 + an * hankel2_k0) / jv_kd
    return an, cn


def total_field(points, n, a, k_0, eps_r, mu_r, pol):
    k_d = np.sqrt(eps_r * mu_r) * k_0
    rho = np.hypot(*points.T)
    phi = np.arctan2(*np.flip(points, axis=1).T)
    size = rho.shape[0]
    nn = np.arange(n)
    en = np.sign(nn) + 1
    u = np.zeros(size, dtype=complex)
    an, cn = coeffs(nn, a, k_0, eps_r, mu_r)
    for chunk in np.array_split(np.arange(size), size // 50 + 1):
        inside = chunk[rho[chunk] < a]
        outside = chunk[rho[chunk] >= a]
        jv_kd = jv(nn[:, np.newaxis], k_d * rho[np.newaxis, inside])
        cos_in = np.cos(nn[:, np.newaxis] * phi[np.newaxis, inside])
        jv_k0 = jv(nn[:, np.newaxis], k_0 * rho[np.newaxis, outside])
        hankel2_k0 = hankel2(nn[:, np.newaxis], k_0 * rho[np.newaxis, outside])
        cos_out = np.cos(nn[:, np.newaxis] * phi[np.newaxis, outside])
        u[inside] = np.sum((en * cn * 1j**-nn)[:, np.newaxis] * jv_kd * cos_in, axis=0)
        u[outside] = np.sum((en * 1j**-nn)[:, np.newaxis] * (jv_k0 + an[:, np.newaxis] * hankel2_k0) * cos_out, axis=0)
    return u


def current(points, n, a, k_0, eps_r, mu_r, pol):
    k_d = np.sqrt(eps_r * mu_r) * k_0
    rho = np.hypot(*points.T)
    phi = np.arctan2(*np.flip(points, axis=1).T)
    eta_d = np.sqrt(mu_r / eps_r) * np.sqrt(cte.MU_0 / cte.EPSILON_0)
    nn = np.arange(n)
    en = np.sign(nn) + 1
    _, cn = coeffs(nn, a, k_0, eps_r, mu_r)

    jv_kd = jv(nn[:, np.newaxis], k_d * rho[np.newaxis, :])
    jv_prime_kd = jv_prime(nn[:, np.newaxis], k_d * rho[np.newaxis, :]) #type: ignore
    cos_term = np.cos(nn[:, np.newaxis] * phi[np.newaxis, :])
    i_m = np.sum((en * cn * 1j**-nn)[:, np.newaxis] * jv_kd * cos_term, axis=0)
    i_j = np.sum((en * cn * 1j ** (-(nn + 1)))[:, np.newaxis] * jv_prime_kd * cos_term, axis=0) / eta_d
    return i_j, i_m


def total_field(points, n, a, k_0, eps_r, mu_r, pol):
    k_d = np.sqrt(eps_r * mu_r) * k_0
    rho = np.hypot(*points.T)
    phi = np.arctan2(*np.flip(points, axis=1).T)
    size = rho.shape[0]
    nn = np.arange(n)
    en = np.sign(nn) + 1
    u = np.zeros(size, dtype=complex)
    an, cn = coeffs(nn, a, k_0, eps_r, mu_r)
    for chunk in np.array_split(np.arange(size), size // 50 + 1):
        inside = chunk[rho[chunk] < a]
        outside = chunk[rho[chunk] >= a]
        jv_kd = jv(nn[:, np.newaxis], k_d * rho[np.newaxis, inside])
        cos_in = np.cos(nn[:, np.newaxis] * phi[np.newaxis, inside])
        jv_k0 = jv(nn[:, np.newaxis], k_0 * rho[np.newaxis, outside])
        hankel2_k0 = hankel2(nn[:, np.newaxis], k_0 * rho[np.newaxis, outside])
        cos_out = np.cos(nn[:, np.newaxis] * phi[np.newaxis, outside])
        u[inside] = np.sum((en * cn * 1j**-nn)[:, np.newaxis] * jv_kd * cos_in, axis=0)
        u[outside] = np.sum((en * 1j**-nn)[:, np.newaxis] * (jv_k0 + an[:, np.newaxis] * hankel2_k0) * cos_out, axis=0)
    return u
