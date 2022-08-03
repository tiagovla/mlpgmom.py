from scipy.special import hankel2, jv


def jv_prime(v: int, z: float) -> complex:
    """First derivative of the jv function.

    Parameters
    ----------
    v : int
        Order of the function.
    z : float
        Argument of the function.

    Returns
    -------
    complex :
        Evaluated function.

    """
    return 0.5 * (jv(v - 1, z) - jv(v + 1, z))


def hankel2_prime(v: int, z: float) -> complex:
    """First derivative of the hankel2 function.

    Parameters
    ----------
    v : int
        Order of the function.
    z : float
        Argument of the function.

    Returns
    -------
    complex :
        Evaluated function.

    """
    return 0.5 * (hankel2(v - 1, z) - hankel2(v + 1, z))
