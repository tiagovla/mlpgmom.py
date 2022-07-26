import numpy as np

def circle(radius: float, delh: float = 1 / 10) -> tuple[np.ndarray, ...]:
    nodes = np.zeros((1, 2))
    bc_size = 0
    for r in np.arange(delh, radius + delh, delh):
        n = np.floor(np.pi / (2 * np.arcsin(delh * 0.5 / r)))
        alpha = np.concatenate(((np.pi / n) * np.arange(n + 1), -np.flip((np.pi / n) * np.arange(1, n))))
        layer = r * np.stack((np.cos(alpha), np.sin(alpha)), axis=1)
        nodes = np.concatenate((nodes, layer))
        bc_size = alpha.shape[0]
    bc = np.arange(nodes.shape[0] - bc_size, nodes.shape[0])
    return nodes, bc


def square(radius, delh: float = 1 / 10) -> tuple[np.ndarray, ...]:
    x_, y_ = np.meshgrid(np.arange(-radius + delh, radius, delh), np.arange(-radius + delh, radius, delh))
    nodes = np.stack((x_.flatten(order="F"), y_.flatten(order="F")), axis=1)
    n_side = int(1 + np.ceil(2 * radius / delh))
    s1 = np.stack((np.linspace(radius, -radius, n_side), radius * np.ones(n_side)), axis=1)
    s2 = np.stack((-radius * np.ones(n_side), np.linspace(radius - delh, -radius + delh, n_side)), axis=1)
    s3 = np.stack((np.linspace(-radius, +radius, n_side), -radius * np.ones(n_side)), axis=1)
    s4 = np.stack((radius * np.ones(n_side), np.linspace(-radius + delh, +radius - delh, n_side)), axis=1)
    nodes = np.concatenate((nodes, s1, s2, s3, s4))
    bc_size = np.sum([s.shape[0] for s in (s1, s2, s3, s4)])
    bc = np.arange(nodes.shape[0] - bc_size, nodes.shape[0])
    return nodes, bc
