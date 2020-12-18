import numpy as np
import tqdm
import matplotlib.pylab as plt


def gradient_step(v: np.ndarray, gradient: np.ndarray, step_size: float) -> np.ndarray:
    """
    Moves 'step_size' in the 'gradient' direction from 'v'
    """
    assert v.shape[0] == gradient.shape[0]
    step = step_size * gradient
    return v + step


def direction(w: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(w, 2)
    return w / mag


def directional_variance(data: np.ndarray, w: np.ndarray) -> float:
    """
    Returns he variance of x in the direction of w
    """
    w_dir = direction(w)
    return sum((v @ w_dir) ** 2 for v in data)  # ((data @ w_dir) ** 2).sum()


def directional_variance_gradient(data: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    The gradient of directional variance with respect to w
    """
    w_dir = direction(w)
    dir_ = np.array([sum(2 * (v @ w_dir) * v[i] for v in data) for i in range(len(w))])
    return dir_


def project(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """return the projection of v onto the direction w"""
    return (v @ w) * w


def remove_projection_from_vector(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """projects v onto w and subtracts the result from v"""
    return v - project(v, w)


def remove_projection(data: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.array([remove_projection_from_vector(v, w) for v in data])


if __name__ == '__main__':
    # pick a random starting point
    n = 1000
    x = np.array([i for i in range(-n//2 + 10, n//2 + 10)])
    y = np.array([20 * np.random.standard_normal() - 0.6 * (i + 10)  for i in x])

    dmd_x = x - x.mean()
    dmd_y = y - y.mean()
    data = np.c_[dmd_x, dmd_y]

    # Start with a random guess
    guess = np.array([1.0 for _ in data[0]])
    step_size = 0.1

    with tqdm.trange(n) as t:
        for _ in t:  # range(100):
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = guess + step_size * gradient  # gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.6f}")

    cpa1 = direction(guess)

    data2 = remove_projection(data, cpa1)

    # plt.scatter(data[:, 0], data[:, 1])
    # plt.scatter(data2[:, 0], data2[:, 1])
    # plt.arrow(0, 0, cpa1[0] * 30, cpa1[1] * 30)

    guess = np.array([1.0 for _ in data2[0]])
    with tqdm.trange(n) as t:
        for _ in t:  # range(100):
            dv = directional_variance(data2, guess)
            gradient = directional_variance_gradient(data2, guess)
            guess = guess + step_size * gradient  # gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.6f}")

    cpa2 = direction(guess)

    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1])
    plt.arrow(0, 0, cpa1[0] * 30, cpa1[1] * 30)
    plt.arrow(0, 0, cpa2[0] * 30, cpa2[1] * 30)
    lim = [-100, 100]
    plt.xlim(lim)
    plt.ylim(lim)

    covmat = np.cov(data.T)
    lbds, eigv = np.linalg.eigh(covmat)
    print(eigv @ cpa1, ': close to +/- e2')  # 2 since increasingly ordered
    print(eigv @ cpa2, ': close to +/- e1')



