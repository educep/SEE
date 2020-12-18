import random
import numpy as np
from typing import TypeVar, List, Iterator


def gradient_step(v: np.ndarray, gradient: np.ndarray, step_size: float) -> np.ndarray:
    """
    Moves 'step_size' in the 'gradient' direction from 'v'
    :param v: initial point
    :param gradient: direction
    :param step_size: size
    :return: array
    """
    assert v.shape[0] == gradient.shape[0]
    step = step_size * gradient
    return v + step


def linear_gradient(x: float, y: float, theta: np.ndarray) -> np.ndarray:
    """

    :param x: predictor
    :param y: actual value
    :param theta: linear model (slope, intercept)
    :return: gradient
    """
    predicted = theta[0] * x + theta[1]          # the prediction of the model
    error = predicted - y                        # error = predicted - actual
    squared_error = error ** 2                   # We'll minimize squared error
    grad = np.array([2 * error * x, 2 * error])  # using its gradient
    return grad


T = TypeVar('T')  # this allows us to type "generic" functions


def minibatches(dataset: np.array, batch_size: int, shuffle: bool = True) -> Iterator[np.array]:
    """
    generates batch_size-sized minibatches from the dataset
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle:
        random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


if __name__ == '__main__':
    """
    minimizing f((x1, x2, x3)) = x1^2 + x2^2 + x3^2
    """
    # pick a random starting point
    v = np.array([random.uniform(-10, 10) for i in range(3)])

    for epoch in range(1000):
        grad = 2 * np.array([vi for vi in v])  # gradient of the function we minimize (sum of squares)
        v = gradient_step(v, grad, -0.01)      # we take a negative step

    assert np.linalg.norm(v - np.array([0, 0, 0]), 1) < 1e-6

    """
    use gradient to Fit Models
    """
    # we 'know' from what model the data come
    inputs = np.array([(x, 20 * x + 5) for x in range(-50, 50)])

    # Start with random values for slope and intercept
    theta = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    learning_rate = .001

    for epoch in range(5000):
        # compute the mean of the gradients
        grad = np.array([linear_gradient(x, y, theta) for x, y in inputs]).mean(axis=0)
        # take a step in that direction
        theta = gradient_step(theta, grad, -learning_rate)

    print('slope (close to 20): ', theta[0])
    print('intercept (close to 5): ', theta[1])

    """
    Minibatch Gradient
    """
    # we 'know' from what model the data come
    inputs = np.array([(x, 20 * x + 5) for x in range(-50, 50)])

    # Start with random values for slope and intercept
    theta = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = np.array([linear_gradient(x, y, theta) for x, y in batch]).mean(axis=0)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    print('slope (close to 20): ', theta[0])
    print('intercept (close to 5): ', theta[1])

    """
    Stochastic Gradient
    """

    # Start with random values for slope and intercept
    theta = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    for epoch in range(1000):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    print('slope (close to 20): ', theta[0])
    print('intercept (close to 5): ', theta[1])
