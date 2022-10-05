import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils

matplotlib.use("TkAgg")


def boundary(weights, limits=(-5, 5), num_divisions=100):
    assert len(weights) == 3
    grid = utils.make_grid(limits, num_divisions, count=2)
    f_eval = function(grid[...,:2])
    output = np.ones_like(grid)[...,:2]
    output[..., 0] = f_eval
    return grid[..., :2], np.sum(weights[:2]*output, axis=-1)


def plot_boundary(X, y, axes):
    X_minus, y_minus = X[y < 0], y[y < 0]
    X_plus, y_plus = X[y >= 0], y[y >= 0]
    axes.scatter3D(X_minus[..., 0], X_minus[..., 1], y_minus, c="black")
    axes.scatter3D(X_plus[..., 0], X_plus[..., 1], y_plus, c="red")

def plot_surface(X, y, axes):
    assert(X.shape[0] == X.shape[1])
    assert(X.shape[:2] == y.shape)
    axes.plot_surface(X[..., 0], X[..., 1], y)

def function(X):
    # Using a non-linear transform we can form boundary surfaces for, many
    # functions. The example is that of a parabola in both directions.
    return np.sum(np.dstack((X[...,0]**2,(X[...,1]**2))), axis=-1)


if __name__ == "__main__":
    WEIGHTS = np.random.random(3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    b_X, b_y = boundary(WEIGHTS)
    plot_surface(b_X,b_y, ax)
    plt.show()
