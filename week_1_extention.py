import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils

matplotlib.use("TkAgg")


def boundary(weights, function, limits=(-5, 5), num_divisions=10):
    """
    Creates a 2D surface

    Args:
        function: The function of x,y for which you would like a surface
            plotted.
        weights: Weights for the boundary should have len(3). Where the
            boundary equation is w.f(X) = 0
        limits: The symmetric limits for the grid of values that the
        function is evaluated on.
        num_divisions: Number of the divisions in the grid along one axis


    Returns:
        X: The points on the grid for which the function is evaluated.
             shape(num_divisions, num_divisions, 2)
        y: The values of the function at each point on the grid.
            shape(num_divisions, num_divisions)
    """
    assert len(weights) == 3
    grid = utils.make_grid(limits, num_divisions, count=2)
    f_eval = function(grid[..., :2])
    output = np.ones_like(grid)[..., :2]
    output[..., 0] = f_eval
    return grid[..., :2], np.sum(weights[:2] * output, axis=-1)


def plot_classes(weights, function,  axes, num_samples= 1000, limits=(-5,5)):
    """
    Utility for taking scatterd points in a 2d space and classifying them.

    Args:
        X: 2D values
        y:
        axes:
    """
    X = utils.make_random(num_samples, limits, count=2)
    y = function(X)*weights[0] + weights[1]
    X_minus = np.array([X])[y < 0]
    X_plus = np.array([X])[y >= 0]
    axes.scatter(X_minus[..., 0], X_minus[..., 1],  c="black")
    axes.scatter(X_plus[..., 0], X_plus[..., 1],  c="red")


def plot_surface(X, y, axes):
    assert X.shape[0] == X.shape[1]
    assert X.shape[:2] == y.shape
    axes.plot_surface(X[..., 0], X[..., 1], y)


def function(X):
    # Using a non-linear transform we can form boundary surfaces for, many
    # functions. The example is that of a parabola in both directions.
    return np.sum(np.dstack((X[..., 0] ** 2, np.exp(X[..., 1]))), axis=-1) - 3


if __name__ == "__main__":
    WEIGHTS = np.random.random(3)
    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    b_X, b_y = boundary(WEIGHTS, function)
    plot_surface(b_X, b_y, ax)
    ax_2 = fig.add_subplot(212)
    plot_classes(WEIGHTS, function, ax_2)
    plt.show()