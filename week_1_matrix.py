import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils
matplotlib.use("TkAgg")

def matrix_class(limits=(-5,5), weights=np.random.random((3,3)), num_samples=1000):
    X = np.ones((num_samples, 3))
    X[:,:2] = utils.make_random(num_samples,limits, count=2)
    y = np.zeros(num_samples)
    for i,x in enumerate(X):
        #find solution to doing these mat prods.
        y[i] = np.einsum("i,ij,j",x,weights,x)
    return X[:, :2], y



def plot_classes_3D(X, y, axes):
    """
    Utility for taking scatterd points in a 2d space and classifying them.

    Args:
        X: 2D values
        y:
        axes:
    """

    X_minus = X[y < 0]
    X_plus = X[y >= 0]
    axes.scatter3D(X_minus[..., 0], X_minus[..., 1],y[y<0] , c="black")
    axes.scatter3D(X_plus[..., 0], X_plus[..., 1], y[y>=0], c="red")
if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for _ in range(10):
        X, y = matrix_class()
        plot_classes_3D(X, y, ax)
        plt.show()
