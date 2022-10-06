import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils
matplotlib.use("TkAgg")

def matrix_class(limits=(-5,5), weights=np.random.random((3,3)), num_samples=1000):
    X = np.ones((num_samples, 3))
    X[:,:2] = utils.make_random(num_samples,limits, count=2)
    y =[]
    for x in X.T:
        #find solution to doing these mat prods.
        intermediate_prod = np.matmul(weights,x.T)
        y.append(np.matmul(x,intermediate_prod))
if __name__ == "__main__":
    matrix_class()\