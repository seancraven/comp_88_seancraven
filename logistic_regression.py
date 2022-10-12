import numpy as np
import matplotlib.pyplot as plt
from week_2 import gradient_descent
from sklearn.datasets import load_iris
from week_1 import  generate_linearly_separable


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def one_hot_map(categorical_list):
    unique_ent = set(categorical_list)
    output = np.zeros((len(categorical_list), len(unique_ent)))
    ent_dict = {ent: i for i, ent in enumerate(unique_ent)}
    for i, cat in enumerate(categorical_list):
        output[i, ent_dict[cat]] = 1
    return output, ent_dict


def multinomial_logistic_regression(X, Y):
    def loss(W):
        assert W.shape[0] == X.shape[1]
        assert W.shape[1] == Y.shape[1]

        n = X.shape[0]

        assert n == Y.shape[0]

        Y_hat = softmax(X @ W)
        assert Y_hat.all() >= 0
        assert Y_hat.all() <= 1
        assert Y_hat.shape == Y.shape

        arg_mod = Y.T @ np.log(Y_hat)
        mod = np.sum(abs(arg_mod))
        return mod/n


    def grad_loss(W):
        Y_hat = softmax(X @ W)
        return X.T @ (Y_hat - Y)

    W0 = np.ones_like(X.T @ Y)
    test_grad_loss = grad_loss(W0)
    print(test_grad_loss)
    ww, ll = gradient_descent(W0, loss_func=loss, grad_func=grad_loss, lr=0.01)
    return ww, ll


if __name__ == "__main__":
    iris = load_iris()
    X, (y, map_dict) = np.array(iris.data), one_hot_map(iris.target)
    X_y = np.hstack((X, y))
    np.random.shuffle(X_y)
    X_train = X_y
    Y_train = X_y
    ww, ll = multinomial_logistic_regression(X_train, Y_train)
    Y_pred = softmax(X_train @ ww[-1])
    most_probable_class = np.argmax(Y_pred, axis=1)
    correct = 0
    for true, pred in zip(np.argmax(Y_train, axis=1), most_probable_class):
        if true == pred:
            correct += 1
    accuracy = correct/len(Y_train)
    print(f"Iris Accuracy {accuracy}")



