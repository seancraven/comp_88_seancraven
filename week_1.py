#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COMP0088 lab exercises for week 1.

This first introductory set of exercises is largely intended
as a warm up and practice session. It is an opportunity to check
that you have a functioning Python 3 system with the requisite libraries, to get
a feel for some basic data manipulation and plotting, and to ensure that
everything makes sense and runs smoothly.

Add your code as specified below. You shouldn't need to load further external
code that isn't already explicitly imported.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_1.py

A 4-panel figure, `week_1.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys, os, os.path
import argparse

import numpy as np
import numpy.random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

def generate_noisy_linear(num_samples, weights, sigma, limits, rng):
    """
    Draw samples from a linear model with additive Gaussian noise.
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples output values
    """
    # Rather than having the reduced dimensions as suggested in the question,
    # Have the same number of features as weights so an elementwise product
    # gives the required solution
    epsilon = np.random.normal(loc=0, scale=sigma, size=num_samples)
    x = np.random.uniform(*limits, size=(num_samples, len(weights)))
    x[:, 0] = 1
    y = np.sum(x * weights, axis=1) + epsilon
    return x[:, 1:], y


def plot_noisy_linear_1d(axes, num_samples, weights, sigma, limits, rng):
    """
    Generate and plot points from a noisy single-feature linear model,
    along with a line showing the true (noiseless) relationship.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        None
    """
    assert(len(weights)==2)
    X, y = generate_noisy_linear(num_samples, weights, sigma, limits, rng)
    axes.scatter(X, y, marker = "x")






def plot_noisy_linear_2d(axes, resolution, weights, sigma, limits, rng):
    """
    Produce a plot illustrating a noisy two-feature linear model.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        resolution: how densely should the model be sampled?
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        None
    """
    assert(len(weights)==3)
    samples = 5000
    X, y = generate_noisy_linear(samples, weights, sigma, limits, rng)
    axes.scatter(X[:,0], X[:,1], c = y, cmap="viridis")





# -- Question 2 --

def generate_linearly_separable(num_samples, weights, limits, rng):
    """
    Draw samples from a binary model with a given linear
    decision boundary.

    # Arguments
        num_samples: number of hts)==2)
    X, y = generate_noisy_linear(num_samples, weights, sigma, limits, rng)
    numsamples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples binary labels
    """
    X = np.random.uniform(*limits, (num_samples, len(weights)))
    X[:,0] = 1
    f_X = np.sum(X*weights, axis=1)
    y = np.zeros(num_samples)
    y[f_X >= 0] = 1
    return X[:, 1:], y



def plot_linearly_separable_2d(axes, num_samples, weights, limits, rng):
    """
    Plot a linearly separable binary data set in a 2d feature space.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        None
    """
    assert(len(weights)==3)
    X, y = generate_linearly_separable(num_samples, weights, limits, rng)
    y_bool = y.astype(bool)
    axes.scatter(X[y_bool, 0], X[y_bool, 1], marker="x", color="black")
    axes.scatter(X[~y_bool, 0], X[~y_bool, 1] ,marker="o", color="red")
    # Plotting the intersection of the division plane denoted by weights
    x_0 = np.linspace(*limits, num=20)
    x_1_on_line = (x_0*weights[1] + weights[0])/-weights[2]
    axes.plot(x_0, x_1_on_line, linestyle="--", color="grey")
    axes.set_ylim(*limits)
    axes.set_xlim(*limits)
    arrow_grad = weights[2]/weights[1]
    axes.arrow(0,-weights[0]/weights[2],-1, -1*arrow_grad, color="grey", head_width=1)
# -- Question 3 --

def random_search(function, count, num_samples, limits, rng):
    """
    Randomly sample from a function of `count` features and return
    the best feature vector found.

    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_samples: the number of samples to generate & search
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        x: a vector of length count, containing the found features
    """

    random_feature_grid = utils.make_random(num_samples,limits=limits, rng=rng,count=count)
    output = np.zeros(num_samples)
    for i in range(num_samples):
        output[i] = function(random_feature_grid[i, :])
    return random_feature_grid[np.argmin(output), :]

def grid_search(function, count, num_divisions, limits):
    """
    Perform a grid search for a function of `count` features and
    return the best feature vector found.
    
    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_divisions: the number of samples along each feature
            dimension (including endpoints)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
    
    # Returns
        x: a vector of length count, containing the found features
    """

    grid = utils.make_grid(limits=limits, num_divisions=num_divisions,count=count)
    output = function(grid)
    first_index = np.argmin(output) // num_divisions
    second_index = np.argmin(output) % num_divisions
    return grid[first_index, second_index, :]

def plot_searches_2d(axes, function, limits, resolution,
                     num_divisions, num_samples, rng, true_min=None):
    """
    Plot a 2D function aling with minimum values found by
    grid and random searching.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        function: a function taking a single input array of
            shape (..., 2), where the last dimension
            indexes the features
        limits: a tuple (low, high) specifying the value
            range of both input features x1 and x2
        resolution: number of samples along each side
            (including endpoints) for an image representation
            of the function
        num_divisions: the number of samples along each side
            (including endpoints) for a grid search for
            the function minimum
        num_samples: number of samples to draw for a random
            search for the function minimum
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
        true_min: an optional (x1, x2) tuple specifying
            the location of the actual function minimum
            
    # Returns
        None
    """
    grid = utils.make_grid(limits, num_divisions, count=2)
    f_grid = function(grid)
    grid_search_val = grid_search(function, 2, num_divisions,limits)
    random_search_val = random_search(function, 2, num_samples, limits, rng)
    axes.contourf(grid[...,0], grid[...,1], f_grid, cmap="viridis")
    axes.scatter(*true_min, c="red", label="True Min")
    axes.scatter(*grid_search_val, c="blue", label="Grid Search")
    axes.scatter(*random_search_val, c="green", label="Random Search")
    axes.legend()

def plots_3d(axes, num_samples, weights, sigma, limits, rng):
    X, y = generate_noisy_linear(num_samples, weights, sigma, limits, rng)

    axes.scatter(X[:, 0], X[:, 1], y, c="black")
#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 1 labwork script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_1.pdf')
    return ap.parse_args()


def test_func(X):
    """
    Simple example function of 2 variables for
    testing grid & random optimisation.
    """
    return (X[..., 0]-1)**2 + X[...,1]**2 + 2 * np.abs((X[...,0]-1) * X[...,1])

WEIGHTS = np.array([0.5, -0.4, 0.6])
LIMITS = (-5, 5)

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)

    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(nrows=2, ncols=2)
    
    print('Q1: noisy continuous data')
    print('plotting 1D data')
    plot_noisy_linear_1d(axs[0, 0], 50, WEIGHTS[1:], 0.5, LIMITS, rng)
    print('plotting 2D data')
    plot_noisy_linear_2d(axs[0, 1], 100, WEIGHTS, 0.2, LIMITS, rng)
    
    print('\nQ2: binary separable data')
    print('plotting 2D labelled data')
    plot_linearly_separable_2d(axs[1, 0], num_samples=100, weights=WEIGHTS, limits=LIMITS, rng=rng)
    
    print('\nQ3: searching for a minimiser')
    print('plotting searches')
    plot_searches_2d(axs[1, 1], test_func, limits=LIMITS, resolution=100, num_divisions=10, num_samples=100, rng=rng, true_min=(1,0))



    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)

    print("\nQ4: 3D Plotting")
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111, projection="3d")
    ax.view_init(30,20)
    plots_3d(ax, 1000, WEIGHTS, 1, LIMITS, rng)
    fig_2.tight_layout(pad=1)
    fig_2.savefig("week_1_3D.pdf")
    plt.close(fig_2)