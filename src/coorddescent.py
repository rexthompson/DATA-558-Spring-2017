"""
This code consists of my own Python implementation of least-squares
regression with elastic net regularization.

Implementation by Rex Thompson for DATA 558
University of Washington, Spring 2017
RexS.Thompson@gmail.com
"""

import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def compute_obj(beta, X, y, lam, alpha):
    """
    Compute objective function value
    :param beta: coefficient values
    :param X: predictors (numpy array)
    :param y: response
    :param lam: regularization parameter
    :param alpha: elastic net coefficient
    :return: objective function value
    """
    obj = 1/len(y) * np.sum((y - X.dot(beta))**2) + \
          lam*(alpha*np.sum(np.abs(beta)) + (1 - alpha)*beta.dot(beta))
    return obj


def compute_beta_j_new(beta, X, y, lam, alpha, j, match_skl=False):
    """
    Perform partial minimization of a single component of beta
    :param beta: coefficient values
    :param X: predictors (numpy array)
    :param y: response
    :param lam: regularization parameter
    :param alpha: elastic net coefficient
    :param j: coordinate index to minimize
    :param match_skl: Boolean indicating whether to adjust output
    to match scikit-learn
    :return: updated coefficient
    """
    x_j = X[:, j]
    beta_dropJ = np.delete(beta, j)
    X_dropJ = np.delete(X, j, 1)
    n = len(y)

    if match_skl:
        num = 1
    else:
        num = 2

    a_j = (num/n)*(x_j.dot(x_j) + lam*n*(1-alpha))
    c_j = (num/n)*x_j.dot(y - X_dropJ.dot(beta_dropJ))

    if c_j < -lam*alpha:
        return (c_j + lam*alpha)/a_j
    elif c_j > lam*alpha:
        return (c_j - lam*alpha)/a_j
    else:
        return 0


def coorddescent(beta_init, X, y, lam, alpha, desc_type="cyclic", tol=0,
                 max_iter=1000, random_seed=None, match_skl=False):

    """
    Perform coordinate descent
    :param beta_init: coefficient values
    :param X: predictors (numpy array)
    :param y: response
    :param lam: regularization parameter
    :param alpha: elastic net coefficient
    :param desc_type: descent type: "cyclic" or "random"
    :param tol: convergence tolerance; default = 0
    :param max_iter: maximum number of iterations
    :param random_seed: random seed
    :param match_skl: Boolean indicating whether to adjust output
    to match scikit-learn
    :return: coordinates at each step of minimization
    """

    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    else:
        pass

    beta = copy.deepcopy(beta_init)
    step_coords = copy.deepcopy([beta])
    p = len(beta)

    # initialize variables for while loop
    dist = copy.deepcopy(tol)
    it = 0
    max_iter = max_iter*p

    while dist >= tol and it < max_iter:

        # pick a coordinate
        if desc_type == "cyclic":
            j = np.mod(it, p)
        elif desc_type == "random":
            j = np.random.randint(p)
        else:
            raise Exception("Invalid descent type: " + desc_type)

        # find new beta_j by minimizing F(beta)
        # w.r.t. beta_j; other coords stay the same
        beta[j] = compute_beta_j_new(beta, X, y, lam, alpha, j, match_skl)
        step_coords.append(np.array(beta))

        # compute distance from last attempt for convergence check
        if np.mod(it, p) == p-1:
            last_beta = step_coords[-p-1]
            dist = np.sqrt(sum((beta - last_beta)**2))
        it += 1

    return step_coords


def coorddescentCV(X, y, lambdas, alpha=.9, folds=3, desc_type="cyclic",
                   tol=0, max_iter=1000, random_seed=None, match_skl=False):

    """
    Perform coordinate descent cross-validation (CV)
    :param X: predictors (numpy array)
    :param y: response
    :param lambdas: regularization parameters on which to perform CV
    :param alpha: elastic net coefficient
    :param folds: number of folds
    :param desc_type: descent type: "cyclic" or "random"
    :param tol: convergence tolerance; default = 0
    :param max_iter: maximum number of iterations
    :param random_seed: random seed
    :param match_skl: Boolean indicating whether to adjust output
    to match scikit-learn
    :return: mean squared error for all folds and all lambdas
    """

    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    else:
        pass

    if isinstance(folds, int):
        pass
    else:
        raise ValueError()

    n, p = X.shape
    all_MSEs = np.empty((0, len(lambdas)))

    idxs = (list(range(folds)) * (n//folds+1))[0:n]
    np.random.shuffle(idxs)
    idxs = np.array(idxs)

    for i in range(folds):
        X_train_CV = X[idxs != i, :]
        X_test_CV = X[idxs == i, :]
        y_train_CV = y[idxs != i]
        y_test_CV = y[idxs == i]
        MSEs_mine = []
        for lam in lambdas:
            betas = coorddescent(beta_init=np.zeros(p), X=X_train_CV,
                                 y=y_train_CV, lam=lam, alpha=alpha,
                                 desc_type=desc_type, tol=tol,
                                 max_iter=max_iter, match_skl=match_skl)
            y_pred = np.dot(X_test_CV, betas[-1])
            MSEs_mine.append(mean_squared_error(y_test_CV, y_pred))
        all_MSEs = np.vstack((all_MSEs, MSEs_mine))

    return all_MSEs


def plot_MSEs(lambdas, all_MSEs):
    """
    Plot mean squared errors from coorddescentCV
    :param lambdas: lambda values used in CV
    :param all_MSEs: output from coorddescentCV
    :return: None
    """
    all_MSEs_mean = np.mean(all_MSEs, axis=0)
    for i in range(len(all_MSEs)):
        plt.plot(np.log(lambdas), all_MSEs[i], '--', linewidth=.8)
    plt.plot(np.log(lambdas), all_MSEs_mean, 'black', linewidth=2)
    plt.xlabel('log' + r'($\lambda)$')
    plt.ylabel('MSE')


def get_best_lambda(lambdas, all_MSEs):
    """
    Return best regularization parameter from coorddescentCV
    :param lambdas: lambda values used in CV
    :param all_MSEs: output from coorddescentCV
    :return: lambda value corresponding to lowest average MSE from CV
    """
    all_MSEs_mean = np.mean(all_MSEs, axis=0)
    max_min_index = np.max(np.where(all_MSEs_mean == all_MSEs_mean.min()))
    best_lam = lambdas[max_min_index]

    return best_lam


def plot_coefficient_steps(betas_cyclic, betas_random, cutoffs):
    """
    Create 2-4 sets of plots showing coefficients throughout the
    coordinate descent training process. Various cutoffs allow for
    zooming in on the more interesting periods of development.
    :param betas_cyclic: coefficients from cyclic coordinate descent
    :param betas_random: coefficients from random coordinate descent
    :param cutoffs: iteration cutoff for second and third plots
    :return: None
    """
    n_iter, p = np.array(betas_cyclic).shape
    cutoffs = np.append(-1, np.array(cutoffs)*p)
    row_count = len(cutoffs)
    fig = plt.figure(figsize=[14, 4*row_count])
    for i in range(2*row_count):

        if np.mod(i, 2) == 0:
            data = betas_cyclic
        else:
            data = betas_random

        cutoff = cutoffs[i//2]
        newIter = [j/p for j in range(n_iter)][0:cutoff]
        data = data[0:cutoff]

        plot_location = str(100*row_count + 20 + i+1)
        ax1 = fig.add_subplot(plot_location)
        ax1.plot(newIter, data)
        if i == 0:
            plt.title("Cyclic Coordinate Descent")
        elif i == 1:
            plt.title("Random Coordinate Descent")
        elif i >= 2*row_count - 2:
            plt.xlabel("Iteration Count (individual minimizations / p)")


def plot_obj_vals_steps(betas_cyclic, betas_random, X, y, lam, alpha, cutoff):
    """
    Plot objective values by iteration for all iterations and for a
    subset of all iterations.
    :param betas_cyclic: coefficients from cyclic coordinate descent
    :param betas_random: coefficients from random coordinate descent
    :param X: predictors (numpy array)
    :param y: response
    :param lam: regularization parameter
    :param alpha: elastic net coefficient
    :param cutoff: iteration cutoff for second plot
    :return: None
    """
    obj_vals_cyclic = [compute_obj(b, X, y, lam, alpha) for b in betas_cyclic]
    obj_vals_random = [compute_obj(b, X, y, lam, alpha) for b in betas_random]

    n_iter, p = np.array(betas_cyclic).shape
    newIter = [j/p for j in range(n_iter)]

    fig = plt.figure(figsize=[14, 4])
    ax1 = fig.add_subplot(121)
    ax1.plot(newIter, obj_vals_cyclic)
    ax1.plot(newIter, obj_vals_random)
    plt.xlabel('Iteration Counter')
    plt.ylabel('Objective Function Value')
    ax1.legend(['cyclic', 'random'])
    ax2 = fig.add_subplot(122)
    ax2.plot(newIter[0:p*cutoff], obj_vals_cyclic[0:p*cutoff])
    ax2.plot(newIter[0:p*cutoff], obj_vals_random[0:p*cutoff])
    plt.xlabel('Iteration Counter')
    plt.ylabel('Objective Function Value')
    ax2.legend(['cyclic', 'random'])
