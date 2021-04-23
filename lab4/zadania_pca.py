#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_sklearn(data, n_comp=None):
    """Reduces dimensionality using decomposition.PCA class from scikit-learn library.

    :param data: (np.array) Input data. Rows: observations, columns: features.
    :param n_comp: (int) Number of dimensions (components). By default equal to the number of features (variables).
    :return: (np.array) The transformed data after changing the coordinate system, possibly with removed dimensions with
    the smallest variance.
    """
    assert isinstance(data, np.ndarray)
    assert n_comp <= data.shape[1], "The number of components cannot be greater than the number of features"
    if n_comp is None:
        n_comp = data.shape[1]
    # TODO: Implement PCA using scikit-learn library, class: sklearn.decomposition.PCA
    # TODO: Return the transformed data.
    return PCA(n_components=n_comp).fit_transform(data)


def pca_manual(data, n_comp=None):
    """Reduces dimensionality by using your own PCA implementation.

    :param data: (np.array) Input data. Rows: observations, columns: features.
    :param n_comp: (int) Number of dimensions (components). By default equal to the number of features (variables).
    :return: (np.array) The transformed data after changing the coordinate system, possibly reduced.
    """
    assert isinstance(data, np.ndarray)
    assert n_comp <= data.shape[1], "The number of components cannot be greater than the number of features"
    if n_comp is None:
        n_comp = data.shape[1]

    # TODO: 1) Adjust the data so that the mean of every column is equal to 0.
    data -= data.mean(axis=0)

    # TODO: 2) Compute the covariance matrix. You can use the function from numpy (numpy.cov), or multiply appropriate matrices.
    # Warning: numpy.cov expects dimensions to be in rows and different observations in columns.
    #          You can transpose data or set rowvar=False flag.
    cov_matrix = np.cov(data.T)

    print("\nCOVARIANCE MATRIX:")
    print(cov_matrix)

    # TODO: 3) Calculate the eigenvectors and eigenvalues of the covariance matrix.
    # You may use np.linalg.eig, which returns a tuple (eigval, eigvec).
    # Make sure that eigenvectors are unit vectors (PCA needs unit vectors).
    eigval, eigvec = np.linalg.eig(cov_matrix)
    eigvec = eigvec.T

    # TODO: 4) Sort eigenvalues (and their corresponding eigenvectors) in the descending order (e.g. by using argsort),
    #          and construct the matrix K with eigenvectors in the columns.
    order = np.argsort(eigval)[::-1]
    sorted_vectors = eigvec[order]

    print("\nSORTED EIGEN VALUES:")
    print(eigval[order])
    print("\nSORTED EIGEN VECTORS:")
    print(sorted_vectors)

    # TODO: 5) Select the components (n_comp).
    selected_vectors = sorted_vectors[:n_comp]

    # TODO: 6) Calculate the transformed data.
    transformed = np.dot(selected_vectors, data.T).T

    # TODO: 7) Calculate the covariance matrix of the transformed data.
    cov_transform_matrix = np.cov(transformed.T)

    print("\nCOVARIANCE MATRIX OF THE TRANSFORMED DATA:")
    print(cov_transform_matrix)

    # TODO: 8) Return the transformed data.
    return transformed


def plot_pca_result_2d(X, Y1, Y2):
    assert Y1.shape == Y2.shape
    if Y1.shape[1] > 2:
        return None
    elif Y1.shape[1] == 1:
        zeros = np.zeros((Y1.shape[0], 1))
        Y1 = np.hstack((Y1, zeros))
        Y2 = np.hstack((Y2, zeros))

    min_x = min(min(X[:, 0]), min(Y1[:, 0]), min(Y2[:, 0]))
    max_x = max(max(X[:, 0]), max(Y1[:, 0]), max(Y2[:, 0]))
    min_y = min(min(X[:, 1]), min(Y1[:, 1]), min(Y2[:, 1]))
    max_y = max(max(X[:, 1]), max(Y1[:, 1]), max(Y2[:, 1]))
    fig = plt.figure(figsize=(10, 5))
    ax0 = plt.subplot(1, 3, 1)
    ax0.set_xlim([min_x, max_x])
    ax0.set_ylim([min_y, max_y])
    ax0.set_title("Original data")
    ax0.scatter(X[:, 0], X[:, 1])

    ax1 = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
    ax1.set_title("Transformed data (PCA custom)")
    ax1.scatter(Y1[:, 0], Y1[:, 1])

    ax2 = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
    plt.title("Transformed data (PCA scikit)")
    plt.scatter(Y2[:, 0], Y2[:, 1])
    plt.show()


def run_pca_comparison(X, n_comp=1):
    Y1 = pca_manual(X, n_comp=n_comp)
    Y2 = pca_sklearn(X, n_comp=n_comp)
    print("\nDifference: {0}".format(abs((Y1 - Y2)).sum()))
    plot_pca_result_2d(X, Y1, Y2)


def rotation_matrix(alpha):
    return np.array([[math.cos(alpha), -math.sin(alpha)],
                     [math.sin(alpha), math.cos(alpha)]])


def data_random(n_points, x_mean, x_std, y_mean, y_std, angle=0.0):
    x = np.random.normal(x_mean, x_std, n_points)
    y = np.random.normal(y_mean, y_std, n_points)
    data = np.vstack((x, y)).T
    return np.dot(data, rotation_matrix(angle))


def data_example1():
    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, 'data', 'example1.csv')
    return np.genfromtxt(path, delimiter=',')


if __name__ == "__main__":
    data = data_example1()
    run_pca_comparison(data, n_comp=2)

    data = data_random(100, x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=4.0, angle=-math.pi / 4)
    run_pca_comparison(data, n_comp=2)
