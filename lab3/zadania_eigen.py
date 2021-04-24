#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    eigvec = np.linalg.eig(A)[1]
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec.T)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    l, K = np.linalg.eig(A)
    L = np.diag(l)
    K_inv = np.linalg.inv(K)

    print("Macierz K\n", K)
    print("Macierz L\n", L)
    print("Macierz K_inv\n", K_inv)

    A_hat = K.dot(L).dot(K_inv)
    print("Macierz A\n", A)
    print("Odtworzona macierz A\n", A_hat)
    print("Są równe?", np.array_equal(A, A_hat.astype(int)))
    print(30 * "-")


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    eigen_value, eigen_vector = np.linalg.eig(A)
    eigen_vector = eigen_vector.T

    attractors = {
        'red': eigen_vector[0],
        'orange': eigen_vector[0] * -1,
    }
    if not np.allclose(eigen_vector[0], eigen_vector[1]):
        attractors = {
            **attractors,
            'green': eigen_vector[1],
            'turquoise': eigen_vector[1] * -1,
        }

    for color, attractor in attractors.items():
        plt.quiver(0.0, 0.0, attractor[0], attractor[1], width=0.01, color=color, scale_units='xy', angles='xy',
                   scale=1, zorder=4)
        num = 0 if color in ['red', 'orange'] else 1
        text = 'eigv{0}'.format(num) if color in ['red', 'green'] else ''
        plt.text(attractor[0] * 1.1, attractor[1] * 1.1, text, color=color, zorder=10)

    for vector in vectors:
        t_vec = vector.copy()
        vector = vector / np.linalg.norm(vector)
        for _ in range(500):
            t_vec = A.dot(t_vec)
            t_vec /= np.linalg.norm(t_vec)

        distances = [(np.mean(np.abs(t_vec - a)), color, a) for color, a in attractors.items()]
        distance, color, attractor = min(distances, key=itemgetter(0))
        if distance > 0.01 and not np.allclose(attractor, t_vec):
            # Don't converge
            color = 'black'

        # plt.quiver(0.0, 0.0, t_vec[0], t_vec[1], width=0.005, color=color, scale_units='xy', angles='xy',
        #            scale=1, zorder=6)
        plt.quiver(0.0, 0.0, vector[0], vector[1], width=0.005, color=color, scale_units='xy', angles='xy',
                   scale=1, zorder=6)

    plt.grid()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.margins(0.05)
    plt.show()


def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)

    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)

    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)

    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
