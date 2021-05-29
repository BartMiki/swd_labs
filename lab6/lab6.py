import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def fill_into(arr, shape):
    """Fills input `arr` into new shape (n, m). Pads with zeros or trims depending on the size."""
    target = np.zeros(shape, dtype=arr.dtype)
    i, j = list(map(min, zip(target.shape, arr.shape)))
    target[:i, :j] = arr[:i, :j]
    return target


def custom_svd(x):
    """Run custom implementation of SVD. Returns values similar in shape to Numpy counterpart."""
    cmm = x.T.dot(x)
    einval, einvec = np.linalg.eigh(cmm)
    ordering = np.argsort(einval)[::-1]

    einval, einvec = einval[ordering], einvec[:, ordering]
    v = einvec

    sigma = fill_into(np.diagflat(einval), x.shape)
    sigma_t = sigma.T
    sigma_inv = np.divide(1, sigma_t, where=~np.isclose(sigma_t, 0.0, rtol=1e-8))
    u = x.dot(v).dot(sigma_inv)

    return u, np.diag(sigma), v.T


METHODS = {
    'custom': custom_svd,
    'library': np.linalg.svd
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=Path, required=True, help="Path to an image")
    parser.add_argument('-out', type=Path, default=None, help="Optional path where to save output image")
    parser.add_argument('-svd', choices=METHODS.keys(), default='custom', help="SVD implementation")
    parser.add_argument('-k', type=int, default=None, help="Number of eigen values to use")

    return parser.parse_args()


def compress_image(image, k: int, compress_f: callable):
    """Run SVD compression on a image of shape (n,m) with given implementation."""
    u, sigma, v = compress_f(image)
    if k is None:
        k = len(sigma)

    new_image = u[:, :k].dot(np.diagflat(sigma)[:k, :k]).dot(v[:k, :])
    return np.clip(new_image, 0, 1)


def main():
    args = parse_args()
    image = plt.imread(args.f) / 255.0
    if image.ndim == 3:
        r, g, b = [compress_image(image[:, :, c], args.k, METHODS[args.svd]) for c in range(3)]
        new_image = np.stack([r, g, b], axis=2)
    else:
        # Support for grayscale images
        new_image = compress_image(image, args.k, METHODS[args.svd])

    if args.out is None:
        plt.imshow(new_image)
        plt.title("f='{}' | k={} | svd='{}'".format(args.f.name, args.k, args.svd))
        plt.axis('off')
        plt.show()
    else:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        plt.imsave(str(args.out), new_image)


if __name__ == '__main__':
    main()
