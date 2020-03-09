import numpy as np


from numpy.lib.stride_tricks import as_strided


def main():

    window_size = 4

    # Get data
    A = np.loadtxt('chall1.txt')

    # Get view shape
    window_shape = (window_size, window_size)
    view_shape = tuple(np.subtract(A.shape, window_shape) + 1) + window_shape

    # Get view and resize
    view = as_strided(A, view_shape, A.strides * 2).reshape(
        (view_shape[0] * view_shape[1], view_shape[2], view_shape[3]))

    sums = []

    # For all overlapping sub-matrices of size 4x4
    for a in view:
        # Get the products of all rows
        sums.extend([np.prod(a[i, :]) for i in range(window_size)])
        # Get the products of all columns
        sums.extend([np.prod(a.T[i, :]) for i in range(window_size)])
        # Get the product of the diagonal
        sums.append(np.prod(a.diagonal()))

    print(max(sums))


if __name__ == "__main__":
    main()
