import numpy as np
from functools import reduce


def power_method(A, max_iter=1000, vec_tol=1e-6):
    """
    The power method for solving an eigenvalue problem. This method
    iteratively projects out the eigenstate with the largest (in
    magnitude) eigenvalue.

    Args:
    -----

        A : `np.ndarray`
            Matrix to diagonalize.

    Returns:
    --------

        w : float
            Dominant eigenvalue.

        v : `np.ndarray`
            Dominant eigenvector.

    """
    # Put in method class
    converged = False
    method_name = "Power Method"

    # Method specific
    v = np.random.rand(A.shape[1])
    v_old = v.copy()

    # Iterative search for dominant eigenvector
    for it in range(max_iter):
        v_i = np.dot(A, v)  # Calculate new eigenvector
        v = v_i / np.linalg.norm(v_i)  # Normalize it (we want unit vectors!)

        # Calculate distance relative to old vector
        delta1 = np.linalg.norm(v - v_old)
        delta2 = np.linalg.norm(v + v_old)
        delta = np.min([delta1, delta2])
        if delta < vec_tol:
            converged = True
            print(r"%s converged in %d iterations with \Delta = %.2e" %
                  (method_name, it, delta))
            break

        # Update
        v_old = v.copy()

    # Assert convergence
    if not converged:
        raise AssertionError("%s not converged!" % method_name)

    # Using Rayleigh quotient (i.e. expectation value of A operator on v)
    w = reduce(np.dot, (v, A, v))

    return w, v
