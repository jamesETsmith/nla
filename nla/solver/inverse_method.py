import numpy as np
from functools import reduce


def inverse_method(A, w_guess, max_iter=1000, vec_tol=1e-6):
    """
    The inverse method for solving an eigenvalue problem.

    Args:
    -----

        A : `np.ndarray`
            Matrix to diagonalize.

        w_guess : float
            Initial guess for eigenvalue.

    Returns:
    --------

        w : float
            Eigenvalue closest to guess.

        v : `np.ndarray`
            Eigenvector corresponding to the eigenvalue closest to the guess.

    """

    # Put in method class
    converged = False
    method_name = "Inverse Method"

    # Method specific
    v = np.random.rand(A.shape[0])
    v_old = v.copy()
    A_inv = np.linalg.inv(A - w_guess * np.eye(A.shape[0]))

    # Iterative search for dominant eigenvector
    for it in range(max_iter):
        v_i = np.dot(A_inv, v)  # Calculate new eigenvector
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
    w = 1 / reduce(np.dot, (v, A_inv, v)) + w_guess

    return w, v
