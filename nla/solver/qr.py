import numpy as np


def gram_schmidt(A):
    """
    """

    # Gram-Schmidt Decomposition of A
    Q = np.zeros(A.shape)

    for i, _ in enumerate(A):
        Ai = A[:, i]
        Q[:, i] = Ai

        # Project out components along previous eigenvectors
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, j], Ai) * Q[:, j]  # Assumes Q[j] is norm

        # Normalize
        Q[:, i] /= np.linalg.norm(Q[:, i])

    # print(np.linalg.norm(np.dot(Q, Q.T) - np.eye(Q.shape[0])))

    # Calculate R
    R = np.dot(Q.T, A)

    return Q, R
