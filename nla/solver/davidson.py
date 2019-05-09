import numpy as np
from time import time
from functools import reduce


def davidson_liu(A, n_roots=5):
    """
    Use the Davidson method to solve or multiple roots simultaneously.

    See Sherrill 1999 ADVANCES IN QUANTUM CHEMISTRY, VOLUME 34. for more
    details. Equations numbers are from fig 5.
    """
    # Put in method class
    converged = False
    method_name = "Davidson Method"

    # Method Specific
    L = n_roots
    N = A.shape[0]

    # Step 1
    b = np.zeros((L * 20, N))  # Guess vectors
    b[np.diag_indices(n_roots)] = 1
    delta = np.zeros((n_roots, N))

    lamb_old = None  # Old eigenvalues

    for iter in range(100):
        print("Iteration", iter)

        # Form Hamiltonian subspace
        # Step 2
        G = reduce(np.dot, (b[:L], A, b[:L].T))
        lamb, alpha = np.linalg.eig(G)
        lamb = lamb[:n_roots]
        alpha = alpha[:n_roots]

        # Step 3
        m = 0
        for k in range(n_roots):
            rk = reduce(np.dot, ((A - lamb[k] * np.eye(N)), b[:L].T, alpha[k]))
            dk = rk / (lamb[k] * np.ones(N) - A[np.diag_indices(N)])

            # Step 4
            dk /= np.linalg.norm(dk)

            # Step 5
            dk_copy = dk.copy()
            for i in range(L):
                dk -= np.dot(dk, b[i]) * b[i]
            if np.linalg.norm(dk) > 1e-3:
                b[L] = dk_copy
                m += 1
        L += m
        if iter > 0:
            delta_lamb = np.linalg.norm(lamb[:n_roots] - lamb_old[:n_roots])
            if delta_lamb < 1e-6:
                print("Breaking")
                return lamb
            else:
                print("\Delta \Lambda", delta_lamb)
        lamb_old = lamb


np.random.seed(20)
N = 3000
sparsity = 0.0001
A = np.zeros((N, N))
for i in range(0, N):
    A[i, i] = i + 1
A = A + sparsity * np.random.randn(N, N)
A = (A.T + A) / 2

t0 = time()
lamb = davidson_liu(A)
lamb.sort()
print(time() - t0)

t1 = time()
lamb_true, _ = np.linalg.eig(A)
lamb_true.sort()
print(time() - t1)

print(lamb[:5])
print(lamb_true[:5])
