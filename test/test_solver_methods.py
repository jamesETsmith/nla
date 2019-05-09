import numpy as np
import numpy.testing as npt
from nla.solver.power_method import power_method
from nla.solver.inverse_method import inverse_method
from nla.solver.qr import gram_schmidt

decimals = 7
np.random.seed(20)
N = 100


def eigenvector_test(v_true, v_approx):
    """
    Utility function to compare eigenvectors.
    """

    # if the norm is big try switching the sign, sometimes the random guess
    # vector will have the opposite sign as the numpy eigenvector
    # print(np.linalg.norm(v_true - v_approx))
    # print(np.linalg.norm(v_true + v_approx))
    if np.linalg.norm(v_true - v_approx) < 10**(-decimals):
        npt.assert_almost_equal(v_true, v_approx, decimal=decimals)
    else:
        npt.assert_almost_equal(v_true, -v_approx, decimal=decimals)


#
# Unit tests
#


def test_power_method():
    """
    """
    A_unsym = np.random.random((N, N))
    A = np.dot(A_unsym, A_unsym.T)

    # Reference
    w, v = np.linalg.eig(A)

    # Approximate method
    w_approx, v_approx = power_method(A)

    # Test
    eigenvector_test(v[:, 0], v_approx)
    npt.assert_almost_equal(w_approx, w[0], decimal=decimals)


def test_inverse_method():
    """
    """
    A_unsym = np.random.random((N, N))
    A = np.dot(A_unsym, A_unsym.T)

    # Reference
    w, v = np.linalg.eig(A)

    # Approximate method
    w_approx, v_approx = inverse_method(A, w[0] + 0.001)

    # Test
    eigenvector_test(v[:, 0], v_approx)
    npt.assert_almost_equal(w_approx, w[0], decimal=decimals)


def test_gram_schmidt():
    """
    """
    A_unsym = np.random.random((N, N))
    A = np.dot(A_unsym, A_unsym.T)
    # A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])

    # Reference
    Q, R = np.linalg.qr(A)
    # print(R)
    # print(Q)

    # Approximate method
    Q_approx, R_approx = gram_schmidt(A)

    # Test Qs
    print(np.linalg.norm(Q - Q_approx))
    print(np.linalg.norm(Q + Q_approx))
    print(np.linalg.norm(R - R_approx))
    print(np.linalg.norm(R + R_approx))

    if np.linalg.norm(Q - Q_approx) < 10**-decimals:
        npt.assert_almost_equal(Q, Q_approx, decimal=10**-decimals)
    else:
        npt.assert_almost_equal(Q, -Q_approx, decimal=10**-decimals)

    # Test Rs
    if np.linalg.norm(R - R_approx) < 10**-decimals:
        npt.assert_almost_equal(R, R_approx, decimal=10**-decimals)
    else:
        npt.assert_almost_equal(R, -R_approx, decimal=10**-decimals)
