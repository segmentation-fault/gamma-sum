__author__ = 'antonio franco'

'''
Copyright (C) 2019  Antonio Franco (antonio_franco@live.it)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
from scipy.special import loggamma


def generate_gamma_sum_MC(rates, shapez, n_samples):
    """
    Generates n_samples from an rv representing the sum of len(rates) gamma variates, with rate rates(i) and shape
    shapez(i)
    :param rates (list of floats): rate parameters
    :param shapez (list of floats): shape parameters. Must be have the same length as len(rates)
    :param n_samples (int): number of samples to generate
    :return (list of floats): samples
    """
    assert (isinstance(rates, list))
    assert (isinstance(shapez, list))
    assert (len(rates) == len(shapez))

    S = np.zeros(n_samples)

    for l, a in zip(rates, shapez):
        S += np.random.gamma(a, 1.0 / l, n_samples)

    return S.tolist()


def eval_integrand(a, b, c, d, z, s):
    """
    Evaluates the integrand of the Fox's \overbar{H} function in (z,s)
    :param a (list of tuples): first vector of length n. Each element is a tuple (\alpha_j, A_j, a_j)
    :param b (list of tuples): second vector of length p - n. Each element is a tuple (\alpha_j, A_j)
    :param c (list of tuples): third vector of length m. Each element is a tuple (\beta_j, B_j)
    :param d (list of tuples): fourth vector of length q - m. Each element is a tuple (\beta_j, B_j, b_j)
    :param z (float): value of the \overbar{H} function to evaluate
    :param s (complex): value where to evaluate the integrand
    :return (float): the value of the integrand in (z,s)
    """
    assert (all(len(x) == 3 for x in a))
    assert (all(len(x) == 2 for x in b))
    assert (all(len(x) == 3 for x in d))
    assert (all(len(x) == 2 for x in c))
    assert (z != 0)

    n = len(a)
    m = len(c)
    p = n + len(b)
    q = m + len(d)

    nom1_exp = 0.0
    for i in range(0, n):
        v = a[i]
        nom1_exp += v[2] * loggamma(1.0 - v[0] + v[1] * s)

    nom2_exp = 0.0
    for i in range(0, m):
        v = c[i]
        nom2_exp += loggamma(v[0] - v[1] * s)

    nom_exp = nom1_exp + nom2_exp

    den1_exp = 0.0
    for i in range(0, p - n):
        v = b[i]
        den1_exp += loggamma(v[0] - v[1] * s)

    den2_exp = 0.0
    for i in range(0, q - m):
        v = d[i]
        den2_exp += v[2] * loggamma(1.0 - v[0] + v[1] * s)

    den_exp = den1_exp + den2_exp

    M = exp(nom_exp - den_exp) * np.power(z + 0j, s)

    return M


from mpmath import *


def FoxHBar(a, b, c, d, z):
    """
    Evaluates the Fox's \overbar{H} function in z
    :param a (list of tuples): first vector of length n. Each element is a tuple (\alpha_j, A_j, a_j)
    :param b (list of tuples): second vector of length p - n. Each element is a tuple (\alpha_j, A_j)
    :param c (list of tuples): third vector of length m. Each element is a tuple (\beta_j, B_j)
    :param d (list of tuples): fourth vector of length q - m. Each element is a tuple (\beta_j, B_j, b_j)
    :param z (float): value of the \overbar{H} function to evaluate
    :return (mpc): the value of the function in z
    """
    assert (all(len(x) == 3 for x in a))
    assert (all(len(x) == 2 for x in b))
    assert (all(len(x) == 3 for x in d))
    assert (all(len(x) == 2 for x in c))
    assert (z != 0)

    M = 1e6  # Sufficiently big interval
    # The only singularity on the Re(s) = 0 axis could be in s = 0, so we split the integral into (-1j*inf, 0)
    # and (0, 1j*inf)
    H = 1.0 / (2.0 * np.pi * 1j) * chop(quad(lambda s: eval_integrand(a, b, c, d, z, s), [-M * 1j, 0, M * 1j],
                                             maxdegree=6))

    return H


def gamma_sum_PDF(rates, shapez, y):
    """
    Evaluates the PDF of a sum of len(rates) gamma variates, with rate rates(i) and shape shapez(i) in y according to:
    "New Results on the Sum of Gamma Random Variates With Application to the Performance of Wireless Communication Systems
    over Nakagami-m Fading Channels", https://arxiv.org/abs/1202.2576
    :param rates (list of floats): rate parameters
    :param shapez (list of floats): shape parameters. Must be have the same length as len(rates)
    :param y (float): value where to evaluate the PDF
    :return (mpc): value of the PDF in y
    """
    assert (isinstance(rates, list))
    assert (isinstance(shapez, list))
    assert (len(rates) == len(shapez))

    z = exp(y)

    psi1 = []
    psi2 = []

    K = 1.0

    for l, a in zip(rates, shapez):
        psi1.append([1.0 - l, 1.0, a])
        psi2.append([-l, 1.0, a])
        K *= l ** a

    f = K * FoxHBar(psi1, [], [], psi2, z)

    return f


def gamma_sum_CDF(rates, shapez, y):
    """
    Evaluates the CDF of a sum of len(rates) gamma variates, with rate rates(i) and shape shapez(i) in y according to:
    "New Results on the Sum of Gamma Random Variates With Application to the Performance of Wireless Communication Systems
    over Nakagami-m Fading Channels", https://arxiv.org/abs/1202.2576
    :param rates (list of floats): rate parameters
    :param shapez (list of floats): shape parameters. Must be have the same length as len(rates)
    :param y (float): value where to evaluate the PDF
    :return (mpc): value of the CDF in y
    """
    assert (isinstance(rates, list))
    assert (isinstance(shapez, list))
    assert (len(rates) == len(shapez))

    z = exp(y)

    psi1 = []
    psi2 = []

    K = 1.0

    for l, a in zip(rates, shapez):
        psi1.append([1.0 - l, 1.0, a])
        psi2.append([-l, 1.0, a])
        K *= l ** a

    psi1.append([1, 1, 1])
    psi2.append([0, 1, 1])

    f = 0.5 + K * FoxHBar(psi1, [], [], psi2, z)

    return f


import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Tests the algorithm against a Montecarlo simulation
    """
    np.random.seed(19680801)  # For reproducibility

    # Main parameters
    n_samples = int(1e4)
    n_gammas = 5
    n_bins = 30

    rates = np.random.rand(n_gammas).tolist()
    shapez = np.random.rand(n_gammas) * 5.0
    shapez = shapez.tolist()

    S = generate_gamma_sum_MC(rates, shapez, n_samples)

    plt.figure()
    n, bins, patches = plt.hist(S, n_bins, density=True, facecolor='b', alpha=0.75, label='Montecarlo')
    Y = np.linspace(bins.min(), bins.max(), n_bins * 2)
    F = []
    for y in Y:
        f = gamma_sum_PDF(rates, shapez, y)
        F.append(float(f.real))
    plt.plot(Y, F, label='Analytical')
    plt.xlabel('Y')
    plt.ylabel('PDF')
    plt.title('Sum of ' + str(len(rates)) + ' Gamma random variables')
    plt.legend()

    plt.figure()
    n, bins, patches = plt.hist(S, n_bins, density=True, facecolor='b', alpha=0.75, label='Montecarlo', cumulative=True)
    Y = np.linspace(bins.min(), bins.max(), n_bins * 2)
    F = []
    for y in Y:
        f = gamma_sum_CDF(rates, shapez, y)
        F.append(float(f.real))
    plt.plot(Y, F, label='Analytical')
    plt.xlabel('Y')
    plt.ylabel('CDF')
    plt.title('Sum of ' + str(len(rates)) + ' Gamma random variables')
    plt.legend()

    plt.show()
