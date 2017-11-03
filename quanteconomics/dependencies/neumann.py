"""
Filename: neumann.py
Generalized von Neumann growth model

Author: Balint Szoke
Date: 07/17/2015
Last update: 10/07/2016

"""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import fsolve, linprog

from textwrap import dedent


class neumann(object):
    """
    This class describes the Generalized von Neumann growth model as it was
    discussed in Kemeny et al. (1956, ECTA) and Gale (1960, Chapter 9.5):

    Let:
    n ... number of goods
    m ... number of activities
    A ... input matrix is m-by-n
        a_{i,j} - amount of good j consumed by activity i
    B ... output matrix is m-by-n
        b_{i,j} - amount of good j produced by activity i

    x ... intensity vector (m-vector) with nonnegative entries
        x'B - the vector of goods produced
        x'A - the vector of goods consumed
    p ... price vector (n-vector) with nonnegative entries
        Bp - the revenue vector for every activity
        Ap - the cost of each activity

    Both A and B have nonnegative entries. Moreover, we assume that
    (1) Assumption I (every good which is consumed is also produced):
        for all j, b_{.,j} > 0, i.e. at least one entry is strictly positive
    (2) Assumption II (no free lunch):
        for all i, a_{i,.} > 0, i.e. at least one entry is strictly positive

    Parameters
    ----------
    A : array_like or scalar(float)
        Part of the state transition equation.  It should be `n x n`
    B : array_like or scalar(float)
        Part of the state transition equation.  It should be `n x k`

    Attributes
    ----------
    A, B: see Parameters
    n, m : scalar(int)
        number of goods and activities, respectively
    """

    def __init__(self, A, B):

        self.A, self.B = list(map(self.convert, (A, B)))
        self.m, self.n = self.A.shape

        # Check if (A,B) satisfy the basic assumptions
        assert self.A.shape == self.B.shape, 'The input and output matrices must have the same dimensions!'
        assert (self.A >= 0).all() and (self.B >= 0).all(), 'The input and output matrices must have only nonnegative entries!'

        # (1) Check whether Assumption I is satisfied:
        if (np.sum(B, 0) <= 0).any():
            self.AI = False
        else:
            self.AI = True

        # (2) Check whether Assumption II is satisfied:
        if (np.sum(A, 1) <= 0).any():
            self.AII = False
        else:
            self.AII = True

        # Check irreducibility:
        #self.irreducible = True


    def __repr__(self):
        return self.__str__()


    def __str__(self):

        me = """
        Generalized von Neumann expanding model:
          - number of goods          : {n}
          - number of activities     : {m}

        Assumptions:
          - AI:  every column of B has a positive entry    : {AI}
          - AII: every row of A has a positive entry       : {AII}

        """
        #Irreducible                                        : {irr}
        return dedent(me.format(n = self.n, m = self.m,
                                AI = self.AI, AII = self.AII))
                                #irr = self.irreducible))


    def convert(self, x):
        """
        Convert array_like objects (lists of lists, floats, etc.) into
        well formed 2D NumPy arrays
        """
        return np.atleast_2d(np.asarray(x))


    def bounds(self):
        """
        Calculate the trivial upper and lower bounds for alpha (expansion rate) and
        beta (interest factor). See the proof of Theorem 9.8 in Gale (1960).

        Outputs:
        --------
        LB: scalar
            lower bound for alpha, beta
        UB: scalar
            upper bound for alpha, beta
        """

        n, m = self.n, self.m
        A, B = self.A, self.B

        f = lambda alpha: ((B - alpha*A) @ np.ones((n, 1))).max()
        g = lambda beta: (np.ones((1, m)) @ (B - beta*A)).min()

        UB = np.asscalar(fsolve(f, 1))
        LB = np.asscalar(fsolve(g, 2))

        return LB, UB


    def zerosum(self, gamma, dual = False):
        """
        Given gamma, calculate the value and optimal strategies of a two-player
        zero-sum game given by the matrix

                M(gamma) = B - gamma*A.

        Row player maximizing, column player minimizing

        Zero-sum game as an LP (primal --> alpha)

            max (0', 1) @ (x', v)
            subject to
            [-M', ones(n, 1)] @ (x', v)' <= 0
            (x', v) @ (ones(m, 1), 0) = 1
            (x', v) >= (0', -inf)

        Zero-sum game as an LP (dual --> beta)

            min (0', 1) @ (p', u)
            subject to
            [M, -ones(m, 1)] @ (p', u)' <= 0
            (p', u) @ (ones(n, 1), 0) = 1
            (p', u) >= (0', -inf)

        Outputs:
        --------
        value: scalar
            value of the zero-sum game

        strategy: vector
            if dual = False, it is the intensity vector,
            if dual = True, it is the price vector
        """

        A, B, n, m = self.A, self.B, self.n, self.m
        M = B - gamma*A

        if dual == False:
            # Solve the primal LP (for details see the description)
            # (1) Define the problem for v as a maximization (linprog minimizes)
            c = np.hstack([np.zeros(m), -1])

            # (2) Add constraints :
            # ... non-negativity constaints
            bounds = tuple(m * [(0, None)] + [(None, None)])
            # ... inequality constaints
            A_iq = np.hstack([-M.T, np.ones((n, 1))])
            b_iq = np.zeros((n, 1))
            # ... normalization
            A_eq = np.hstack([np.ones(m), 0]).reshape(1, m + 1)
            b_eq = 1

            res = linprog(c, A_ub = A_iq, b_ub = b_iq, A_eq = A_eq, b_eq = b_eq,
                          bounds = bounds, options = dict(bland = True, tol = 1e-8))

        else:
            # Solve the dual LP (for details see the description)
            # (1) Define the problem for v as a maximization (linprog minimizes)
            c = np.hstack([np.zeros(n), 1])

            # (2) Add constraints :
            # ... non-negativity constaints
            bounds = tuple(n * [(0, None)] + [(None, None)])
            # ... inequality constaints
            A_iq = np.hstack([M, -np.ones((m, 1))])
            b_iq = np.zeros((m, 1))
            # ... normalization
            A_eq = np.hstack([np.ones(n), 0]).reshape(1, n + 1)
            b_eq = 1

            res = linprog(c, A_ub = A_iq, b_ub = b_iq, A_eq = A_eq, b_eq = b_eq,
                          bounds = bounds, options = dict(bland = True, tol = 1e-8))

        if res.status != 0:
            print(res.message)

        # Pull out the required quantities
        value = res.x[-1]
        strategy = res.x[:-1]

        return value, strategy


    def expansion(self, tol = 1e-8, maxit = 1000):
        """
        The algorithm used here is described in Hamburger-Thompson-Weil (1967, ECTA).
        It is based on a simple bisection argument and utilizes the idea that for
        a given gamma (= alpha or beta), the matrix "M = B - gamma*A" defines a
        two-player zero-sum game, where the optimal strategies are the (normalized)
        intensity and price vector.

        Outputs:
        --------
        alpha: scalar
            optimal expansion rate
        """

        LB, UB = self.bounds()

        for iter in range(maxit):

            gamma = (LB + UB) / 2
            ZS = self.zerosum(gamma = gamma)
            V = ZS[0]     # value of the game with gamma

            if V >= 0:
                LB = gamma
            else:
                UB = gamma

            if abs(UB - LB) < tol:
                gamma = (UB + LB) / 2
                x = self.zerosum(gamma = gamma)[1]
                p = self.zerosum(gamma = gamma, dual = True)[1]
                break

        return gamma, x, p

    def interest(self, tol = 1e-8, maxit = 1000):
        """
        The algorithm used here is described in Hamburger-Thompson-Weil (1967, ECTA).
        It is based on a simple bisection argument and utilizes the idea that for
        a given gamma (= alpha or beta), the matrix "M = B - gamma*A" defines a
        two-player zero-sum game, where the optimal strategies are the (normalized)
        intensity and price vector.

        Outputs:
        --------
        beta: scalar
            optimal interest rate
        """

        LB, UB = self.bounds()

        for iter in range(maxit):
            gamma = (LB + UB) / 2
            ZS = self.zerosum(gamma = gamma, dual = True)
            V = ZS[0]

            if V > 0:
                LB = gamma
            else:
                UB = gamma

            if abs(UB - LB) < tol:
                gamma = (UB + LB) / 2
                p = self.zerosum(gamma = gamma, dual = True)[1]
                x = self.zerosum(gamma = gamma)[1]
                break

        return gamma, x, p
