"""
File: Wald_Friedman.py

Authors: Chase Coleman, Tom Sargent

This presents a version of the problem that Friedman couldn't solve, and
will use dynamic programming in a clever way in order to help the Navy
choose the appropriate projectile.

References
----------
Friedman, Milton and Friedman, Rose. Two lucky people: memoirs / Milton & Rose
D. Friedman. The University of Chicago Press. Chicago. 1998
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.stats as st
import seaborn as sb
import quantecon as qe


class WaldFriedman(object):
    """
    This class is used to solve the problem presented in the "Wald Friedman"
    notebook presented on the QuantEcon website.

    Parameters
    ----------
    c : scalar(Float64)
        Cost of postponing decision
    L0 : scalar(Float64)
        Cost of choosing model 0 when the truth is model 1
    L1 : scalar(Float64)
        Cost of choosing model 1 when the truth is model 0
    f0 : array_like(Float64)
        A finite state probability distribution
    f1 : array_like(Float64)
        A finite state probability distribution
    m : scalar(Int)
        Number of points to use in function approximation
    """
    def __init__(self, c, L0, L1, f0, f1, m=25):
        self.c = c
        self.L0, self.L1 = L0, L1
        self.m = m
        self.pgrid = np.linspace(0.0, 1.0, m)

        # Renormalize distributions so nothing is "too" small
        f0 = np.clip(f0, 1e-8, 1-1e-8)
        f1 = np.clip(f1, 1e-8, 1-1e-8)
        self.f0 = f0 / np.sum(f0)
        self.f1 = f1 / np.sum(f1)
        self.J = np.zeros(m)

    def current_distribution(self, p):
        """
        This function takes a value for the probability with which
        the correct model is model 0 and returns the mixed
        distribution that corresponds with that belief.
        """
        return p*self.f0 + (1-p)*self.f1

    def bayes_update_k(self, p, k):
        """
        This function takes a value for p, and a realization of the
        random variable and calculates the value for p tomorrow.
        """
        f0_k = self.f0[k]
        f1_k = self.f1[k]

        p_tp1 = p*f0_k / (p*f0_k + (1-p)*f1_k)

        return np.clip(p_tp1, 0, 1)

    def bayes_update_all(self, p):
        """
        This is similar to `bayes_update_k` except it returns a
        new value for p for each realization of the random variable
        """
        return np.clip(p*self.f0 / (p*self.f0 + (1-p)*self.f1), 0, 1)

    def payoff_choose_f0(self, p):
        "For a given probability specify the cost of accepting model 0"
        return (1-p)*self.L0

    def payoff_choose_f1(self, p):
        "For a given probability specify the cost of accepting model 1"
        return p*self.L1

    def EJ(self, p, J):
        """
        This function evaluates the expectation of the value function
        at period t+1. It does so by taking the current probability
        distribution over outcomes:

            p(z_{k+1}) = p_k f_0(z_{k+1}) + (1-p_k) f_1(z_{k+1})

        and evaluating the value function at the possible states
        tomorrow J(p_{t+1}) where

            p_{t+1} = p f0 / ( p f0 + (1-p) f1)

        Parameters
        ----------
        p : Scalar(Float64)
            The current believed probability that model 0 is the true
            model.
        J : Function
            The current value function for a decision to continue

        Returns
        -------
        EJ : Scalar(Float64)
            The expected value of the value function tomorrow
        """
        # Pull out information
        f0, f1 = self.f0, self.f1

        # Get the current believed distribution and tomorrows possible dists
        # Need to clip to make sure things don't blow up (go to infinity)
        curr_dist = self.current_distribution(p)
        tp1_dist = self.bayes_update_all(p)

        # Evaluate the expectation
        EJ = curr_dist @ J(tp1_dist)

        return EJ

    def payoff_continue(self, p, J):
        """
        For a given probability distribution and value function give
        cost of continuing the search for correct model
        """
        return self.c + self.EJ(p, J)

    def bellman_operator(self, J):
        """
        Evaluates the value function for a given continuation value
        function; that is, evaluates

            J(p) = min( (1-p)L0, pL1, c + E[J(p')])

        Uses linear interpolation between points
        """
        payoff_choose_f0 = self.payoff_choose_f0
        payoff_choose_f1 = self.payoff_choose_f1
        payoff_continue = self.payoff_continue
        c, L0, L1, f0, f1 = self.c, self.L0, self.L1, self.f0, self.f1
        m, pgrid = self.m, self.pgrid

        J_out = np.empty(m)
        J_interp = interp.UnivariateSpline(pgrid, J, k=1, ext=0)

        for (p_ind, p) in enumerate(pgrid):
            # Payoff of choosing model 0
            p_c_0 = payoff_choose_f0(p)
            p_c_1 = payoff_choose_f1(p)
            p_con = payoff_continue(p, J_interp)

            J_out[p_ind] = min(p_c_0, p_c_1, p_con)

        return J_out

    def solve_model(self, tol=1e-7):
        J =  qe.compute_fixed_point(self.bellman_operator, np.zeros(self.m),
                                    error_tol=tol, verbose=False)

        self.J = J
        return J

    def find_cutoff_rule(self, J):
        """
        This function takes a value function and returns the corresponding
        cutoffs of where you transition between continue and choosing a
        specific model
        """
        payoff_choose_f0 = self.payoff_choose_f0
        payoff_choose_f1 = self.payoff_choose_f1
        m, pgrid = self.m, self.pgrid

        # Evaluate cost at all points on grid for choosing a model
        p_c_0 = payoff_choose_f0(pgrid)
        p_c_1 = payoff_choose_f1(pgrid)

        # The cutoff points can be found by differencing these costs with
        # the Bellman equation (J is always less than or equal to p_c_i)
        lb = pgrid[np.searchsorted(p_c_1 - J, 1e-10) - 1]
        ub = pgrid[np.searchsorted(J - p_c_0, -1e-10)]

        return (lb, ub)

    def simulate(self, f, p0=0.5):
        """
        This function takes an initial condition and simulates until it
        stops (when a decision is made).
        """
        # Check whether vf is computed
        if np.sum(self.J) < 1e-8:
            self.solve_model()

        # Unpack useful info
        lb, ub = self.find_cutoff_rule(self.J)
        update_p = self.bayes_update_k
        curr_dist = self.current_distribution
        drv = qe.discrete_rv.DiscreteRV(f)

        # Initialize a couple useful variables
        decision_made = False
        p = p0
        t = 0

        while decision_made is False:
            # Maybe should specify which distribution is correct one so that
            # the draws come from the "right" distribution
            k = drv.draw()[0]
            t = t+1
            p = update_p(p, k)
            if p < lb:
                decision_made = True
                decision = 1
            elif p > ub:
                decision_made = True
                decision = 0

        return decision, p, t

    def simulate_tdgp_f0(self, p0=0.5):
        """
        Uses the distribution f0 as the true data generating
        process
        """
        decision, p, t = self.simulate(self.f0, p0)

        if decision == 0:
            correct = True
        else:
            correct = False

        return correct, p, t

    def simulate_tdgp_f1(self, p0=0.5):
        """
        Uses the distribution f1 as the true data generating
        process
        """
        decision, p, t = self.simulate(self.f1, p0)

        if decision == 1:
            correct = True
        else:
            correct = False

        return correct, p, t

    def stopping_dist(self, ndraws=250, tdgp="f0"):
        """
        Simulates repeatedly to get distributions of time needed to make a
        decision and how often they are correct.
        """
        if tdgp=="f0":
            simfunc = self.simulate_tdgp_f0
        else:
            simfunc = self.simulate_tdgp_f1

        # Allocate space
        tdist = np.empty(ndraws, int)
        cdist = np.empty(ndraws, bool)

        for i in range(ndraws):
            correct, p, t = simfunc()
            tdist[i] = t
            cdist[i] = correct

        return cdist, tdist


def make_distribution_plots(f0, f1):
    """
    This generates the figure that shows the initial versions
    of the distributions and plots their combinations.
    """
    fig, ax = plt.subplots(2, figsize=(10, 8))

    ax[0].set_title("Original Distributions")
    ax[0].set_xlabel(r"$k$ Values")
    ax[0].set_ylabel(r"Probability of $z_k$")
    ax[0].plot(f0, label=r"$f_0$")
    ax[0].plot(f1, label=r"$f_1$")
    ax[0].legend()

    ax[1].set_title("Mixtures of Original Distributions")
    ax[1].set_xlabel(r"$k Values$")
    ax[1].set_ylabel(r"Probability of $z_k$")
    ax[1].plot(0.25*f0 + 0.75*f1, label=r"$p_k$ = 0.25")
    ax[1].plot(0.5*f0 + 0.5*f1, label=r"$p_k$ = 0.50")
    ax[1].plot(0.75*f0 + 0.25*f1, label=r"$p_k$ = 0.75")
    ax[1].legend()

    fig.tight_layout()
    return fig

def WaldFriedman_Interactive(m):
    # NOTE: Could add sliders over other variables
    #       as well, but only doing over n for now

    # Choose parameters
    c = 1.25
    L0 = 25.0
    L1 = 25.0

    # Choose n points and distributions
    f0 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=2.0, b=2.5), 1e-6, np.inf)
    f0 = f0 / np.sum(f0)
    f1 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=2.5, b=2.0), 1e-6, np.inf)
    f1 = f1 / np.sum(f1)  # Make sure sums to 1

    # Create WaldFriedman class
    wf = WaldFriedman(c, L0, L1, f0, f1, m=m)

    # Solve via VFI
    # Solve using qe's `compute_fixed_point` function
    J = qe.compute_fixed_point(wf.bellman_operator, np.zeros(m),
                               error_tol=1e-7, verbose=False,
                               max_iter=1000)

    lb, ub = wf.find_cutoff_rule(J)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    fig.suptitle("Value function", size=18)
    ax.set_xlabel("Probability of Model 0")
    ax.set_ylabel("Value Function")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.5 * max(L0, L1))
    ax.plot(wf.pgrid, J)

    ax.annotate(r"$\beta$", xy=(ub+0.025, 0.5), size=14)
    ax.annotate(r"$\alpha$", xy=(lb+0.025, 0.5), size=14)
    ax.vlines(lb, 0.0, wf.payoff_choose_f1(lb), linestyle="--")
    ax.vlines(ub, 0.0, wf.payoff_choose_f0(ub), linestyle="--")

    fig.show()

def all_param_interact(c, L0, L1, a0, b0, a1, b1, m):
    f0 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=a0, b=b0), 1e-6, np.inf)
    f0 = f0 / np.sum(f0)
    f1 = np.clip(st.beta.pdf(np.linspace(0, 1, m), a=a1, b=b1), 1e-6, np.inf)
    f1 = f1 / np.sum(f1)  # Make sure sums to 1

    # Create an instance of our WaldFriedman class
    wf = WaldFriedman(c, L0, L1, f0, f1, m=m)
    # Solve using qe's `compute_fixed_point` function
    J = qe.compute_fixed_point(wf.bellman_operator, np.zeros(m),
                               error_tol=1e-7, verbose=False,
                               print_skip=10, max_iter=500)
    lb, ub = wf.find_cutoff_rule(J)

    # Get draws
    ndraws = 500
    cdist, tdist = wf.stopping_dist(ndraws=ndraws)

    fig, ax = plt.subplots(2, 2, figsize=(22, 14))

    ax[0, 0].plot(f0, marker="o", markersize=2.5, linestyle="None", label=r"$f_0$")
    ax[0, 0].plot(f1, marker="o", markersize=2.5, linestyle="None", label=r"$f_1$")
    ax[0, 0].set_ylabel(r"Probability of $z_k$")
    ax[0, 0].set_xlabel(r"$k$")
    ax[0, 0].set_title("Distributions over Outcomes", size=24)

    ax[0, 1].plot(wf.pgrid, J)
    ax[0, 1].annotate(r"$\alpha$", xy=(lb+0.025, 0.5), size=14)
    ax[0, 1].annotate(r"$\beta$", xy=(ub+0.025, 0.5), size=14)
    ax[0, 1].vlines(lb, 0.0, wf.payoff_choose_f1(lb), linestyle="--")
    ax[0, 1].vlines(ub, 0.0, wf.payoff_choose_f0(ub), linestyle="--")
    ax[0, 1].set_ylim(0, 0.5*max(L0, L1))
    ax[0, 1].set_ylabel("Value of Bellman")
    ax[0, 1].set_xlabel(r"$p_k$")
    ax[0, 1].set_title("Bellman Equation", size=24)

    ax[1, 0].hist(tdist, bins=np.max(tdist))
    ax[1, 0].set_title("Stopping Times", size=24)
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Density")

    ax[1, 1].hist(cdist, bins=2)
    ax[1, 1].set_title("Correct Decisions", size=24)
    ax[1, 1].annotate("Percent Correct p={}".format(np.mean(cdist)), xy=(0.05, ndraws/2), size=18)

    fig.tight_layout()
    fig.show()

def convert_rgb(x):
    return tuple(map(lambda c: int(256*c), x))

def convert_rgb_hex(rgb):
    if isinstance(rgb[0], int):
        return '#%02x%02x%02x' % rgb
    else:
        rgbint = convert_rgb(rgb)
        return '#%02x%02x%02x' % rgbint
