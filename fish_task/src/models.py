"""
Shared environment, agent models, and utility functions for the Fish Tank task.

The Fish Tank task is a probabilistic reversal learning paradigm in which
subjects observe fish drawn from one of three tanks and must identify which
tank the fish came from. Each tank has a distinct distribution over three
fish colors (Blue, Orange, Green). The source tank can reverse mid-block,
requiring subjects to adapt their beliefs.

Three cognitive models are provided:
  - 'heuristic'  : simple probability tracking (free param: p)
  - 'RL'         : Rescorla-Wagner reinforcement learning (free params: alpha, tau)
  - 'BL'         : Bayesian belief updating (free param: lambda)
"""

import numpy as np
from scipy.special import betainc


class fish_tank:
    """Simulates the Fish Tank environment.

    Parameters
    ----------
    n : int
        Number of tanks (always 3 in the standard task).
    tank1, tank2, tank3 : list of float
        Probability distributions over fish colors [Blue, Orange, Green]
        for each tank. Must each sum to 1.
    """

    def __init__(self, n, tank1, tank2, tank3):
        self.n = n
        self.tank1 = tank1
        self.tank2 = tank2
        self.tank3 = tank3
        self.tanks = [self.tank1, self.tank2, self.tank3]
        self.history = []
        self.t = 0

        for i, tank in enumerate([tank1, tank2, tank3], 1):
            if abs(sum(tank) - 1) > 1e-9:
                raise ValueError(f"Probabilities in tank{i} do not sum to 1.")

    def step(self, tank):
        """Draw one fish from the given tank distribution and record it."""
        self.history.append(np.random.choice([0, 1, 2], 1, p=tank)[0])
        self.t += 1

    def reset(self):
        """Clear the history and reset the trial counter."""
        self.history = []
        self.t = 0

    def choose_block(self):
        """Sample a block reversal schedule.

        Returns an array of length 10 indicating the number of reversals in
        each block: 1 block with 0 reversals, 8 with 1, 1 with 2.
        """
        bl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        block1 = np.random.permutation(bl)
        block2 = np.random.permutation(bl)
        block3 = np.random.permutation(bl)
        return [block1, block2, block3][np.random.choice(3)]

    def visualize(self):
        """Plot pie charts of the fish-color distributions for all three tanks."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, (tank, title) in enumerate(zip(
            [self.tank1, self.tank2, self.tank3],
            ['Tank 1', 'Tank 2', 'Tank 3']
        )):
            ax[i].pie(tank, labels=['Blue', 'Orange', 'Green'])
            ax[i].set_title(f'{title} - Fish distribution', fontsize=12)
        plt.tight_layout()
        return fig


class agent:
    """Cognitive agent that learns which fish tank is the source.

    Parameters
    ----------
    policy : str
        One of 'heuristic', 'RL', or 'BL'.
    **kwargs
        Policy-specific parameters:
        - heuristic : p (float) — probability increment per observation
        - RL        : alpha (float), tau (float) — learning rate and softmax temperature
        - BL        : lmbda (float) — likelihood concentration parameter
    """

    def __init__(self, policy=None, **kwargs):
        self.choices = np.array([0, 1, 2])
        self.policy = policy

        if self.policy == 'heuristic':
            self.p = kwargs['p']
        elif self.policy == 'RL':
            self.alpha = kwargs['alpha']
            self.tau = kwargs['tau']
        elif self.policy == 'BL':
            self.lmbda = kwargs['lmbda']
            self._build_likelihood(self.lmbda)
        elif self.policy == 'BL_beta':
            # Dynamic Bayesian: λ(prior) = th1 - betainc(alpha, beta, prior) * th2
            # Use dynamic_lambda(prior, alpha, beta, th1, th2) to compute λ at runtime.
            self.alpha = kwargs['alpha']
            self.beta = kwargs['beta']
            self.range = kwargs['th1']   # start value (max λ)
            self.start = kwargs['th2']   # range of λ variation
        else:
            raise ValueError(f"Unknown policy '{policy}'. Choose from: heuristic, RL, BL.")

    # ------------------------------------------------------------------
    # Trial iteration helpers
    # ------------------------------------------------------------------

    def iterateTrials_heuristic(self, TANKS_, tank_no, est_prob, trial):
        likelyTank = TANKS_.tanks[tank_no]
        TANKS_.step(likelyTank)
        observ = TANKS_.history[trial]
        pred = np.random.choice(self.choices, 1, p=est_prob[trial, :])[0]

        est_prob[trial + 1, observ] = est_prob[trial, observ] + self.p
        est_prob[trial + 1, :] = self.compute_prob(est_prob[trial + 1, :], option='Normalize')
        return pred, est_prob

    def iterateTrials_RL(self, TANKS_, tank_no, est_prob, est_value, trial):
        likelyTank = TANKS_.tanks[tank_no]
        TANKS_.step(likelyTank)
        pred = np.random.choice(self.choices, 1, p=est_prob[trial, :])[0]
        reward = 100 if pred == tank_no else 0

        est_value = self.update_vals(est_value, trial, pred, reward)
        est_prob[trial + 1, :] = self.compute_prob(est_value[trial + 1, :], option='Softmax')
        return pred, est_prob, est_value

    def iterateTrials_BL(self, TANKS_, tank_no, est_post, trial):
        likelyTank = TANKS_.tanks[tank_no]
        TANKS_.step(likelyTank)
        observ = TANKS_.history[trial]
        pred = np.random.choice(self.choices, 1, p=est_post[max(trial - 1, 0), :])[0]

        prior = est_post[trial, :] if trial == 0 else est_post[trial - 1, :]
        den = np.dot(self.llhood[observ, :], prior)
        est_post[trial, :] = np.multiply(self.llhood[observ, :], prior) / den

        est_post[trial, :] = np.maximum(0.05, est_post[trial, :])
        est_post[trial, :] /= est_post[trial, :].sum()
        return pred, est_post

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_prob(self, val, option):
        """Convert values to probabilities via normalization or softmax."""
        if option == 'Normalize':
            return val / val.sum()
        elif option == 'Softmax':
            e = np.exp(val / self.tau)
            return e / e.sum()

    def update_vals(self, vals, trial, pred, rwrd):
        """Rescorla-Wagner value update."""
        vals[trial + 1, :] = vals[trial, :]
        vals[trial + 1, pred] = vals[trial, pred] + self.alpha * (rwrd - vals[trial, pred])
        return vals

    def initialize_prob(self, n_trials, option='Softmax'):
        """Allocate and initialize probability/value arrays for a block."""
        if self.policy == 'heuristic':
            return np.ones((n_trials + 1, len(self.choices))) / len(self.choices)

        elif self.policy == 'RL':
            value = np.zeros((n_trials + 1, 3))
            prob = np.zeros((n_trials + 1, 3))
            value[0, :] = [33, 33, 33]
            prob[0, :] = self.compute_prob(value[0, :], option=option)
            return value, prob

        elif self.policy in ('BL', 'BL_beta'):
            return np.ones((n_trials, len(self.choices))) / len(self.choices)

    def update_llhood(self, lmbda):
        """Rebuild the likelihood matrix for a new lambda value."""
        self.lmbda = lmbda
        self._build_likelihood(lmbda)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_likelihood(self, lmbda):
        sigma = (1 - lmbda) / 2
        off = np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]])
        self.llhood = off * sigma + np.eye(3) * lmbda


# ---------------------------------------------------------------------------
# Analytical gradient functions for gradient-descent parameter fitting
# ---------------------------------------------------------------------------

def gradient_alpha(tau_, val, prev_val, rwrd, pred):
    """d/d_alpha of log P(choice | values; tau) under the Rescorla-Wagner model.

    Parameters
    ----------
    tau_     : float  — softmax temperature
    val      : array  — current value estimates (shape 3)
    prev_val : array  — previous value estimates (shape 3)
    rwrd     : float  — reward received on this trial
    pred     : int    — chosen option index
    """
    num = np.dot((rwrd - prev_val) / tau_, np.exp(val / tau_))
    den = np.exp(val / tau_).sum()
    d_den = num / den
    d_num = (rwrd - prev_val[pred]) / tau_
    return d_num - d_den


def gradient_tau(tau_, val, prev_val, rwrd, pred):
    """d/d_tau of log P(choice | values; tau) under the Rescorla-Wagner model.

    Parameters
    ----------
    tau_     : float  — softmax temperature
    val      : array  — current value estimates (shape 3)
    prev_val : array  — previous value estimates (shape 3, unused but kept for API symmetry)
    rwrd     : float  — reward received on this trial (unused but kept for API symmetry)
    pred     : int    — chosen option index
    """
    num = np.dot(-val / tau_**2, np.exp(val / tau_))
    den = np.exp(val / tau_).sum()
    d_den = num / den
    d_num = -val[pred] / tau_**2
    return d_num - d_den


def dynamic_lambda(prior, alpha, beta, th1, th2):
    """Compute the dynamic likelihood concentration λ for the BL_beta model.

    The likelihood parameter is not fixed but varies as a function of the
    agent's current prior belief, shaped by the regularized incomplete Beta
    function:

        λ(prior) = th1 - betainc(alpha, beta, prior) * th2

    Depending on the sign of th2 and the shape of the Beta CDF, λ can be
    monotonically decreasing (th2 > 0) or increasing (th2 < 0) in the prior.

    Parameters
    ----------
    prior : float or array  — current prior probability (in [0, 1])
    alpha : float           — Beta distribution shape parameter α > 0
    beta  : float           — Beta distribution shape parameter β > 0
    th1   : float           — start (maximum) value of λ
    th2   : float           — signed range controlling direction and magnitude

    Returns
    -------
    float or array — dynamic λ value(s), clipped to [0, 1]
    """
    lmbda = th1 - betainc(alpha, beta, prior) * th2
    return np.clip(lmbda, 0.0, 1.0)


def gradient_prob(p_, prob, pred, observ):
    """d/d_p of log P(choice | prob; p) under the Heuristic model.

    Parameters
    ----------
    p_     : float — probability increment parameter
    prob   : array — current choice probability vector (shape 3)
    pred   : int   — chosen option index
    observ : int   — observed fish color index
    """
    if pred == observ:
        d_num = (1 - prob[pred]) / (1 + p_)
        d_den = 1 / (prob[pred] + p_)
        return d_num * d_den
    else:
        return -1 / (1 + p_)
