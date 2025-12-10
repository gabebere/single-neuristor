

"""
wiener_demo.py

Small standalone demo of the Wiener process ΔW used as the stochastic
noise term in the neuristor model.

In the continuous-time formulation we write, schematically,

    dX(t) = ... + σ dW(t)

where W(t) is a Wiener process (Brownian motion) and dW(t) is its
increment. In discrete time with time step Δt we approximate this as

    ΔW_n ≈ sqrt(Δt) * ξ_n,  with  ξ_n ~ N(0, 1) i.i.d.,

so that ΔW_n ~ N(0, Δt). The Wiener process itself is the cumulative
sum of these Gaussian increments,

    W_{n+1} = W_n + ΔW_n,   with  W_0 = 0.

This script generates one realization of W(t) and plots both W(t) and
its increments ΔW_n over time, to illustrate that ΔW is the
stochastic noise driving the system.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_wiener(T: float = 1.0, dt: float = 1e-3, seed: int | None = 0):
    """
    Generate a single realization of a Wiener process on [0, T].

    Parameters
    ----------
    T : float
        Total time horizon (arbitrary units).
    dt : float
        Time step. The variance of each increment is dt.
    seed : int | None
        Seed for the random number generator (for reproducibility).
        Use None for a fresh random seed each run.

    Returns
    -------
    t : np.ndarray, shape (N+1,)
        Time grid from 0 to T.
    W : np.ndarray, shape (N+1,)
        Wiener process values W(t).
    dW : np.ndarray, shape (N,)
        Increments ΔW_n ~ N(0, dt).
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    # Draw independent standard normal variables ξ_n ~ N(0, 1)
    # and scale them to get ΔW_n ~ N(0, dt).
    dW = sqrt_dt * rng.standard_normal(size=n_steps)

    # Cumulative sum gives a realization of W(t), with W(0) = 0.
    W = np.empty(n_steps + 1)
    W[0] = 0.0
    W[1:] = np.cumsum(dW)

    return t, W, dW


def main():
    # Choose a short time horizon and fine time step – the units are
    # arbitrary here, we just want to visualize the random walk.
    T = 1.0
    dt = 1e-3

    t, W, dW = generate_wiener(T, dt, seed=42)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    # Top panel: the Wiener process W(t) itself (integrated noise)
    ax1.plot(t, W, lw=1.5)
    ax1.set_ylabel("W(t)")
    ax1.set_title("One realization of a Wiener process W(t)")

    # Bottom panel: the increments ΔW_n that appear in the discretized SDE
    # We use a step plot to emphasize piecewise-constant increments.
    ax2.step(t[:-1], dW, where="post", lw=1.0)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("ΔW")
    ax2.set_title("Gaussian increments ΔW ~ N(0, Δt) (stochastic noise)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()