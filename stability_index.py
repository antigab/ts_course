import numpy as np


def calculate_si(real, reference):
    """
    Calculates population stability index using following formula
    SI = Σ (pA i - pB i ) * ( ln(pA i ) - ln(pB i ) )

    Parameters
    ----------
    real : 1d np.array
        Distribution for
    reference : 1d np.array
        Reference distribution.

    Returns
    -------
    stability_index : float
    """
    si = (real - reference) * (np.log(real) - np.log(reference))

    return np.sum(si)
