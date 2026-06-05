import numpy as np

def get_hamming_win(fs):
    """Generate hamming window for given sampling points.

    Args:
        fs (_type_): _description_

    Returns:
        _type_: _description_
    """
    tmp = np.linspace(0, fs - 1, fs)
    hamming_win = 0.54 - 0.46 * np.cos(2 * np.pi * tmp / (fs - 1))
    return hamming_win