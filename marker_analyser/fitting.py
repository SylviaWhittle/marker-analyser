"""Code for fitting models to marker fd curves."""

import numpy as np
import numpy.typing as npt
from lumicks import pylake

def fit_model_to_data(
    distances: npt.NDArray[np.float_],
    forces: npt.NDArray[np.float_],
    model: pylake."
)