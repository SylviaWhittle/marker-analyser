"""Code for fitting models to marker fd curves."""

import numpy as np
import numpy.typing as npt
from lumicks import pylake


def fit_model_to_data(
    distances: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    model: pylake.fitting.model.Model,
    lp_value: float | None = None,
    lp_lower_bound: float | None = None,
    lp_upper_bound: float | None = None,
    lc_value: float | None = None,
    force_offset_lower_bound: float | None = None,
    force_offset_upper_bound: float | None = None,
) -> tuple[pylake.FdFit, npt.NDArray[np.float64], pylake.FdFit.params, float]:
    """
    Fit a model to force-distance data.

    Parameters
    ----------
    distances : npt.NDArray[np.float64]
        Array of distance values.
    forces : npt.NDArray[np.float64]
        Array of force values.
    model : pylake.fitting.model.Model
        The model to fit to the data.
    lp_value : float | None, optional
        Initial guess for persistence length.
    lp_lower_bound : float | None, optional
        Lower bound for persistence length.
    lp_upper_bound : float | None, optional
        Upper bound for persistence length.
    lc_value : float | None, optional
        Initial guess for contour length.
    force_offset_lower_bound : float | None, optional
        Lower bound for force offset.
    force_offset_upper_bound : float | None, optional
        Upper bound for force offset.

    Returns
    -------
    tuple[pylake.FdFit, npt.NDArray[np.float64], pylake.FdFit.params, float]
        A tuple containing the fit object, modelled forces, fit parameters, and fit error.
    """

    model = model(name="fit") + pylake.force_offset(name="fit")
    fit = pylake.FdFit(model)
    fit.add_data(name="data", f=forces, d=distances)
    if lp_value is not None:
        fit["fit/Lp"].value = lp_value  # Initial guess for persistence length in nm
    if lp_lower_bound is not None:
        fit["fit/Lp"].lower_bound = lp_lower_bound  # Lower bound for persistence length in nm
    if lp_upper_bound is not None:
        fit["fit/Lp"].upper_bound = lp_upper_bound  # Upper bound for persistence length in nm
    if lc_value is not None:
        fit["fit/Lc"].value = lc_value  # Initial guess for contour length in nm
    if lc_value is not None:
        fit["fit/Lc"].value = lc_value  # Initial guess for contour length in nm
    if force_offset_lower_bound is not None:
        fit["fit/f_offset"].lower_bound = force_offset_lower_bound  # Lower bound for force offset
    if force_offset_upper_bound is not None:
        fit["fit/f_offset"].upper_bound = force_offset_upper_bound  # Upper bound for force offset
    # Perform the fit
    fit.fit()
    # Note that the error for some reason is the same value repeated for each point. Let's take
    # the mean value just in case this changes.
    fit_error = np.mean(fit.sigma)

    modelled_forces = model(independent=distances, params=fit.params)

    return pylake.FdFit, modelled_forces, fit.params, fit_error
