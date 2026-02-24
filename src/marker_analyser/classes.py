"""Classes and dataclasses for data storage and handling."""

# pylint: disable=too-many-lines

from pathlib import Path

import re
from typing import Any, Generator
from pydantic import BaseModel, ConfigDict, PrivateAttr
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# lumicks doesn't provide type stubs, so have ignored mypy warnings for missing imports in the mypy arguments in
# settings.
from lumicks import pylake
from skimage.morphology import label

from marker_analyser.fitting import fit_model_to_data
from marker_analyser.plotting import PALETTE
from marker_analyser.parsers import extract_metadata_from_fd_curve_name_with_regex
from marker_analyser.data_manipulation import create_df_from_uneven_data


class MarkerAnalysisBaseModel(BaseModel):
    """Data object to hold settings for Models used in the project."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForcePeakModel(MarkerAnalysisBaseModel):
    """A data object to hold force peak data."""

    distance: float
    force: float
    index: int


class FitResult(MarkerAnalysisBaseModel):
    """A data object to hold fit result data."""

    fitted_forces: npt.NDArray[np.float64]
    params: Any
    fit_error: float


# pylint: disable=too-many-instance-attributes
class OscillationModel(MarkerAnalysisBaseModel):
    """A data object to hold oscillation data."""

    id: str
    curve_id: str
    marker_filename: str
    metadata: dict[str, float | str | int | None]
    increasing_force: npt.NDArray[np.float64]
    increasing_distance: npt.NDArray[np.float64]
    _increasing_force_raw: npt.NDArray[np.float64] | None = PrivateAttr(None)
    _increasing_distance_raw: npt.NDArray[np.float64] | None = PrivateAttr(None)
    decreasing_force: npt.NDArray[np.float64]
    decreasing_distance: npt.NDArray[np.float64]
    _decreasing_force_raw: npt.NDArray[np.float64] | None = PrivateAttr(None)
    _decreasing_distance_raw: npt.NDArray[np.float64] | None = PrivateAttr(None)

    # Fitting
    force_peaks: list[ForcePeakModel] | None = None
    num_peaks: int | None = None
    increasing_fit: FitResult | None = None
    decreasing_fit: FitResult | None = None
    fit_both: FitResult | None = None

    # Masking
    force_maximum: float | None = None
    distance_minimum: float | None = None
    increasing_mask: npt.NDArray[np.bool_] | None = None
    decreasing_mask: npt.NDArray[np.bool_] | None = None

    # pylint: disable=arguments-differ
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialisation hook to run after the model is created.

        Parameters
        ----------
        __context : Any
            Unsure, was in an example I found.
        """
        # Create copies of the raw data for masking purposes
        self._increasing_force_raw = self.increasing_force.copy()
        self._increasing_distance_raw = self.increasing_distance.copy()
        self._decreasing_force_raw = self.decreasing_force.copy()
        self._decreasing_distance_raw = self.decreasing_distance.copy()
        # Create mask of the data based on thresholds
        self.calculate_masks()

    @property
    def increasing_force_raw(self) -> npt.NDArray[np.float64]:
        """
        Get the raw increasing force data.

        Returns
        -------
        npt.NDArray[np.float64]
            The raw increasing force data.
        """
        assert self._increasing_force_raw is not None
        return self._increasing_force_raw

    @property
    def increasing_distance_raw(self) -> npt.NDArray[np.float64]:
        """
        Get the raw increasing distance data.

        Returns
        -------
        npt.NDArray[np.float64]
            The raw increasing distance data.
        """
        assert self._increasing_distance_raw is not None
        return self._increasing_distance_raw

    @property
    def decreasing_force_raw(self) -> npt.NDArray[np.float64]:
        """
        Get the raw decreasing force data.

        Returns
        -------
        npt.NDArray[np.float64]
            The raw decreasing force data.
        """
        assert self._decreasing_force_raw is not None
        return self._decreasing_force_raw

    @property
    def decreasing_distance_raw(self) -> npt.NDArray[np.float64]:
        """
        Get the raw decreasing distance data.

        Returns
        -------
        npt.NDArray[np.float64]
            The raw decreasing distance data.
        """
        assert self._decreasing_distance_raw is not None
        return self._decreasing_distance_raw

    # Getter for force_both (both increasing and decreasing concatenated)
    @property
    def forces_both(self) -> npt.NDArray[np.float64]:
        """
        Get the concatenated force data (both increasing and decreasing).

        Returns
        -------
        npt.NDArray[np.float64]
            The concatenated force data.
        """
        return np.concatenate([self.increasing_force, self.decreasing_force])

    # Getter for distance_both (both increasing and decreasing concatenated)
    @property
    def distances_both(self) -> npt.NDArray[np.float64]:
        """
        Get the concatenated distance data (both increasing and decreasing).

        Returns
        -------
        npt.NDArray[np.float64]
            The concatenated distance data.
        """
        return np.concatenate([self.increasing_distance, self.decreasing_distance])

    def __repr__(self) -> str:
        increasing_fit_err_str = f"err: {self.increasing_fit.fit_error:.2f}" if self.increasing_fit else ""
        increasing_fit_str = f"increasing fit: {self.increasing_fit is not None} {increasing_fit_err_str}"
        decreasing_fit_err_str = f"err: {self.decreasing_fit.fit_error:.2f}" if self.decreasing_fit else ""
        decreasing_fit_str = f"decreasing fit: {self.decreasing_fit is not None} {decreasing_fit_err_str}"
        return f"OscillationModel | num_peaks: {self.num_peaks} | {increasing_fit_str} | {decreasing_fit_str}"

    def __str__(self) -> str:
        """
        Print representation of the object.

        Returns
        -------
        str
            String representation of the object.
        """
        return self.__repr__()

    def _repr_pretty_(self, printer) -> None:
        """
        IPython pretty-printer (notebook / rich display).

        Parameters
        ----------
        printer : Any
            The IPython pretty-printer object.
        """
        printer.text(repr(self))

    def calculate_masks(
        self,
    ) -> None:
        """Calculate masks for the oscillation data based on thresholds."""
        increasing_mask = np.ones_like(self.increasing_force, dtype=bool)
        decreasing_mask = np.ones_like(self.decreasing_force, dtype=bool)
        if self.force_maximum is not None:
            increasing_mask &= self.increasing_force <= self.force_maximum
            decreasing_mask &= self.decreasing_force <= self.force_maximum
        if self.distance_minimum is not None:
            increasing_mask &= self.increasing_distance >= self.distance_minimum
            decreasing_mask &= self.decreasing_distance >= self.distance_minimum
        self.increasing_mask = increasing_mask
        self.decreasing_mask = decreasing_mask
        self.increasing_force = self.increasing_force[self.increasing_mask]
        self.increasing_distance = self.increasing_distance[self.increasing_mask]
        self.decreasing_force = self.decreasing_force[self.decreasing_mask]
        self.decreasing_distance = self.decreasing_distance[self.decreasing_mask]

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-positional-arguments
    def find_peaks(
        self,
        peak_height: tuple[float, float] = (0.5, 30),
        prominence: float = 0.8,
        verbose: bool = False,
        plotting: bool = False,
        oscillation_index: int | None = None,
        curve_id: str | None = None,
    ) -> None:
        """
        Find force peaks in the oscillation data's increasing distance-force curve.

        Note that I think we are making a huge assumption that the distance changes at a constant rate, since we don't
        fit the peaks on the 2d data, but rather only on the force data, ignoring the distance components.

        Parameters
        ----------
        peak_height : tuple[float, float], optional
            The minimum and maximum height of the peaks to be detected.
        prominence : float, optional
            The prominence of the peaks to be detected.
        verbose : bool, optional
            If True, print additional information about the peaks found.
        plotting : bool, optional
            If True, plot the increasing segment with the detected peaks.
        oscillation_index : int
            The index of the oscillation within the force-distance curve.
        curve_id : str
            The ID of the force-distance curve containing the oscillation.
        """
        increasing_force = self.increasing_force
        increasing_distance = self.increasing_distance
        peak_indexes, _ = find_peaks(
            increasing_force,
            height=peak_height,
            prominence=prominence,
        )

        if verbose:
            print(
                f"Found {len(peak_indexes)} peaks in increasing segment of"
                f"oscillation {oscillation_index} for curve {curve_id}."
            )

        # Check if the peaks meet the criteria for having at least a certain ratio of the data preceding it being
        # increasing. This is to hopefully avoid noise spikes being detected as peaks.
        vetted_peak_indexes = []
        for peak_number, peak_index in enumerate(peak_indexes):
            if peak_number == 0:
                # always keep the first
                vetted_peak_indexes.append(peak_index)
            else:
                previous_peak_index = peak_indexes[peak_number - 1]
                if verbose:
                    print(f"vetting peak {peak_number} at index {peak_index}")
                if self.should_vet_peak_increasing_force_criteria(
                    peak_index=peak_index,
                    previous_peak_index=previous_peak_index,
                ):
                    print(
                        f"Peak {peak_number} at index {peak_index} ({increasing_distance[peak_index]:.2f} um) in"
                        f"oscillation {oscillation_index} for curve {curve_id} "
                        f"did not meet the increasing force criteria and was removed."
                    )
                else:
                    vetted_peak_indexes.append(peak_index)

        if len(vetted_peak_indexes) == 0:
            if verbose:
                print(
                    f"No peaks passed vetting in increasing segment of oscillation {oscillation_index} for"
                    f"curve {curve_id}."
                )
            self.force_peaks = []
            self.num_peaks = 0
            return
        if plotting:
            # plot the increasing segment with the peaks
            plt.plot(increasing_distance, increasing_force, label="increasing")
            # vlines for the peaks
            plt.vlines(
                increasing_distance[vetted_peak_indexes],
                ymin=np.min(increasing_force),
                ymax=np.max(increasing_force),
                color="grey",
                label="peaks",
                linestyle="--",
            )
            plt.title(f"Peaks in increasing segment of oscillation {oscillation_index} for curve {curve_id}")
            plt.xlabel("Distance (um)")
            plt.ylabel("Force (pN)")
            plt.legend()
            plt.show()
        force_peaks = []
        for peak_index in vetted_peak_indexes:
            force_peak = ForcePeakModel(
                distance=increasing_distance[peak_index],
                force=increasing_force[peak_index],
                index=peak_index,
            )
            force_peaks.append(force_peak)
        self.force_peaks = force_peaks
        self.num_peaks = len(force_peaks)

    def should_vet_peak_increasing_force_criteria(
        self,
        peak_index: int,
        previous_peak_index: int,
        minimum_increasing_decreasing_ratio: float = 0.5,
        verbose: bool = False,
    ) -> bool:
        """
        Check if a peak meets the criteria for strength for continued analysis.

        Checks if a peak meets the criteria for having at least a certain ratio of the data preceding
        it being increasing.

        This is to hopefully avoid noise spikes being detected as peaks.

        Parameters
        ----------
        peak_index : int
            The index of the peak to be checked.
        previous_peak_index : int
            The index of the previous peak.
        minimum_increasing_decreasing_ratio : float, optional
            The minimum ratio of the distance after the minimum force to the distance before the minimum force.
            Default is 0.5.
        verbose : bool, optional
            If True, print additional information during the vetting process. Default is False.

        Returns
        -------
        bool
            True if the peak does not meet the criteria and should be deleted, False otherwise.
        """
        # find the minimum force between the previous peak and this peak
        oscillation_distance_data = self.increasing_distance
        oscillation_force_data = self.increasing_force
        between_peak_minimum_force_index = (
            np.argmin(oscillation_force_data[previous_peak_index : peak_index + 1]) + previous_peak_index
        )
        if verbose:
            print(
                f"indexes: previous peak {previous_peak_index}, current peak {peak_index}, minimum"
                f"force {between_peak_minimum_force_index}"
            )
        distance_at_previous_peak = oscillation_distance_data[previous_peak_index]
        distance_at_minimum_force = oscillation_distance_data[between_peak_minimum_force_index]
        distance_at_current_peak = oscillation_distance_data[peak_index]
        distance_before_minimum = distance_at_minimum_force - distance_at_previous_peak
        distance_after_minimum = distance_at_current_peak - distance_at_minimum_force
        if verbose:
            print(
                f"distance at: previous peak: {distance_at_previous_peak}, minimum force: {distance_at_minimum_force},"
                f"current peak: {distance_at_current_peak}"
            )
            print(f"Distance before minimum: {distance_before_minimum}")
            print(f"Distance after minimum: {distance_after_minimum}")
        distance_increasing_decreasing_ratio = distance_after_minimum / distance_before_minimum
        if verbose:
            print(f"Distance increasing/decreasing ratio: {distance_increasing_decreasing_ratio:.2f}")
        if distance_increasing_decreasing_ratio < minimum_increasing_decreasing_ratio:
            return True
        return False

    def plot(
        self,
        increasing_colour: str = "tab:blue",
        decreasing_colour: str = "tab:green",
        show: bool = True,
        increasing_segment: bool = True,
        decreasing_segment: bool = True,
    ) -> None:
        """
        Plot the oscillation's force-distance data.

        Parameters
        ----------
        increasing_colour : str, optional
            The colour to use for the increasing segment.
        decreasing_colour : str, optional
            The colour to use for the decreasing segment.
        show : bool, optional
            Whether to show the plot immediately.
        increasing_segment : bool, optional
            Whether to plot the increasing segment.
        decreasing_segment : bool, optional
            Whether to plot the decreasing segment.
        """

        if increasing_segment:
            plt.plot(self.increasing_distance, self.increasing_force, color=increasing_colour, alpha=0.5)
            if self.increasing_fit is not None:
                plt.plot(self.increasing_distance, self.increasing_fit.fitted_forces, color=increasing_colour, alpha=1)
        if decreasing_segment:
            plt.plot(self.decreasing_distance, self.decreasing_force, color=decreasing_colour, alpha=0.5)
            if self.decreasing_fit is not None:
                plt.plot(self.decreasing_distance, self.decreasing_fit.fitted_forces, color=decreasing_colour, alpha=1)
        plt.xlabel("Distance (um)")
        plt.ylabel("Force (pN)")
        plt.title("")
        if show:
            plt.show()

    def fit_model(
        self,
        segment: str,
        lp_value: float | None = None,
        lp_lower_bound: float | None = None,
        lp_upper_bound: float | None = None,
        lc_value: float | None = None,
        force_offset_lower_bound: float | None = None,
        force_offset_upper_bound: float | None = None,
    ) -> None:
        """
        Fit the specified segment of the oscillation.

        Parameters
        ----------
        segment : str
            The segment to fit, either "increasing" or "decreasing" or "both".
        lp_value : float | None
            Initial guess for persistence length.
        lp_lower_bound : float | None
            Lower bound for persistence length.
        lp_upper_bound : float | None
            Upper bound for persistence length.
        lc_value : float | None
            Initial guess for contour length.
        force_offset_lower_bound : float | None
            Lower bound for force offset.
        force_offset_upper_bound : float | None
            Upper bound for force offset.
        """
        if segment == "increasing":
            try:
                _fit, fitted_forces, fit_params, fit_error = fit_model_to_data(
                    distances=self.increasing_distance,
                    forces=self.increasing_force,
                    model=pylake.ewlc_odijk_force,
                    lp_value=lp_value,
                    lp_lower_bound=lp_lower_bound,
                    lp_upper_bound=lp_upper_bound,
                    lc_value=lc_value,
                    force_offset_lower_bound=force_offset_lower_bound,
                    force_offset_upper_bound=force_offset_upper_bound,
                )
            except np.linalg.LinAlgError:
                print("Fit failed due to nonconvergence. Skipping.")
                return
            self.increasing_fit = FitResult(
                fitted_forces=fitted_forces,
                params=fit_params,
                fit_error=fit_error,
            )
        elif segment == "decreasing":
            try:
                _fit, fitted_forces, fit_params, fit_error = fit_model_to_data(
                    distances=self.decreasing_distance,
                    forces=self.decreasing_force,
                    model=pylake.ewlc_odijk_force,
                    lp_value=lp_value,
                    lp_lower_bound=lp_lower_bound,
                    lp_upper_bound=lp_upper_bound,
                    lc_value=lc_value,
                    force_offset_lower_bound=force_offset_lower_bound,
                    force_offset_upper_bound=force_offset_upper_bound,
                )
            except np.linalg.LinAlgError:
                print("Fit failed due to nonconvergence. Skipping.")
                return
            self.decreasing_fit = FitResult(
                fitted_forces=fitted_forces,
                params=fit_params,
                fit_error=fit_error,
            )
        elif segment == "both":
            try:
                _fit, fitted_forces, fit_params, fit_error = fit_model_to_data(
                    distances=self.distances_both,
                    forces=self.forces_both,
                    model=pylake.ewlc_odijk_force,
                    lp_value=lp_value,
                    lp_lower_bound=lp_lower_bound,
                    lp_upper_bound=lp_upper_bound,
                    lc_value=lc_value,
                    force_offset_lower_bound=force_offset_lower_bound,
                    force_offset_upper_bound=force_offset_upper_bound,
                )
            except np.linalg.LinAlgError:
                print("Fit failed due to nonconvergence. Skipping.")
                return
            self.fit_both = FitResult(
                fitted_forces=fitted_forces,
                params=fit_params,
                fit_error=fit_error,
            )


class OscillationCollection(MarkerAnalysisBaseModel):
    """A data object to hold many oscillations together as a dataset."""

    oscillations: dict[str, OscillationModel]

    def __repr__(self) -> str:
        num_oscillations = len(self.oscillations)
        sample_keys = list(self.oscillations.keys())[:10]
        tail = "..." if num_oscillations > 10 else ""
        return f"OscillationCollection(num_oscillations={num_oscillations}, sample_keys={sample_keys}{tail})"

    def __str__(self) -> str:
        """
        Print representation of the object.

        Returns
        -------
        str
            String representation of the object.
        """
        return self.__repr__()

    def _repr_pretty_(self, printer) -> None:
        """
        IPython pretty-printer (notebook / rich display).

        Parameters
        ----------
        printer : Any
            The IPython pretty-printer object.
        """
        printer.text(repr(self))

    # Mapping methods to allow dict-like access
    def __getitem__(self, key: str) -> OscillationModel:
        """
        Return the oscillation corresponding to the given key.

        Parameters
        ----------
        key : str
            The key of the oscillation to retrieve.

        Returns
        -------
        OscillationModel
            The oscillation corresponding to the given key.
        """
        return self.oscillations[key]

    def __iter__(self) -> Generator[tuple[str, OscillationModel], None, None]:
        """
        Return a generator over (key, OscillationModel) pairs.

        This is apparently the correct way to do it for pydantic BaseModel classes.

        Returns
        -------
        Generator[tuple[str, OscillationModel], None, None]
            A generator over (key, OscillationModel) pairs.
        """
        return (item for item in self.oscillations.items())

    def __len__(self) -> int:
        """
        Return the number of oscillations in the collection.

        Returns
        -------
        int
            The number of oscillations in the collection.
        """
        return len(self.oscillations)

    def __contains__(self, key: str) -> bool:
        """
        Check if the collection contains the given key.

        Parameters
        ----------
        key : str
            The key to check for.

        Returns
        -------
        bool
            True if the key is in the collection, False otherwise.
        """
        return key in self.oscillations

    def items(self):
        """
        Return the items of the oscillations dictionary.

        Returns
        -------
        ItemsView
            The items of the oscillations dictionary.
        """
        return self.oscillations.items()

    def keys(self):
        """
        Return the keys of the oscillations dictionary.

        Returns
        -------
        KeysView
            The keys of the oscillations dictionary.
        """
        return self.oscillations.keys()

    def values(self):
        """
        Return the values of the oscillations dictionary.

        Returns
        -------
        ValuesView
            The values of the oscillations dictionary.
        """
        return self.oscillations.values()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return the value for the given key if it exists, otherwise return the default value.

        Parameters
        ----------
        key : str
            The key to look for.
        default : Any, optional
            The default value to return if the key does not exist. Default is None.

        Returns
        -------
        Any
            The value for the given key, or the default value.
        """
        return self.oscillations.get(key, default)

    def save_fitting_parameters_to_csv_file(self, file_path: Path, segment: str) -> None:
        """
        Save the fitting parameters of the oscillations in the collection to a CSV file.

        Parameters
        ----------
        file_path : Path
            The path to the CSV file to save the data to.
        segment : str
            The segment to save, either "increasing" or "decreasing" or "both".
        """

        # format for csv: columns: oscillation_id, curve_id, marker_filename, lp_value ...
        data_to_save = []
        for oscillation_id, oscillation in self.oscillations.items():
            if segment == "increasing" and oscillation.increasing_fit is not None:
                fit_params = oscillation.increasing_fit.params
                data_to_save.append(
                    {
                        "oscillation_id": oscillation_id,
                        "curve_id": oscillation.curve_id,
                        "marker_filename": oscillation.marker_filename,
                        "segment": "increasing",
                        "lp_value": fit_params["fit/Lp"].value,
                        "lp_error": fit_params["fit/Lp"].stderr,
                        "lc_value": fit_params["fit/Lc"].value,
                        "lc_error": fit_params["fit/Lc"].stderr,
                        "st_value": fit_params["fit/St"].value,
                        "st_error": fit_params["fit/St"].stderr,
                        "force_offset_value": fit_params["fit/f_offset"].value,
                        "force_offset_error": fit_params["fit/f_offset"].stderr,
                        "kT_value": fit_params["kT"].value,
                        "kT_error": fit_params["kT"].stderr,
                        **oscillation.metadata,
                    }
                )
            elif segment == "decreasing" and oscillation.decreasing_fit is not None:
                fit_params = oscillation.decreasing_fit.params
                data_to_save.append(
                    {
                        "oscillation_id": oscillation_id,
                        "curve_id": oscillation.curve_id,
                        "marker_filename": oscillation.marker_filename,
                        "segment": "decreasing",
                        "lp_value": fit_params["fit/Lp"].value,
                        "lp_error": fit_params["fit/Lp"].stderr,
                        "lc_value": fit_params["fit/Lc"].value,
                        "lc_error": fit_params["fit/Lc"].stderr,
                        "st_value": fit_params["fit/St"].value,
                        "st_error": fit_params["fit/St"].stderr,
                        "force_offset_value": fit_params["fit/f_offset"].value,
                        "force_offset_error": fit_params["fit/f_offset"].stderr,
                        "kT_value": fit_params["kT"].value,
                        "kT_error": fit_params["kT"].stderr,
                        **oscillation.metadata,
                    }
                )
            elif segment == "both" and oscillation.fit_both is not None:
                fit_params = oscillation.fit_both.params
                data_to_save.append(
                    {
                        "oscillation_id": oscillation_id,
                        "curve_id": oscillation.curve_id,
                        "marker_filename": oscillation.marker_filename,
                        "segment": "both",
                        "lp_value": fit_params["fit/Lp"].value,
                        "lp_error": fit_params["fit/Lp"].stderr,
                        "lc_value": fit_params["fit/Lc"].value,
                        "lc_error": fit_params["fit/Lc"].stderr,
                        "st_value": fit_params["fit/St"].value,
                        "st_error": fit_params["fit/St"].stderr,
                        "force_offset_value": fit_params["fit/f_offset"].value,
                        "force_offset_error": fit_params["fit/f_offset"].stderr,
                        "kT_value": fit_params["kT"].value,
                        "kT_error": fit_params["kT"].stderr,
                        **oscillation.metadata,
                    }
                )

        # Create dataframe from the data
        df = pd.DataFrame(data_to_save)
        df.to_csv(file_path, index=False)

    # pylint: disable=too-many-branches
    def save_dataset_data_to_csv_file(
        self, file_path: Path, segment: str, fitted_or_measured: str = "measured"
    ) -> None:
        """
        Save the dataset data to a CSV file.

        Parameters
        ----------
        file_path : Path
            The path to the CSV file to save the data to.
        segment : str
            The segment to save, either "increasing" or "decreasing" or "both".
        fitted_or_measured : str, optional
            The type of data to save, either "measured" or "fitted".
        """

        data_to_save = {}
        for oscillation_id, oscillation in self.oscillations.items():
            if segment == "increasing":
                if fitted_or_measured == "measured":
                    data_to_save[f"{oscillation_id}_increasing_distances"] = oscillation.increasing_distance
                    data_to_save[f"{oscillation_id}_increasing_forces"] = oscillation.increasing_force
                elif fitted_or_measured == "fitted":
                    if oscillation.increasing_fit is not None:
                        data_to_save[f"{oscillation_id}_increasing_distances"] = oscillation.increasing_distance
                        data_to_save[f"{oscillation_id}_increasing_fitted_forces"] = (
                            oscillation.increasing_fit.fitted_forces
                        )
                    else:
                        print(
                            f"Skipping oscillation {oscillation_id} increasing segment fitted data, no fit available."
                        )
            elif segment == "decreasing":
                if fitted_or_measured == "measured":
                    data_to_save[f"{oscillation_id}_decreasing_distances"] = oscillation.decreasing_distance
                    data_to_save[f"{oscillation_id}_decreasing_forces"] = oscillation.decreasing_force
                elif fitted_or_measured == "fitted":
                    if oscillation.decreasing_fit is not None:
                        data_to_save[f"{oscillation_id}_decreasing_distances"] = oscillation.decreasing_distance
                        data_to_save[f"{oscillation_id}_decreasing_fitted_forces"] = (
                            oscillation.decreasing_fit.fitted_forces
                        )
                    else:
                        print(
                            f"Skipping oscillation {oscillation_id} decreasing segment fitted data, no fit available."
                        )
            elif segment == "both":
                if fitted_or_measured == "measured":
                    data_to_save[f"{oscillation_id}_both_distances"] = oscillation.distances_both
                    data_to_save[f"{oscillation_id}_both_forces"] = oscillation.forces_both
                elif fitted_or_measured == "fitted":
                    if oscillation.fit_both is not None:
                        data_to_save[f"{oscillation_id}_both_distances"] = oscillation.distances_both
                        data_to_save[f"{oscillation_id}_both_fitted_forces"] = oscillation.fit_both.fitted_forces
                    else:
                        print(f"Skipping oscillation {oscillation_id} both segment fitted data, no fit available.")

        # Create dataframe from the data, note that the columns are not of equal length
        df = create_df_from_uneven_data(data_dict=data_to_save)
        df.to_csv(file_path, index=False)

    def plot_all(
        self,
        increasing_segment: bool = True,
        decreasing_segment: bool = True,
        random_colours: bool = False,
    ) -> None:
        """
        Plot all oscillations in the dataset on a single figure.

        Parameters
        ----------
        increasing_segment : bool, optional
            Whether to plot the increasing segment.
        decreasing_segment : bool, optional
            Whether to plot the decreasing segment.
        random_colours : bool, optional
            Whether to use random colours for each oscillation.
        """
        for _oscillation_id, oscillation in self.oscillations.items():
            if random_colours:
                colour = np.random.choice(PALETTE)
                oscillation.plot(
                    show=False,
                    increasing_segment=increasing_segment,
                    decreasing_segment=decreasing_segment,
                    increasing_colour=colour,
                    decreasing_colour=colour,
                )
            else:
                oscillation.plot(
                    show=False,
                    increasing_segment=increasing_segment,
                    decreasing_segment=decreasing_segment,
                )
        plt.show()

    def fit_individual_model_to_each(
        self,
        segment: str,
        lp_value: float | None = None,
        lp_lower_bound: float | None = None,
        lp_upper_bound: float | None = None,
        lc_value: float | None = None,
        force_offset_lower_bound: float | None = None,
        force_offset_upper_bound: float | None = None,
    ) -> None:
        """
        Fit the specified segment of all oscillations in the dataset using one model per oscillation.

        Parameters
        ----------
        segment : str
            The segment to fit, either "increasing" or "decreasing".
        lp_value : float | None
            Initial guess for persistence length.
        lp_lower_bound : float | None
            Lower bound for persistence length.
        lp_upper_bound : float | None
            Upper bound for persistence length.
        lc_value : float | None
            Initial guess for contour length.
        force_offset_lower_bound : float | None
            Lower bound for force offset.
        force_offset_upper_bound : float | None
            Upper bound for force offset.
        """
        for _oscillation_id, oscillation in self.oscillations.items():
            oscillation.fit_model(
                segment=segment,
                lp_value=lp_value,
                lp_lower_bound=lp_lower_bound,
                lp_upper_bound=lp_upper_bound,
                lc_value=lc_value,
                force_offset_lower_bound=force_offset_lower_bound,
                force_offset_upper_bound=force_offset_upper_bound,
            )


class ReducedFDCurveModel(MarkerAnalysisBaseModel):
    """
    A data object to hold reduced force-distance curve data.

    Attributes
    ----------
    filename: str
        The name of the marker file that the curve was loaded from.
    id: str
        The id of the fd curve.
    all_forces: npt.NDArray[np.float64]
        The force data of the fd curve, in pico-newtons.
    all_distances: npt.NDArray[np.float64]
        The distance data of the fd curve, in micrometres.
    oscillations: dict[str,OscillationModel] | None
        A dictionary of oscillations found in the fd curve.
    include_in_processing: bool
        Whether to include this fd curve in further processing.
    metadata: dict[str, str | float | int | None]
        A dictionary of metadata extracted from the fd curve name.
    """

    filename: str
    id: str
    all_forces: npt.NDArray[np.float64]
    all_distances: npt.NDArray[np.float64]
    oscillations: dict[str, OscillationModel] | None = None
    include_in_processing: bool = True
    metadata: dict[str, str | float | int | None] = {}


class ReducedMarkerModel(MarkerAnalysisBaseModel):
    """A data object to hold marker data in a reduced form."""

    file_path: Path
    file_name: str | None = None
    include_in_processing: bool = True
    fd_curves: dict[str, ReducedFDCurveModel]

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        verbose: bool = False,
        metadata_regex: str | None = None,
        force_maximum: float | None = None,
        distance_minimum: float | None = None,
    ) -> "ReducedMarkerModel":
        """
        Factory method to create ReducedMarkerModel from a file.

        Parameters
        ----------
        file_path : Path
            The path to the file to load the data from.
        verbose : bool, optional
            If True, print additional information during loading. Default is False.
        metadata_regex : str | None, optional
            A regex pattern to extract metadata from fd curve names. Default is None.
        force_maximum : float | None, optional
            Maximum force to allow in the data. Datapoints above this value will be excluded. Default: None.
        distance_minimum : float | None, optional
            Minimum distance to allow in the data. Datapoints below this value will be excluded. Default: None.

        Returns
        -------
        ReducedMarkerModel
            An instance of ReducedMarkerModel containing the loaded data.
        """

        file_name = file_path.name
        # Note that lumicks does not seem to close files after reading them, it will need to be closed manually.
        # This can be done with lumicks_file.h5.close().
        lumicks_file = pylake.File(filename=file_path)
        fd_curves = cls.load_fd_curves(
            filename=file_name,
            pylake_file_fd_curves=lumicks_file.fdcurves,
            verbose=verbose,
            metadata_regex=metadata_regex,
            force_maximum=force_maximum,
            distance_minimum=distance_minimum,
        )
        # close the lumicks file
        lumicks_file.h5.close()
        return cls(
            file_path=file_path,
            file_name=file_name,
            fd_curves=fd_curves,
            include_in_processing=True,
        )

    @staticmethod
    def get_file_metadata_matt(filename: str) -> dict[str, str | float | int | None]:
        """
        Obtain file metadata from the filename.

        Parameters
        ----------
        filename : str
            The name of the file to extract metadata from.

        Returns
        -------
        dict[str, str | float | int | None]
            A dictionary containing the extracted metadata.
        """
        # grab tel reps
        tel_reps_match = re.search(r"Tel(\d+)", filename)
        if tel_reps_match:
            tel_reps = int(tel_reps_match.group(1))
        else:
            tel_reps = None
        # grab concentration, assumed to be the float before a "nM".
        concentration_match = re.search(r"(\d+\.?\d*)nM", filename)
        if concentration_match:
            concentration = float(concentration_match.group(1))
        else:
            concentration = None
        # grab protein name, assumed to be before the string "Marker X"
        protein_name_match = re.search(r" (\w+)(?= Marker \d+)", filename)
        if protein_name_match:
            protein_name = protein_name_match.group(1)
        else:
            protein_name = None

        return {
            "telreps": tel_reps,
            "protein_name": protein_name,
            "concentration": concentration,
        }

    # pylint: disable=too-many-locals
    @staticmethod
    def load_fd_curves(
        filename: str,
        pylake_file_fd_curves: dict[str, pylake.file.FdCurve],
        verbose: bool = False,
        metadata_regex: str | None = None,
        force_maximum: float | None = None,
        distance_minimum: float | None = None,
    ) -> dict[str, ReducedFDCurveModel]:
        """
        Load the force-distance curves from the file.

        Parameters
        ----------
        filename : str
            The name of the file to load the curves from.
        pylake_file_fd_curves : dict[str, pylake.file.FdCurve]
            A dictionary of force-distance curves.
        verbose : bool, optional
            If True, print additional information about the curves being loaded. Default is False.
        metadata_regex : str | None, optional
            A regex pattern to extract metadata from fd curve names. Default is None.
        force_maximum : float | None, optional
            Maximum force to allow in the data. Datapoints above this value will be excluded. Default: None.
        distance_minimum : float | None, optional
            Minimum distance to allow in the data. Datapoints below this value will be excluded. Default: None.

        Returns
        -------
        dict[str, ReducedFDCurve]
            A dictionary of reduced force-distance curves, where the keys are the curve IDs and the values are
            instances of ReducedFDCurve containing the force and distance data, as well as the oscillations found in
            the curves.
        """
        fd_curves: dict[str, ReducedFDCurveModel] = {}
        for curve_id, curve_data in pylake_file_fd_curves.items():
            if verbose:
                print(f"Loading curve {curve_id} with {len(curve_data.d.data)} data points")
            curve_name = curve_data.name
            metadata = extract_metadata_from_fd_curve_name_with_regex(
                curve_name=curve_name,
                regex_pattern=metadata_regex,
            )
            force_data: npt.NDArray[np.float64] = curve_data.f.data
            distance_data: npt.NDArray[np.float64] = np.asarray(curve_data.d.data, dtype=np.float64)

            # calculate the starting distance to identify flat regions
            flat_distance_um = ReducedMarkerModel._calculate_fd_curve_starting_distance(
                distance_data=distance_data,
                curve_id=curve_id,
                filename=filename,
            )
            if flat_distance_um is None:
                # skip this curve
                if verbose:
                    print(f"Skipping curve {curve_id} due to inability to determine starting distance.")
                continue

            # trim non-flat regions from the ends of the curve
            distance_data_trimmed, force_data_trimmed, flat_regions_bool_trimmed = (
                ReducedMarkerModel._trim_non_flat_regions_from_ends_of_curve(
                    distance_data=distance_data,
                    force_data=force_data,
                    flat_distance_um=flat_distance_um,
                )
            )

            # extract oscillations from the trimmed data
            oscillations = ReducedMarkerModel._extract_oscillations_from_trimmed_data(
                distance_data=distance_data_trimmed,
                force_data=force_data_trimmed,
                flat_regions_bool=flat_regions_bool_trimmed,
                curve_id=curve_id,
                marker_filename=filename,
                metadata=metadata,
                force_maximum=force_maximum,
                distance_minimum=distance_minimum,
            )

            fd_curve = ReducedFDCurveModel(
                filename=filename,
                id=curve_id,
                all_forces=force_data_trimmed,
                all_distances=distance_data_trimmed,
                oscillations=oscillations,
                metadata=metadata,
            )
            fd_curves[curve_name] = fd_curve
        return fd_curves

    @staticmethod
    def _calculate_fd_curve_starting_distance(
        distance_data: npt.NDArray[np.float64], curve_id: str = "undefined", filename: str = "undefined"
    ) -> np.float64 | None:
        """
        Calculate the starting distance of a force-distance curve, for being able to identify stationary regions.

        Parameters
        ----------
        distance_data : npt.NDArray[np.float64]
            The distance data of the force-distance curve.
        curve_id : str, optional
            The ID of the curve, used for logging purposes. Default is "undefined".
        filename : str, optional
            The name of the file containing the curve, used for logging purposes. Default is "undefined".

        Returns
        -------
        np.float64 | None
            The starting distance of the curve, or None if it could not be determined.
        """

        # Determine starting distance to be the first peak in frequency of the distance data
        bin_size_um = 0.1
        bin_edges = np.arange(np.min(distance_data), np.max(distance_data) + bin_size_um, bin_size_um)
        hist, _ = np.histogram(distance_data, bins=bin_edges)

        # find the largest peak in the histogram
        peak_index = np.argmax(hist)
        # get the midpoint of the bin
        base_distance: np.float64 = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2

        # check that the peak is strong, as in that the peak contains a lot more counts than the other bins
        # criteria: peak should be at least 2x the next highest bin
        next_highest_bin = np.partition(hist, kth=-2)[-2]
        peak_strength = hist[peak_index] / next_highest_bin if next_highest_bin > 0 else 0
        peak_strength_threshold = 2.0
        if peak_strength < peak_strength_threshold:
            print(
                f"Warning: Baseline distance peak strength for curve {curve_id} in file {filename} is low"
                f"({peak_strength:.2f}<{peak_strength_threshold})."
            )
            return None
        return base_distance

    @staticmethod
    def _trim_non_flat_regions_from_ends_of_curve(
        distance_data: npt.NDArray[np.float64],
        force_data: npt.NDArray[np.float64],
        flat_distance_um: np.float64,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """
        Trim non-flat regions from the ends of a force-distance curve.

        Parameters
        ----------
        distance_data : npt.NDArray[np.float64]
            The distance data of the force-distance curve.
        force_data : npt.NDArray[np.float64]
            The force data of the force-distance curve.
        flat_distance_um : np.float64
            The distance value that represents the flat region.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]
            A tuple containing the trimmed distance data, trimmed force data, and a boolean array indicating flat
            regions.
        """
        flat_distance_tolerance_um = 0.1
        flat_regions_bool: npt.NDArray[np.bool_] = np.abs(distance_data - flat_distance_um) < flat_distance_tolerance_um
        flat_regions: list[tuple[int, int]] = []
        current_flat_region_start: int | None = None
        for index, is_flat in enumerate(flat_regions_bool):
            if is_flat and current_flat_region_start is None:
                current_flat_region_start = index
            elif not is_flat and current_flat_region_start is not None:
                flat_regions.append((current_flat_region_start, index - 1))
                current_flat_region_start = None
        if current_flat_region_start is not None:
            flat_regions.append((current_flat_region_start, len(flat_regions_bool) - 1))

        # eliminate non-flat regions at the start and end of the curve
        if flat_regions:
            if flat_regions[0][0] > 0:
                # cut the array to start at the first flat region
                distance_data_trimmed = distance_data[flat_regions[0][0] :]
                force_data_trimmed = force_data[flat_regions[0][0] :]
                flat_regions_bool_trimmed = flat_regions_bool[flat_regions[0][0] :]
            else:
                distance_data_trimmed = distance_data
                force_data_trimmed = force_data
                flat_regions_bool_trimmed = flat_regions_bool
            if flat_regions[-1][1] < len(distance_data_trimmed) - 1:
                distance_data_trimmed = distance_data_trimmed[: flat_regions[-1][1] + 1]
                force_data_trimmed = force_data_trimmed[: flat_regions[-1][1] + 1]
                flat_regions_bool_trimmed = flat_regions_bool_trimmed[: flat_regions[-1][1] + 1]
        else:
            distance_data_trimmed = distance_data
            force_data_trimmed = force_data
            flat_regions_bool_trimmed = flat_regions_bool

        return distance_data_trimmed, force_data_trimmed, flat_regions_bool_trimmed

    @staticmethod
    # pylint: disable=too-many-locals
    def _extract_oscillations_from_trimmed_data(
        distance_data: npt.NDArray[np.float64],
        force_data: npt.NDArray[np.float64],
        flat_regions_bool: npt.NDArray[np.bool_],
        curve_id: str = "undefined",
        marker_filename: str = "undefined",
        metadata: dict[str, int | float | str | None] | None = None,
        force_maximum: float | None = None,
        distance_minimum: float | None = None,
    ) -> dict[str, OscillationModel]:
        """
        Extract oscillations from trimmed force-distance curve data.

        Parameters
        ----------
        distance_data : npt.NDArray[np.float64]
            The distance data of the force-distance curve.
        force_data : npt.NDArray[np.float64]
            The force data of the force-distance curve.
        flat_regions_bool : npt.NDArray[np.bool_]
            A boolean array indicating flat regions in the distance data.
        curve_id : str, optional
            The ID of the curve, used for logging purposes. Default is "undefined".
        marker_filename : str, optional
            The name of the marker file containing the curve, used for logging purposes. Default is "undefined".
        metadata : dict[str, int | float | str | None] | None, optional
            A dictionary of metadata to associate with the oscillations. Default is None.
        force_maximum : float | None, optional
            Maximum force to allow in the data. Datapoints above this value will be excluded from oscillations.
            Default: None.
        distance_minimum : float | None, optional
            Minimum distance to allow in the data. Datapoints below this value will be excluded from oscillations.
            Default: None.

        Returns
        -------
        dict[str, OscillationModel]
            A dictionary of OscillationModel instances representing the extracted oscillations.

        Notes
        -----
        There must be no non-flat regions at the start or end of the data.
        """
        if metadata is None:
            metadata = {}
        non_flat_regions_bool = ~flat_regions_bool
        labelled_non_flat_regions: npt.NDArray[np.int32] = label(non_flat_regions_bool)
        oscillations: dict[str, OscillationModel] = {}
        oscillation_count = 0
        for label_index in range(1, np.max(labelled_non_flat_regions) + 1):
            # get indexes of the current non-flat region
            non_flat_region_indexes = np.where(labelled_non_flat_regions == label_index)
            non_flat_region_start_index = non_flat_region_indexes[0][0]
            non_flat_region_end_index = non_flat_region_indexes[0][-1]
            # get the maximum distance index
            non_flat_region_distances = distance_data[non_flat_region_start_index : non_flat_region_end_index + 1]
            non_flat_region_local_maximum_distance_index = np.argmax(non_flat_region_distances)
            non_flat_region_global_maximum_distance_index = (
                non_flat_region_start_index + non_flat_region_local_maximum_distance_index
            )
            # set the increasing segment to the left of the maximum index
            increasing_segment_start_index = non_flat_region_start_index
            # keep the largest distsance value in the increasing segment
            increasing_segment_end_index = non_flat_region_global_maximum_distance_index + 1
            # set the decreasing segment to the right of the maximum distance index
            decreasing_segment_start_index = non_flat_region_global_maximum_distance_index + 1
            decreasing_segment_end_index = non_flat_region_end_index + 1
            # get the force data for the increasing and decresaing segments
            increasing_force = force_data[increasing_segment_start_index:increasing_segment_end_index]
            increasing_distance = distance_data[increasing_segment_start_index:increasing_segment_end_index]
            decreasing_force = force_data[decreasing_segment_start_index:decreasing_segment_end_index]
            decreasing_distance = distance_data[decreasing_segment_start_index:decreasing_segment_end_index]
            oscillation = OscillationModel(
                id=str(oscillation_count),
                curve_id=curve_id,
                marker_filename=marker_filename,
                metadata=metadata,
                increasing_force=increasing_force,
                increasing_distance=increasing_distance,
                decreasing_force=decreasing_force,
                decreasing_distance=decreasing_distance,
                force_maximum=force_maximum,
                distance_minimum=distance_minimum,
            )
            oscillations[str(oscillation_count)] = oscillation
            oscillation_count += 1
        return oscillations
