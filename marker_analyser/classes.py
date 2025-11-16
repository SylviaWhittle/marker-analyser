"""Classes and dataclasses for data storage and handling."""

from pathlib import Path

import re
from typing import Any
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# lumicks doesn't provide type stubs, so have ignored mypy warnings for missing imports in the mypy arguments in
# settings.
from lumicks import pylake
from skimage.morphology import label

from marker_analyser.fitting import fit_model_to_data


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


class OscillationModel(MarkerAnalysisBaseModel):
    """A data object to hold oscillation data."""

    increasing_force: npt.NDArray[np.float64]
    increasing_distance: npt.NDArray[np.float64]
    decreasing_force: npt.NDArray[np.float64]
    decreasing_distance: npt.NDArray[np.float64]
    force_peaks: list[ForcePeakModel] | None = None
    num_peaks: int | None = None
    increasing_fit: FitResult | None = None
    decreasing_fit: FitResult | None = None

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

    def plot(self, increasing_colour: str = "tab:blue", decreasing_colour: str = "tab:green") -> None:
        """
        Plot the oscillation's force-distance data.

        Parameters
        ----------
        increasing_colour : str, optional
            The colour to use for the increasing segment.
        decreasing_colour : str, optional
            The colour to use for the decreasing segment.
        """

        plt.plot(self.increasing_distance, self.increasing_force, color=increasing_colour, alpha=0.5)
        plt.plot(self.decreasing_distance, self.decreasing_force, color=decreasing_colour, alpha=0.5)
        if self.increasing_fit is not None:
            plt.plot(self.increasing_distance, self.increasing_fit.fitted_forces, color=increasing_colour, alpha=1)
        if self.decreasing_fit is not None:
            plt.plot(self.decreasing_distance, self.decreasing_fit.fitted_forces, color=decreasing_colour, alpha=1)
        plt.xlabel("Distance (um)")
        plt.ylabel("Force (pN)")
        plt.title("")
        plt.show()

    def fit_model(
        self,
        segment: str,
        lp_value: float,
        lp_lower_bound: float,
        lp_upper_bound: float,
        lc_value: float,
        force_offset_lower_bound: float,
        force_offset_upper_bound: float,
    ) -> None:
        """
        Fit the specified segment of the oscillation.

        Parameters
        ----------
        segment : str
            The segment to fit, either "increasing" or "decreasing".
        lp_value : float
            Initial guess for persistence length.
        lp_lower_bound : float
            Lower bound for persistence length.
        lp_upper_bound : float
            Upper bound for persistence length.
        lc_value : float
            Initial guess for contour length.
        force_offset_lower_bound : float
            Lower bound for force offset.
        force_offset_upper_bound : float
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
    oscillations: list[OscillationModel] | None
        A list of oscillations found in the fd curve.
    include_in_processing: bool
        Whether to include this fd curve in further processing.
    """

    filename: str
    id: str
    all_forces: npt.NDArray[np.float64]
    all_distances: npt.NDArray[np.float64]
    oscillations: list[OscillationModel] | None = None
    include_in_processing: bool = True


class MattMarkerMetadataModel(MarkerAnalysisBaseModel):
    """A data object to hold metadata for Matt's markers."""

    protein_name: str | None
    concentration_nM: float | None
    telereps: int | None


class ReducedMarkerModel(MarkerAnalysisBaseModel):
    """A data object to hold marker data in a reduced form."""

    file_path: Path
    file_name: str | None = None
    telreps: str | int | None = None
    protein_name: str | None = None
    concentration: float | str | None = None
    include_in_processing: bool = True
    fd_curves: dict[str, ReducedFDCurveModel]

    @classmethod
    def from_file(cls, file_path: Path, verbose: bool = False) -> "ReducedMarkerModel":
        """
        Factory method to create ReducedMarkerModel from a file.

        Parameters
        ----------
        file_path : Path
            The path to the file to load the data from.
        verbose : bool, optional
            If True, print additional information during loading. Default is False.

        Returns
        -------
        ReducedMarkerModel
            An instance of ReducedMarkerModel containing the loaded data.
        """

        file_name = file_path.name
        # Note that lumicks does not seem to close files after reading them, it will need to be closed manually.
        # This can be done with lumicks_file.h5.close().
        lumicks_file = pylake.File(filename=file_path)
        metadata = cls.get_file_metadata_matt(filename=file_name)
        telreps = metadata.telereps
        protein_name = metadata.protein_name
        concentration = metadata.concentration_nM
        fd_curves = cls.load_fd_curves(
            filename=file_name,
            pylake_file_fd_curves=lumicks_file.fdcurves,
            verbose=verbose,
        )
        # close the lumicks file
        lumicks_file.h5.close()
        return cls(
            file_path=file_path,
            file_name=file_name,
            telreps=telreps,
            protein_name=protein_name,
            concentration=concentration,
            fd_curves=fd_curves,
        )

    @staticmethod
    def get_file_metadata_matt(filename: str) -> MattMarkerMetadataModel:
        """
        Obtain file metadata from the filename.

        Parameters
        ----------
        filename : str
            The name of the file to extract metadata from.

        Returns
        -------
        MattMarkerMetadataModel
            An object containing the extracted metadata, or None if metadata could not be extracted.
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

        return MattMarkerMetadataModel(
            protein_name=protein_name,
            concentration_nM=concentration,
            telereps=tel_reps,
        )

    @staticmethod
    def load_fd_curves(
        filename: str,
        pylake_file_fd_curves: dict[str, pylake.file.FdCurve],
        verbose: bool = False,
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
            )

            fd_curve = ReducedFDCurveModel(
                filename=filename,
                id=curve_id,
                all_forces=force_data_trimmed,
                all_distances=distance_data_trimmed,
                oscillations=oscillations,
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
                f"Warning: Peak strength for curve {curve_id} in file {filename} is low"
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
    ) -> list[OscillationModel]:
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

        Returns
        -------
        list[OscillationModel]
            A list of OscillationModel instances representing the extracted oscillations.

        Notes
        -----
        There must be no non-flat regions at the start or end of the data.
        """
        non_flat_regions_bool = ~flat_regions_bool
        labelled_non_flat_regions: npt.NDArray[np.int32] = label(non_flat_regions_bool)
        oscillations: list[OscillationModel] = []
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
                increasing_force=increasing_force,
                increasing_distance=increasing_distance,
                decreasing_force=decreasing_force,
                decreasing_distance=decreasing_distance,
            )
            oscillations.append(oscillation)
        return oscillations
