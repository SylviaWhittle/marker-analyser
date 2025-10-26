"""classes and dataclasses for data storage and handling"""

from pathlib import Path

import re
from pydantic import BaseModel, ConfigDict
import numpy as np
import numpy.typing as npt

# lumicks doesn't provide type stubs, so have ignored mypy warnings for missing imports in the mypy arguments in
# settings.
from lumicks import pylake
from skimage.morphology import label


class MarkerAnalysisBaseModel(BaseModel):
    """Data object to hold settings for Models used in the project."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ForcePeakModel(MarkerAnalysisBaseModel):
    """A data object to hold force peak data."""

    distance: float
    force: float
    index: int


class OscillationModel(MarkerAnalysisBaseModel):
    """A data object to hold oscillation data."""

    increasing_force: npt.NDArray[np.float64]
    increasing_distance: npt.NDArray[np.float64]
    decreasing_force: npt.NDArray[np.float64]
    decreasing_distance: npt.NDArray[np.float64]
    force_peaks: list[ForcePeakModel] | None = None
    num_peaks: int | None = None


class ReducedFDCurveModel(MarkerAnalysisBaseModel):
    """A data object to hold reduced force-distance curve data."""

    filename: str
    curve_id: str
    all_forces: npt.NDArray[np.float64]
    all_distances: npt.NDArray[np.float64]
    oscillations: list[OscillationModel] | None = None
    include_in_processing: bool = True


class MattMarkerMetadataModel(MarkerAnalysisBaseModel):
    """A data object to hold metadata for Matt's markers."""

    protein_name: str
    concentration_nM: float
    telereps: int


class ReducedMarkerModel(MarkerAnalysisBaseModel):
    """A data object to hold marker data in a reduced form."""

    file_path: Path
    file_name: str | None = None
    telreps: str | int | None = None
    protein_name: str | None = None
    concentration: float | str | None = None
    include_in_processing: bool = True
    fd_curves: dict[str, ReducedFDCurveModel] | None = None

    @classmethod
    def from_file(cls, file_path: Path, verbose: bool = False) -> "ReducedMarkerModel":
        """Factory method to create ReducedMarkerModel from a file."""
        file_name = file_path.name
        # Note that lumicks does not seem to close files after reading them, it will need to be closed manually.
        # This can be done with lumicks_file.h5.close().
        lumicks_file = pylake.File(filename=file_path)
        metadata = cls.get_file_metadata(filename=file_name)
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
    def get_file_metadata(filename: str) -> MattMarkerMetadataModel:
        """
        Obtain file metadata from the filename.

        Parameters
        ----------
        filename : str
            The name of the file to extract metadata from.

        Returns
        -------
        MattMarkerMetadataModel
            An object containing the extracted metadata.
        """
        # grab tel reps
        tel_reps_match = re.search(r"Tel(\d+)", filename)
        if tel_reps_match:
            tel_reps = int(tel_reps_match.group(1))
        else:
            raise ValueError(f"Could not find telereps regex: Tel(\\d+) in file name {filename}")
        # grab concentration, assumed to be the float before a "nM".
        concentration_match = re.search(r"(\d+\.?\d*)nM", filename)
        if concentration_match:
            concentration = float(concentration_match.group(1))
        else:
            concentration = -1.0
        # grab protein name, assumed to be before the string "Marker X"
        protein_name_match = re.search(r" (\w+)(?= Marker \d+)", filename)
        if protein_name_match:
            protein_name = protein_name_match.group(1)
        else:
            raise ValueError(f"Could not find protein name in file name {filename}")

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
                curve_id=curve_id,
                all_forces=force_data_trimmed,
                all_distances=distance_data_trimmed,
                oscillations=oscillations,
            )
            fd_curves[curve_id] = fd_curve
        return fd_curves

    @staticmethod
    def _calculate_fd_curve_starting_distance(
        distance_data: npt.NDArray[np.float64], curve_id: str = "undefined", filename: str = "undefined"
    ) -> np.float64 | None:
        """Calculate the starting distance of a force-distance curve, for being able to identify stationary regions."""
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
        """Trim non-flat regions from the ends of a force-distance curve."""
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
        """Extract oscillations from trimmed force-distance curve data.

        Note: There must be no non-flat regions at the start or end of the data.
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
