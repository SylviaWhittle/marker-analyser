"""Scripts for input/output operations."""

from pathlib import Path

from marker_analyser.classes import ReducedFDCurveModel, ReducedMarkerModel, OscillationCollection


def load_fd_curves_from_directory(directory_path: Path | str) -> dict[str, ReducedFDCurveModel]:
    """
    Load all FDCurves from a given directory.

    Parameters
    ----------
    directory_path : Path | str
        Path to the directory containing marker files.

    Returns
    -------
    dict[str, ReducedFDCurveModel]
        A dictionary of fd curves.
    """
    directory = Path(directory_path)
    fd_curves = {}
    file_paths = list(directory.glob("*.h5"))
    print(f"Found {len(file_paths)} marker files in {directory_path}")
    for file_path in file_paths:
        marker = ReducedMarkerModel.from_file(file_path)
        print(f"Loaded marker '{marker.file_name}' with {len(marker.fd_curves)} FDCurves")
        # Combine fdcurves into the dictionary, avoiding overwriting keys
        for curve_name, fd_curve in marker.fd_curves.items():
            if curve_name in fd_curves:
                print(f"Warning: Duplicate curve name '{curve_name}' found in {file_path}. Skipping this curve.")
                continue
            fd_curves[curve_name] = fd_curve
    return fd_curves


def load_oscillations_from_directory(
    directory_path: Path | str, exclude_curve_ids: list[str] | None = None, metadata_regex: str | None = None
) -> OscillationCollection:
    """
    Load all Oscillations from a given directory.

    Parameters
    ----------
    directory_path : Path | str
        Path to the directory containing marker files.
    exclude_curve_ids : list[str] | None, optional
        List of curve IDs to exclude, by default None.
    metadata_regex : str | None, optional
        Regex pattern to extract metadata from marker names, by default None.

    Returns
    -------
    OscillationCollection
        A collection of oscillations.
    """
    if metadata_regex is None:
        print("Warning: No metadata regex pattern provided. Metadata will not be extracted from marker names.")
    directory = Path(directory_path)
    oscillations = {}
    file_paths = list(directory.glob("*.h5"))
    num_excluded_curves_exclude_list = 0
    num_excluded_curves_duplicate = 0
    print(f"Found {len(file_paths)} marker files in {directory_path}")
    for file_path in file_paths:
        marker = ReducedMarkerModel.from_file(file_path, metadata_regex=metadata_regex)
        # Iterate over fd curves
        for curve_id, fd_curve in marker.fd_curves.items():
            if exclude_curve_ids is not None and curve_id in exclude_curve_ids:
                num_excluded_curves_exclude_list += 1
                continue
            if not fd_curve.oscillations:
                print(f"Warning: No oscillations found in curve '{curve_id}' from file '{file_path}'. Skipping.")
                continue
            for oscillation_id, oscillation in fd_curve.oscillations.items():
                unique_id = f"curve_{curve_id}_oscillation_{oscillation_id}"
                if unique_id in oscillations:
                    print(
                        f"Warning: Duplicate oscillation id '{unique_id}' found in {file_path}."
                        f"Skipping this oscillation."
                    )
                    num_excluded_curves_duplicate += 1
                    continue
                oscillations[unique_id] = oscillation
    print(f"Loaded {len(oscillations)} oscillations from {len(file_paths)} files in {directory_path}.")
    if num_excluded_curves_exclude_list > 0:
        print(f"Excluded {num_excluded_curves_exclude_list} curves based on the exclude list.")
    if num_excluded_curves_duplicate > 0:
        print(f"Excluded {num_excluded_curves_duplicate} duplicate oscillations.")
    return OscillationCollection(oscillations=oscillations)
