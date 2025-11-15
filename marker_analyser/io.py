"""Scripts for input/output operations."""

from pathlib import Path

from marker_analyser.classes import ReducedFDCurveModel, ReducedMarkerModel


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
    print(f"file_paths: {file_paths}")
    print(f"Found {len(file_paths)} marker files in {directory_path}")
    for file_path in file_paths:
        print(f"Loading marker from {file_path}")
        marker = ReducedMarkerModel.from_file(file_path)
        print(f"Loaded marker '{marker.file_name}' with {len(marker.fd_curves)} FDCurves")
        # Combine fdcurves into the dictionary, avoiding overwriting keys
        for curve_name, fd_curve in marker.fd_curves.items():
            if curve_name in fd_curves:
                print(f"Warning: Duplicate curve name '{curve_name}' found in {file_path}. Skipping this curve.")
                continue
            fd_curves[curve_name] = fd_curve
    return fd_curves
