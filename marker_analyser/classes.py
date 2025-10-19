"""classes and dataclasses for data storage and handling"""
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from lumicks import pylake


class ReducedMarker:
class ForcePeakModel(BaseModel):
    """A data object to hold force peak data."""

    distance: float
    force: float
    index: int


class OscillationModel(BaseModel):
    """A data object to hold oscillation data."""

    increasing_force: npt.NDArray[np.float64]
    increasing_distance: npt.NDArray[np.float64]
    decreasing_force: npt.NDArray[np.float64]
    decreasing_distance: npt.NDArray[np.float64]
    force_peaks: list[ForcePeakModel] | None = None
    num_peaks: int | None = None


class ReducedFDCurveModel(BaseModel):
    """A data object to hold reduced force-distance curve data."""

    filename: str
    curve_id: str
    all_forces: npt.NDArray[np.float64]
    all_distances: npt.NDArray[np.float64]
    oscillations: list[OscillationModel] | None = None
    include_in_processing: bool = True


class MattMarkerMetadataModel(BaseModel):
    """A data object to hold metadata for Matt's markers."""

    protein_name: str
    concentration_nM: float
    telereps: int
        # Note that lumicks does not seem to close files after reading them, it will need to be closed manually.
        # This can be done with lumicks_file.h5.close().
        lumicks_file = pylake.File(self.file_path)
        self.metadata = self.get_file_metadata(self.file_path)
        self.telreps = self.metadata["telreps"]
        self.protein_name = self.metadata["protein_name"]
        self.concentration = self.metadata["concentration"]
        self.fd_curves = self.load_fd_curves(
            filename=self.file_name, fdcurves=lumicks_file.fdcurves, verbose=verbose, plotting=plotting
        )
        self.include_in_processing = True
        # close the lumicks file
        lumicks_file.h5.close()
    
    @staticmethod
    def get_file_metadata(filename: str) -> dict[str, str | int | float]:
        """
        Obtain file metadata from the filename.
        
        Parameters
        ----------
        filename : str
            The name of the file to extract metadata from.
        
        Returns
        -------
        dict[str, str | int | float]
            A dictionary containing the metadata extracted from the filename.
        """
        metadata: dict[str, str | int | float] = {}
        # grab tel reps
        tel_reps_match = re.search(r"Tel(\d+)", filename)
        if tel_reps_match:
            tel_reps = int(tel_reps_match.group(1))
        else:
            raise ValueError(f"Could not find telereps regex: Tel(\\d+) in file name {filename}")
        metadata["telereps"] = tel_reps
        # grab concentration, assumed to be the float before a "nM".
        concentration_match = re.search(r"(\d+\.?\d*)nM", filename)
        if concentration_match:
            concentration: float | str = float(concentration_match.group(1))
        else:
            concentration = "NA"
        metadata["concentration_nM"] = concentration
        # grab protein name, assumed to be before the string "Marker X"
        protein_name_match = re.search(r" (\w+)(?= Marker \d+)", filename)
        if protein_name_match:
            protein_name = protein_name_match.group(1)
        else:
            raise ValueError(f"Could not find protein name in file name {filename}")
        metadata["protein_name"] = protein_name

        return metadata


