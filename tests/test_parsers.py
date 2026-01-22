"""Tests for parsers.py"""

import pytest

from marker_analyser.parsers import extract_metadata_from_fd_curve_name_with_regex


@pytest.mark.parametrize(
    ("curve_name", "regex_pattern", "expected_metadata"),
    [
        (
            "CuOda_10uM_M4_3",
            r"^"
            r"(?P<drug>[A-Za-z][\w\-]*)_"
            r"(?P<conc>\d+(?:\.\d+)?(?:pM|nM|uM|μM|mM))_"
            r"[Mm](?P<marker>\d+)_"
            r"(?P<repeat>\d+)"
            r"$",
            {
                "drug": "CuOda",
                "conc": "10uM",
                "marker": 4,
                "repeat": 3,
            },
        )
    ],
)
def test_extract_metadata_from_fd_curve_name_with_regex(
    curve_name: str,
    regex_pattern: str,
    expected_metadata: dict[str, str | float | int | None],
):
    """Test the test_extract_metadata_from_fd_curve_name_with_regex function."""
    metadata = extract_metadata_from_fd_curve_name_with_regex(curve_name, regex_pattern)
    assert metadata == expected_metadata
    assert False
