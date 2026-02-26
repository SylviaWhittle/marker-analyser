"""Scripts for parsing values and extracting data from strings etc."""

import re


def attempt_coerce_string_to_numbers(value: str | int | float) -> str | float | int:
    """
    Attempt to coerce a string to a number.

    Parameters
    ----------
    value : str | int | float
        The string to coerce.

    Returns
    -------
    str | float | int
        The coerced value, or the original string if coercion failed.
    """
    if not isinstance(value, str):
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def extract_metadata_from_fd_curve_name_with_regex(
    curve_name: str,
    regex_pattern: str | None,
) -> dict[str, str | float | int | None]:
    """
    Extract metadata from the fd curve name using a regex pattern.

    Parameters
    ----------
    curve_name : str
        The name of the fd curve to extract metadata from.
    regex_pattern : str
        The regex pattern to use for extraction.

    Returns
    -------
    dict[str, str | float | int | None]
        A dictionary containing the extracted metadata.

    Examples
    --------
    >>> curve_name = "CuOda_10uM_M4_3"
    >>> regex_pattern = r"^"
        r"(?P<drug>[A-Za-z][\\w\\-]*)_"  # drug name
        r"(?P<conc>\\d+(?:\\.\\d+)?(?:pM|nM|uM|μM|mM))_"  # concentration
        r"[Mm](?P<marker>\\d+)_"  # marker number, e.g. M1
        r"(?P<repeat>\\d+)"  # repeat number
        r"$"
    >>> metadata = extract_metadata_from_fd_curve_name_with_regex(curve_name, regex_pattern)
    >>> print(metadata)
    {
        'drug': 'CuOda',
        'conc': '10uM',
        'marker': 4,
        'repeat': 3
    }
    """
    if regex_pattern is None:
        print("Warning: No regex pattern provided for metadata extraction.")
        return {}
    metadata: dict[str, str | float | int | None] = {}
    compiled_regex_pattern = re.compile(regex_pattern)
    match = compiled_regex_pattern.match(curve_name)
    if not match:
        print(f"Warning: No match found for curve name '{curve_name}' with pattern '{regex_pattern}'")

    groupdict = match.groupdict() if match else {}
    missing_values = []
    for key, value in groupdict.items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing_values.append(key)
            metadata[key] = None
        else:
            coerced_value = attempt_coerce_string_to_numbers(value)
            metadata[key] = coerced_value
    if missing_values:
        print(
            f"Warning: Missing values for keys {missing_values} in curve"
            f" name '{curve_name}' with pattern '{regex_pattern}'"
        )
    return metadata
