#!/usr/bin/env python3
"""
Generate `data/BDL_sample_data.csv` from `data/sample_data.csv`.

Produces a censored copy of the Cairngorms geochemical survey for exercising
`funmixer.BDLSampleNetworkUnmixer`. Sample names, coordinates and the trailing metadata
columns are copied verbatim, so the output remains a valid input to `get_sample_graph`.

For each element column independently, the lowest decile of values is replaced by a censored
entry of the form "<X":

    value < p5           ->  "<p5"
    p5 <= value < p10    ->  "<p10"

Giving each element two distinct detection limits exercises the ability to handle a different
limit per element, and a different limit within a single element. Every replacement is
truthful: a censored value is always genuinely below the limit printed in its place, because
detection limits are rounded *up* to the reported precision.

Blank cells are left blank - they are missing data, not censored data. The transformation is
purely deterministic (percentiles of the input), so no random seed is involved.

Run from the repository root:

    python data/make_bdl_sample_data.py
"""

import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# This script lives in `data/`, so the repository root is not on `sys.path` even when run from
# there. Add it, so the script works whether or not funmixer has been `pip install -e .`d.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from funmixer import ELEMENT_LIST  # noqa: E402  (import follows the sys.path fix above)

# Percentiles at which the two detection limits for each element are placed. Values below the
# lower percentile are censored at the lower limit; values between the two are censored at the
# upper limit. Together these censor the lowest decile of each element.
LOWER_PERCENTILE = 5.0
UPPER_PERCENTILE = 10.0

# Significant figures used when printing a detection limit.
LIMIT_SIG_FIGS = 6

# An element column is left untouched if it has fewer than this many valid measurements, since
# percentiles of a handful of points are not meaningful.
MIN_VALID_VALUES = 10


def round_up_to_sig_figs(value: float, sig_figs: int = LIMIT_SIG_FIGS) -> float:
    """
    Round a positive number up to a given number of significant figures.

    Rounding up (rather than to nearest) guarantees that every value censored at the returned
    limit really is below it, so the censored dataset never makes a false statement.

    Args:
        value: The number to round. Must be strictly positive.
        sig_figs: Number of significant figures to retain.

    Returns:
        The smallest number with `sig_figs` significant figures that is >= `value`.

    Raises:
        ValueError: If `value` is not finite and strictly positive.
    """
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"Cannot round '{value}': a detection limit must be finite and positive.")
    factor = 10 ** (sig_figs - 1 - math.floor(math.log10(value)))
    return math.ceil(value * factor) / factor


def censor_column(
    values: pd.Series,
    lower_percentile: float = LOWER_PERCENTILE,
    upper_percentile: float = UPPER_PERCENTILE,
    min_valid_values: int = MIN_VALID_VALUES,
) -> Optional[pd.Series]:
    """
    Replace the lowest values of one element column with censored "<X" strings.

    Args:
        values: Numeric concentrations for one element. Missing entries are ignored.
        lower_percentile: Percentile defining the lower detection limit.
        upper_percentile: Percentile defining the upper detection limit.
        min_valid_values: Minimum number of valid measurements required to censor at all.

    Returns:
        A series of strings with censored entries replaced, or None if the column has too few
        valid or positive values to censor.
    """
    valid = values.dropna()
    if len(valid) < min_valid_values or (valid <= 0).any():
        return None

    lower_limit = round_up_to_sig_figs(float(np.percentile(valid, lower_percentile)))
    upper_limit = round_up_to_sig_figs(float(np.percentile(valid, upper_percentile)))

    def censor(value: float) -> Optional[str]:
        if pd.isna(value):
            return None
        if value < lower_limit:
            return f"<{lower_limit:g}"
        if value < upper_limit:
            return f"<{upper_limit:g}"
        return None

    return values.map(censor)


def make_bdl_dataset(
    source_path: Path,
    output_path: Path,
    lower_percentile: float = LOWER_PERCENTILE,
    upper_percentile: float = UPPER_PERCENTILE,
) -> List[str]:
    """
    Write a censored copy of a sample-data CSV.

    Args:
        source_path: The uncensored sample-data CSV to read.
        output_path: Where to write the censored CSV.
        lower_percentile: Percentile defining the lower detection limit for each element.
        upper_percentile: Percentile defining the upper detection limit for each element.

    Returns:
        Names of the element columns that were censored.

    Raises:
        FileNotFoundError: If `source_path` does not exist.
        ValueError: If the percentiles are not in ascending order.
    """
    if not source_path.exists():
        raise FileNotFoundError(
            f"Could not find '{source_path}'. Run this script from the repository root."
        )
    if lower_percentile >= upper_percentile:
        raise ValueError(
            f"lower_percentile ({lower_percentile}) must be below upper_percentile "
            f"({upper_percentile})."
        )

    # Read twice: once as text so untouched cells keep their exact original formatting, and
    # once numerically so percentiles can be computed.
    text = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    numeric = pd.read_csv(source_path)

    censored_columns: List[str] = []
    for element in ELEMENT_LIST:
        if element not in text.columns:
            continue
        replacements = censor_column(
            numeric[element],
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        if replacements is None:
            continue
        mask = replacements.notna()
        if not mask.any():
            continue
        text.loc[mask, element] = replacements[mask]
        censored_columns.append(element)

    text.to_csv(output_path, index=False)
    return censored_columns


def main() -> None:
    """Generate the censored example dataset and report what was censored."""
    source_path = Path("data/sample_data.csv")
    output_path = Path("data/BDL_sample_data.csv")

    censored_columns = make_bdl_dataset(source_path=source_path, output_path=output_path)

    written = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    n_censored = int(written[censored_columns].map(lambda v: v.startswith("<")).to_numpy().sum())
    n_cells = len(written) * len(censored_columns)
    print(f"Wrote {output_path} ({len(written)} samples, {len(written.columns)} columns).")
    print(f"Censored {len(censored_columns)} element columns.")
    print(
        f"{n_censored} of {n_cells} element measurements are below detection limit "
        f"({100 * n_censored / n_cells:.1f}%)."
    )


if __name__ == "__main__":
    main()
