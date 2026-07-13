"""Raw data loading and schema validation.

Loads the IBM AML HI-Small transaction CSV and validates its schema before
any downstream code touches it. The acceptance criterion for this module is
that loading **fails with a clear message** when the schema does not match
the shape the pipeline was validated against.

Behaviour
---------
* If the raw CSV is absent at ``RAW_CSV_PATH``, it is downloaded from Google
  Drive using ``RAW_CSV_GDRIVE_ID`` (~476 MB, mirrors the notebook).
* Once loaded, the DataFrame is validated against ``RAW_REQUIRED_COLUMNS``:

  - all required columns must be present;
  - ``Is Laundering`` must be strictly binary (0/1);
  - amount and bank columns must be numeric.

* Any violation raises :class:`SchemaValidationError` with the offending
  column named explicitly, so a broken input is diagnosed on the first line
  rather than in the middle of feature engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_CSV_GDRIVE_ID,
    RAW_CSV_PATH,
    RAW_REQUIRED_COLUMNS,
    TARGET,
)

__all__ = [
    "SchemaValidationError",
    "load_raw_transactions",
    "validate_schema",
]

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when the loaded CSV does not match the expected AML schema."""


# Columns that must be numeric for downstream pipeline steps.
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "From Bank",
    "To Bank",
    "Amount Received",
    "Amount Paid",
)


def _download_raw_csv(destination: Path) -> None:
    """Download the HI-Small CSV from Google Drive via ``gdown``."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "The raw CSV is missing and 'gdown' is not installed. "
            "Either place HI-Small_Trans.csv at "
            f"{destination} manually, or `pip install gdown`."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={RAW_CSV_GDRIVE_ID}"
    logger.info("Downloading raw CSV from Google Drive to %s", destination)
    gdown.download(url, str(destination), quiet=False)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that ``df`` matches the expected AML transaction schema.

    Raises :class:`SchemaValidationError` with a specific, actionable
    message when the input does not conform.
    """
    # 1. Presence of all required columns.
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns: {missing}. "
            f"Expected schema: {list(RAW_REQUIRED_COLUMNS)}."
        )

    # 2. Target column must be strictly binary (0/1).
    target_values = set(pd.unique(df[TARGET].dropna()))
    unexpected = target_values - {0, 1}
    if unexpected:
        raise SchemaValidationError(
            f"Column '{TARGET}' must contain only 0/1, "
            f"got unexpected values: {sorted(unexpected)}."
        )

    # 3. Numeric columns must have numeric dtype.
    for col in _NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaValidationError(
                f"Column '{col}' must be numeric, "
                f"got dtype {df[col].dtype}."
            )


def load_raw_transactions(
    csv_path: Path | None = None,
    *,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the raw AML transactions CSV, validating the schema before return.

    Parameters
    ----------
    csv_path
        Optional override for the CSV location. Defaults to
        :data:`src.config.RAW_CSV_PATH`.
    download_if_missing
        If ``True`` (default) and the file is absent, download it from
        Google Drive using ``RAW_CSV_GDRIVE_ID``. Set to ``False`` in tests
        to force a strict "file must exist" behaviour.

    Returns
    -------
    pd.DataFrame
        The full transaction table, schema-validated.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent and downloading is disabled.
    SchemaValidationError
        If the loaded table does not match ``RAW_REQUIRED_COLUMNS``.
    """
    path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Raw CSV not found at {path} and download_if_missing=False."
            )
        _download_raw_csv(path)

    logger.info("Reading %s", path)
    df = pd.read_csv(path)

    validate_schema(df)
    return df


def main() -> None:  # pragma: no cover - convenience CLI
    """Command-line smoke test: load the raw CSV and print a header summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = load_raw_transactions()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {RAW_CSV_PATH}.")
    print(df.head())


if __name__ == "__main__":
    main()
