"""
dataset.py
----------
Creates the project data directories and downloads raw datasets
from the Hugging Face hub into RAW_DATA_DIR.

All cleaning, filtering, and splitting is handled in the notebook.
"""

import pandas as pd
from datasets import load_dataset
from loguru import logger

from config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
)


def setup_dirs():
    """Create the project directory structure if it doesn't exist."""
    for d in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {d}")


def download_counsel_chat():
    """Download the Counsel Chat Q&A dataset to RAW_DATA_DIR."""
    out_path = RAW_DATA_DIR / "counsel_chat_raw.csv"
    logger.info("Downloading Counsel Chat dataset from Hugging Face...")
    df = load_dataset("nbertagnolli/counsel-chat", split="train").to_pandas()
    df.to_csv(out_path, index=False)
    logger.success(f"Counsel Chat saved to {out_path} ({len(df):,} rows)")


def download_crisis_data():
    """
    Download the Reddit suicide/depression detection dataset to RAW_DATA_DIR.

    Source: thePixel42/depression-detection (mirror of the Nikhileswar Komati
    Kaggle 'Suicide and Depression Detection' dataset).

    Schema:
        text  : str    -- Reddit post text
        label : int64  -- 1 = suicide, 0 = non-suicide

    Both the `train` (140k) and `test` (60k) splits are concatenated into a
    single raw CSV so downstream splitting can be done in the notebook.
    """
    out_path = RAW_DATA_DIR / "crisis_raw.csv"
    logger.info("Downloading crisis detection dataset from Hugging Face...")
    ds = load_dataset("thePixel42/depression-detection")
    df = pd.concat(
        [ds["train"].to_pandas(), ds["test"].to_pandas()],
        ignore_index=True,
    )
    df.to_csv(out_path, index=False)
    logger.success(f"Crisis data saved to {out_path} ({len(df):,} rows)")


def main():
    setup_dirs()
    download_counsel_chat()
    download_crisis_data()
    logger.success("All raw datasets downloaded.")


if __name__ == "__main__":
    main()