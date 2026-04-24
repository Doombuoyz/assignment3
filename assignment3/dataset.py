"""
dataset.py
----------
Creates the project data directories and downloads raw datasets into
RAW_DATA_DIR.

Sources:
    - Counsel Chat        -> Hugging Face  (nbertagnolli/counsel-chat)
    - Suicide / Crisis    -> Kaggle        (nikhileswarkomati/suicide-watch)

Kaggle authentication:
    Add these to a `.env` file at the project root (and make sure
    `.env` is in `.gitignore`):

        KAGGLE_USERNAME=your_username
        KAGGLE_KEY=your_api_key_from_kaggle_settings

    Get the values from https://www.kaggle.com/settings
    -> "API" section -> "Create New Token" (downloads kaggle.json).

All cleaning, filtering, and splitting is handled in the notebook /
EDA script. This file only downloads raw data.
"""
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
)


# Kaggle dataset slug and the CSV file inside the zip
KAGGLE_CRISIS_DATASET = "nikhileswarkomati/suicide-watch"
KAGGLE_CRISIS_FILE = "Suicide_Detection.csv"


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


def _load_kaggle_credentials():
    """
    Load Kaggle credentials from .env into environment variables.

    The `kaggle` package reads KAGGLE_USERNAME and KAGGLE_KEY from os.environ
    when authenticate() is called, so we just need to make sure they're set
    before importing/using the Kaggle API.
    """
    load_dotenv()  # looks for .env in current dir and parents

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. Add the following to a .env file "
            "at the project root:\n"
            "    KAGGLE_USERNAME=your_username\n"
            "    KAGGLE_KEY=your_api_key\n"
            "Get them from https://www.kaggle.com/settings (API section)."
        )

    logger.info(f"Loaded Kaggle credentials for user: {username}")


def download_counsel_chat():
    """Download the Counsel Chat Q&A dataset from Hugging Face."""
    out_path = RAW_DATA_DIR / "counsel_chat_raw.csv"
    logger.info("Downloading Counsel Chat dataset from Hugging Face...")
    df = load_dataset("nbertagnolli/counsel-chat", split="train").to_pandas()
    df.to_csv(out_path, index=False)
    logger.success(f"Counsel Chat saved to {out_path} ({len(df):,} rows)")


def download_crisis_data():
    """
    Download the Suicide and Depression Detection dataset from Kaggle.

    Source: nikhileswarkomati/suicide-watch (Komati 2021) — Reddit posts
    from r/SuicideWatch, r/depression, r/teenagers. Binary labels stored
    as strings: 'suicide' / 'non-suicide'.

    Schema after download:
        Unnamed: 0 : int  -- stray index from original pandas save
        text       : str  -- Reddit post text
        class      : str  -- 'suicide' or 'non-suicide'

    The file is saved to RAW_DATA_DIR/crisis_raw.csv with no modification.
    All cleaning and label mapping happens in the EDA/cleaning step.
    """
    out_path = RAW_DATA_DIR / "crisis_raw.csv"

    _load_kaggle_credentials()

    # Import here so credentials are in os.environ before the kaggle package
    # is initialised (it reads env vars at import/auth time).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading {KAGGLE_CRISIS_DATASET} from Kaggle...")
    tmp_dir = RAW_DATA_DIR / "_kaggle_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        KAGGLE_CRISIS_DATASET,
        path=str(tmp_dir),
        unzip=True,
        quiet=False,
    )

    # Move the CSV to its canonical location and clean up the temp dir
    downloaded = tmp_dir / KAGGLE_CRISIS_FILE
    if not downloaded.exists():
        # Fall back to the first CSV in the folder in case the filename changed
        csvs = list(tmp_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV found in {tmp_dir} after Kaggle download. "
                f"Contents: {[p.name for p in tmp_dir.iterdir()]}"
            )
        downloaded = csvs[0]
        logger.warning(
            f"Expected {KAGGLE_CRISIS_FILE} but got {downloaded.name} — using it."
        )

    shutil.move(str(downloaded), str(out_path))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Quick sanity check and row count for the log
    n_rows = sum(1 for _ in open(out_path, encoding="utf-8")) - 1
    logger.success(f"Crisis data saved to {out_path} ({n_rows:,} rows)")


def main():
    setup_dirs()
    download_counsel_chat()
    download_crisis_data()
    logger.success("All raw datasets downloaded.")


if __name__ == "__main__":
    main()