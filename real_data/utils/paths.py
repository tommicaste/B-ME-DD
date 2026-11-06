"""Location of the raw datasets.

Every loader resolves its default paths under one root, `./data` unless the
BMEDD_DATA_DIR environment variable (or `run_experiment --data_dir`) says otherwise.
"""
import os
from pathlib import Path


def data_root():
    return Path(os.environ.get("BMEDD_DATA_DIR", "data"))
