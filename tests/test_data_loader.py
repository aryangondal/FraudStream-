import os
from pathlib import Path

import pytest

from src.data_loader import DATA_FILENAME, get_data_path, load_creditcard_data


def test_data_path_exists():
    data_path = get_data_path()
    assert data_path.name == DATA_FILENAME
    assert data_path.parent.exists()


def test_load_creditcard_data_file_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_creditcard_data()
