import json
import os
import tempfile
from datetime import timedelta

import pytest

from esgf_qa import con_checks as cc
from esgf_qa.con_checks import (
    compare_dicts,
    compare_nested_dicts,
    printtimedelta,
    truncate_str,
)


def test_printtimedelta():
    """
    Test the printtimedelta function.
    """
    # Test cases for timedelta values
    test_cases = [
        (timedelta(seconds=1), "1.0 seconds"),
        (timedelta(seconds=120), "2.0 minutes"),
        (timedelta(seconds=3600), "1.0 hours"),
        (timedelta(days=1), "1.0 days"),
        (timedelta(days=365), "365.0 days"),
    ]
    for timedelta_value, expected_output in test_cases:
        result = printtimedelta(timedelta_value.total_seconds())
        assert result == expected_output


def test_truncate_str():
    """
    Test the truncate_str function.
    """
    # Test cases for different string lengths and max_lengths
    test_cases = [
        ("This is a long string", 10, "This is...string"),
        ("This is a short string", 25, "This is a short string"),
        (
            "This is a really long string that needs to be truncated",
            12,
            "This is...truncated",
        ),
        ("This is not really a short string", 0, "This is not really a short string"),
        ("This is a really short string", 16, "This is...string"),
        (
            "Someone should truncate this truncatable string!",
            -10,
            "Someone should truncate this truncatable string!",
        ),
        ("Someone should truncate this truncatable string!", 20, "Someone...string!"),
    ]
    for s, max_length, expected_output in test_cases:
        result = truncate_str(s, max_length)
        assert result == expected_output


def test_compare_dicts():
    """
    Test the compare_dicts function.
    """
    # Test cases for different dictionary values and exclude keys
    test_cases = [
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 4},
            set(),
            ["c"],
        ),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
            set(),
            [],
        ),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 3, "c": 3},
            {"b"},
            [],
        ),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 3, "c": 3},
            set(),
            ["b"],
        ),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
            {"b", "c"},
            [],
        ),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
            set(),
            [],
        ),
    ]
    for dict1, dict2, exclude_keys, expected_output in test_cases:
        result = compare_dicts(dict1, dict2, exclude_keys)
        assert result == expected_output


def test_compare_nested_dicts():
    """
    Test the compare_nested_dicts function.
    """
    # Test cases for different nested dictionary values and exclude keys
    test_cases = [
        (
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 5}},
            set(),
            {"b": ["y"]},
        ),
        (
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 5}},
            {"b"},
            {"b": ["y"]},
        ),
        (
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 5}},
            {"y"},
            {},
        ),
        (
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            set(),
            {},
        ),
        (
            {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}},
            {"a": {"x": 1, "y": 2}, "b": {"x": 2, "y": 4, "z": 5}},
            set(),
            {"b": ["x", "z"]},
        ),
    ]
    for dict1, dict2, exclude_keys, expected_output in test_cases:
        result = compare_nested_dicts(dict1, dict2, exclude_keys)
        assert result == expected_output


@pytest.fixture
def temp_files():
    """Create temporary JSON files for testing consistency/continuity."""
    files = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 fake files per dataset
        for ds in ["ds1", "ds2"]:
            files[ds] = []
            for i in range(2):
                fpath = os.path.join(tmpdir, f"{ds}_file{i}.json")
                data = {
                    "global_attributes": {"title": "fake", "history": "x"},
                    "global_attributes_non_required": {"notes": "x"},
                    "global_attributes_dtypes": {"title": "str"},
                    "variable_attributes": {"var1": {"units": "m"}},
                    "variable_attributes_dtypes": {"var1": {"units": "str"}},
                    "dimensions": {"time": 10, "lat": 5, "lon": 5},
                    "coordinates": {"lat": [0, 1, 2, 3, 4], "lon": [0, 1, 2, 3, 4]},
                    "time_info": {
                        "timen": 0,
                        "boundn": 10,
                        "time0": 0,
                        "bound0": 10,
                        "units": "days since 2000-01-01",
                        "calendar": "gregorian",
                        "frequency": "day",
                    },
                }
                with open(fpath, "w") as f:
                    json.dump(data, f)
                files[ds].append(fpath)
        yield files  # return dict of dataset -> list of file paths


@pytest.fixture
def files_to_check_dict(temp_files):
    """Create the files_to_check_dict required by con_checks."""
    d = {}
    for ds, flist in temp_files.items():
        for f in flist:
            d[f] = {
                "consistency_file": f,
                "ts": "20000101-20000102",
                "result_file": os.path.join(
                    tempfile.gettempdir(), f"result_{os.path.basename(f)}.json"
                ),
                "result_file_ds": os.path.join(
                    tempfile.gettempdir(), f"result_ds_{ds}.json"
                ),
            }
    return d


@pytest.fixture
def ds_map(temp_files):
    """Map dataset names to file paths."""
    return {ds: flist for ds, flist in temp_files.items()}


class TestConChecks:
    def test_consistency_checks(self, ds_map, files_to_check_dict):
        results = cc.consistency_checks("ds1", ds_map, files_to_check_dict, {})
        assert isinstance(results, dict)
        assert "Required global attributes" in results

    def test_continuity_checks(self, ds_map, files_to_check_dict):
        results = cc.continuity_checks("ds1", ds_map, files_to_check_dict, {})
        assert isinstance(results, dict)
        assert "Time continuity" in results

    def test_compatibility_checks(self, ds_map, files_to_check_dict):
        results = cc.compatibility_checks("ds1", ds_map, files_to_check_dict, {})
        assert isinstance(results, dict)
        # The open_mfdataset will fail on minimal data, so we can check that msgs exist
        assert any("open_mfdataset" in key for key in results.keys())

    def test_dataset_coverage_checks(temp_files):
        # Example: manually create a minimal dataset map with two datasets
        ds_map = {
            "ds1": ["file_ds1_1.json", "file_ds1_2.json"],
            "ds2": ["file_ds2_1.json", "file_ds2_2.json"],
        }

        # Corresponding files_to_check_dict for those files
        files_to_check_dict = {}
        for ds, flist in ds_map.items():
            ts_ranges = (
                ["20000101-20001231", "20010101-20011231"]
                if ds == "ds1"
                else ["20010101-20010630", "20010701-20011231"]
            )
            for idx, f in enumerate(flist):
                files_to_check_dict[f] = {
                    "consistency_file": f,
                    "ts": ts_ranges[idx],
                    "result_file": f"result_{f}.json",
                    "result_file_ds": f"result_ds_{ds}.json",
                }
        results = cc.dataset_coverage_checks(ds_map, files_to_check_dict, {})
        # There should be weight=1 messages for ds2 due to differing start year
        assert "Time coverage" in results["ds2"]
        assert any(
            "Time series starts at '2001'" in msg
            for msg in results["ds2"]["Time coverage"]["msgs"]
        )

    def test_inter_dataset_consistency_checks(self, ds_map, files_to_check_dict):
        results, ref_ds = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, {}
        )
        assert isinstance(results, dict)
        assert isinstance(ref_ds, dict)
        assert "general_reference" in ref_ds
