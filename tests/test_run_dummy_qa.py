import json
import os

import pytest

from esgf_qa.run_qa import (
    process_dataset,
    process_file,
    run_compliance_checker,
)


@pytest.fixture
def tmp_env(tmp_path):
    """Fixture that sets up a temporary environment with paths and sample structures."""
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    progress_file = tmp_path / "progress.txt"
    progress_file.write_text("")
    return {"tmp": tmp_path, "results": result_dir, "progress": progress_file}


@pytest.fixture
def dummy_nc_file(tmp_env):
    """Create a fake dataset file."""
    file_path = tmp_env["tmp"] / "dummy.nc"
    file_path.write_text("fake dataset content")
    return str(file_path)


@pytest.fixture
def fake_check_suite(monkeypatch):
    """Monkeypatch CheckSuite to avoid real compliance logic."""

    class DummyCheck:
        def __init__(self, name):
            self.name = name
            self.weight = 1
            self.value = "PASS"
            self.msgs = []
            self.check_method = "check_method"
            self.children = []

    class DummyCheckSuite:
        def __init__(self, options=None):
            self.options = options or {}
            self.checkers = {}

        def load_all_available_checkers(self):
            pass

        def load_dataset(self, file_path):
            return f"dataset:{file_path}"

        def run_all(self, ds, checkers, include_checks=None, skip_checks=None):
            return {
                checker: (
                    [DummyCheck("time_bounds")],  # flat list of results
                    {},  # errors
                )
                for checker in checkers
            }

    monkeypatch.setattr("esgf_qa.run_qa.CheckSuite", DummyCheckSuite)
    return DummyCheckSuite


class TestDummyQA:
    """Tests for run_compliance_checker, process_file, and process_dataset."""

    def test_run_compliance_checker_basic(self, fake_check_suite, dummy_nc_file):
        checkers = ["cf:latest"]
        results = run_compliance_checker(dummy_nc_file, checkers)
        assert isinstance(results, dict)
        assert "cf:latest" in results
        assert isinstance(results["cf:latest"], tuple)
        assert isinstance(results["cf:latest"][0], list)

    def test_process_file(self, fake_check_suite, tmp_env, dummy_nc_file):
        """When no previous results exist, should run checks and write output."""
        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(tmp_env["results"] / "res.json"),
                "consistency_file": str(tmp_env["results"] / "cons.json"),
            }
        }
        processed_files = []
        checkers = ["cf:latest"]
        checker_options = {}

        file_path, result = process_file(
            dummy_nc_file,
            checkers,
            checker_options,
            files_to_check_dict,
            processed_files,
            str(tmp_env["progress"]),
        )

        # should write JSON to disk
        result_file = files_to_check_dict[dummy_nc_file]["result_file"]
        assert os.path.isfile(result_file)
        with open(result_file) as f:
            data = json.load(f)
        assert "cf" in data
        assert "errors" in data["cf"]

    def test_process_file_cached_result(self, fake_check_suite, tmp_env, dummy_nc_file):
        """Should read from disk if result already exists and no errors."""
        result_file = tmp_env["results"] / "res.json"
        consistency_file = tmp_env["results"] / "cons.json"
        result_file.write_text(json.dumps({"cf": {"errors": {}}}))
        consistency_file.write_text("dummy consistency file")

        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(result_file),
                "consistency_file": str(consistency_file),
            }
        }
        processed_files = [dummy_nc_file]
        checkers = ["cf:latest"]
        checker_options = {}

        file_path, result = process_file(
            dummy_nc_file,
            checkers,
            checker_options,
            files_to_check_dict,
            processed_files,
            str(tmp_env["progress"]),
        )

        # Should reuse cached result, not rewrite
        assert result == {"cf": {"errors": {}}}

    def test_process_dataset(self, fake_check_suite, tmp_env, dummy_nc_file):
        """process_dataset should run checks for not yet checked dataset."""
        ds = "dataset1"
        ds_map = {ds: [dummy_nc_file]}
        result_file_ds = tmp_env["results"] / "res_ds.json"

        files_to_check_dict = {dummy_nc_file: {"result_file_ds": str(result_file_ds)}}

        processed_datasets = set()
        checkers = ["unknown_checker:latest"]
        checker_options = {}

        ds_id, result = process_dataset(
            ds,
            ds_map,
            checkers,
            checker_options,
            files_to_check_dict,
            processed_datasets,
            str(tmp_env["progress"]),
        )

        # should write JSON file for dataset results
        assert ds_id == "dataset1"
        assert os.path.isfile(result_file_ds)
        with open(result_file_ds) as f:
            data = json.load(f)
        assert "unknown_checker" in data
        assert "errors" in data["unknown_checker"]
        assert "msg" in data["unknown_checker"]["errors"]["unknown_checker"]

    def test_process_dataset_cached(self, fake_check_suite, tmp_env, dummy_nc_file):
        """Should read dataset result if already processed and valid."""
        ds = "dataset2"
        ds_map = {ds: [dummy_nc_file]}
        result_file_ds = tmp_env["results"] / "res_ds2.json"
        result_file_ds.write_text(json.dumps({"cf": {"errors": {}}}))

        files_to_check_dict = {dummy_nc_file: {"result_file_ds": str(result_file_ds)}}
        processed_datasets = {ds}
        checkers = ["cf:latest"]
        checker_options = {}

        ds_id, result = process_dataset(
            ds,
            ds_map,
            checkers,
            checker_options,
            files_to_check_dict,
            processed_datasets,
            str(tmp_env["progress"]),
        )

        assert ds_id == ds
        assert result == {"cf": {"errors": {}}}
