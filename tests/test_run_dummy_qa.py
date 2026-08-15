import json
import os
from types import SimpleNamespace

import pytest

from esgf_qa import workers
from esgf_qa.cluster_results import QAResultAggregator
from esgf_qa.con_checks import inter_dataset_consistency_checks
from esgf_qa.workers import (
    process_dataset,
    process_file,
    run_compliance_checker,
    run_dataset_collection_check,
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

    monkeypatch.setattr("esgf_qa.workers.CheckSuite", DummyCheckSuite)
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

    def test_process_file_preserves_multiple_severities(
        self, monkeypatch, tmp_env, dummy_nc_file
    ):
        """Results with the same name and different severities must all be stored."""
        checks = [
            SimpleNamespace(
                name="shared_check",
                weight=3,
                value=(0, 1),
                msgs=["Required failure"],
                check_method="check_shared",
                children=[],
            ),
            SimpleNamespace(
                name="shared_check",
                weight=1,
                value=(0, 1),
                msgs=["Suggested failure"],
                check_method="check_shared",
                children=[],
            ),
        ]
        monkeypatch.setattr(
            "esgf_qa.workers.run_compliance_checker",
            lambda *args, **kwargs: {"cf": (checks, {})},
        )
        result_file = tmp_env["results"] / "multiple-severities.json"
        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(result_file),
                "consistency_file": str(tmp_env["results"] / "cons.json"),
            }
        }

        _, result = process_file(
            dummy_nc_file,
            ["cf"],
            {},
            files_to_check_dict,
            [],
            str(tmp_env["progress"]),
        )

        saved_results = result["cf"]["shared_check"]
        assert isinstance(saved_results, list)
        assert [check["weight"] for check in saved_results] == [3, 1]
        saved_to_disk = json.loads(result_file.read_text())["cf"]["shared_check"]
        assert [check["weight"] for check in saved_to_disk] == [3, 1]
        assert [check["msgs"] for check in saved_to_disk] == [
            ["Required failure"],
            ["Suggested failure"],
        ]

    def test_process_file_records_missing_consistency_output(
        self, fake_check_suite, tmp_env, dummy_nc_file
    ):
        """A missing expected consistency output is a file-level runtime error."""
        consistency_file = tmp_env["results"] / "missing-consistency.json"
        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(tmp_env["results"] / "res.json"),
                "consistency_file": str(consistency_file),
            }
        }

        _, result = process_file(
            dummy_nc_file,
            ["cc6"],
            {},
            files_to_check_dict,
            [],
            str(tmp_env["progress"]),
        )

        error_msg = result["cc6"]["errors"]["consistency_output"]
        assert str(consistency_file) in error_msg

        aggregator = QAResultAggregator()
        aggregator.update(
            {"cc6": {"errors": result["cc6"]["errors"]}},
            "dataset1",
            dummy_nc_file,
        )
        assert aggregator.summary["error"]["[CORDEX-CMIP6] consistency_output"][
            error_msg
        ]["dataset1"] == [dummy_nc_file]

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

    def test_process_file_reruns_cached_runtime_error(
        self, fake_check_suite, tmp_env, dummy_nc_file
    ):
        """A processed file with a runtime error must be checked again."""
        result_file = tmp_env["results"] / "res.json"
        result_file.write_text(
            json.dumps({"cf": {"errors": {"check_old": "previous failure"}}})
        )
        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(result_file),
                "consistency_file": str(tmp_env["results"] / "cons.json"),
            }
        }

        _, result = process_file(
            dummy_nc_file,
            ["cf:latest"],
            {},
            files_to_check_dict,
            [dummy_nc_file],
            str(tmp_env["progress"]),
        )

        assert result["cf"]["errors"] == {}
        assert "time_bounds" in result["cf"]

    def test_process_file_removes_stale_outputs_before_rerun(
        self, monkeypatch, tmp_env, dummy_nc_file
    ):
        """A rerun must not accept an old consistency output as newly generated."""
        result_file = tmp_env["results"] / "res.json"
        consistency_file = tmp_env["results"] / "cons.json"
        result_file.write_text(
            json.dumps({"cc6": {"errors": {"check_old": "previous failure"}}})
        )
        consistency_file.write_text("stale consistency output")

        def run_without_consistency_output(*args, **kwargs):
            assert not result_file.exists()
            assert not consistency_file.exists()
            return {"cc6": ([], {})}

        monkeypatch.setattr(
            workers, "run_compliance_checker", run_without_consistency_output
        )
        files_to_check_dict = {
            dummy_nc_file: {
                "result_file": str(result_file),
                "consistency_file": str(consistency_file),
            }
        }

        _, result = process_file(
            dummy_nc_file,
            ["cc6"],
            {},
            files_to_check_dict,
            [dummy_nc_file],
            str(tmp_env["progress"]),
        )

        assert not consistency_file.exists()
        assert "consistency_output" in result["cc6"]["errors"]
        assert json.loads(result_file.read_text()) == result

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

    def test_process_dataset_preserves_multiple_severities(
        self, monkeypatch, tmp_env, dummy_nc_file
    ):
        """ESGF-QA checks can persist multiple severities under one check name."""
        check_results = {
            "shared_consistency_check": [
                {
                    "weight": 3,
                    "msgs": {"Required mismatch": [dummy_nc_file]},
                },
                {
                    "weight": 1,
                    "msgs": {"Suggested mismatch": [dummy_nc_file]},
                },
            ]
        }
        monkeypatch.setitem(
            workers.DATASET_CHECKERS,
            "cons",
            lambda *args, **kwargs: check_results,
        )
        result_file = tmp_env["results"] / "multiple-dataset-severities.json"
        files_to_check_dict = {dummy_nc_file: {"result_file_ds": str(result_file)}}

        _, result = process_dataset(
            "dataset1",
            {"dataset1": [dummy_nc_file]},
            ["cons"],
            {"cons": {}},
            files_to_check_dict,
            set(),
            str(tmp_env["progress"]),
        )

        saved_results = result["cons"]["shared_consistency_check"]
        assert [check["weight"] for check in saved_results] == [3, 1]
        saved_to_disk = json.loads(result_file.read_text())
        assert saved_to_disk == result

    def test_process_dataset_removes_stale_result_before_rerun(
        self, monkeypatch, tmp_env, dummy_nc_file
    ):
        """An erroneous dataset result is removed before its checks run again."""
        result_file = tmp_env["results"] / "dataset-result.json"
        result_file.write_text(
            json.dumps({"cons": {"errors": {"old": "previous failure"}}})
        )

        def replacement_check(*args, **kwargs):
            assert not result_file.exists()
            return {"replacement": {}}

        monkeypatch.setitem(workers.DATASET_CHECKERS, "cons", replacement_check)
        files_to_check_dict = {dummy_nc_file: {"result_file_ds": str(result_file)}}

        _, result = process_dataset(
            "dataset1",
            {"dataset1": [dummy_nc_file]},
            ["cons"],
            {"cons": {}},
            files_to_check_dict,
            {"dataset1"},
            str(tmp_env["progress"]),
        )

        assert result == {"cons": {"replacement": {}}}
        assert json.loads(result_file.read_text()) == result

    def test_process_dataset_records_runtime_error(self, tmp_env, dummy_nc_file):
        """An exception in con_checks is reported with its dataset and files."""
        second_nc_file = tmp_env["tmp"] / "dummy-2.nc"
        second_nc_file.write_text("fake dataset content")
        dataset_files = [dummy_nc_file, str(second_nc_file)]
        result_file = tmp_env["results"] / "dataset-result.json"
        invalid_consistency_file = tmp_env["results"] / "invalid.json"
        invalid_consistency_file.write_text("not valid JSON")
        files_to_check_dict = {
            file: {
                "result_file_ds": str(result_file),
                "consistency_file": str(invalid_consistency_file),
            }
            for file in dataset_files
        }

        _, result = process_dataset(
            "dataset1",
            {"dataset1": dataset_files},
            ["cons"],
            {"cons": {}},
            files_to_check_dict,
            set(),
            str(tmp_env["progress"]),
        )

        error = result["cons"]["errors"]["consistency_checks"]
        assert "con_checks.py" in error["msg"]
        assert "function/method 'consistency_checks'" in error["msg"]
        assert error["files"] == sorted(dataset_files)

        aggregator = QAResultAggregator()
        aggregator.update_ds(result, "dataset1")
        error_summary = aggregator.summary["error"]["[Consistency] consistency_checks"][
            error["msg"]
        ]
        assert error_summary["dataset1"] == sorted(dataset_files)

    def test_process_dataset_continues_after_runtime_error(
        self, monkeypatch, tmp_env, dummy_nc_file
    ):
        """One failed dataset check does not prevent the remaining checks."""

        def failing_check(*args, **kwargs):
            raise RuntimeError("consistency failure")

        monkeypatch.setitem(workers.DATASET_CHECKERS, "cons", failing_check)
        monkeypatch.setitem(
            workers.DATASET_CHECKERS,
            "cont",
            lambda *args, **kwargs: {"continued": {}},
        )
        result_file = tmp_env["results"] / "dataset-result.json"
        files_to_check_dict = {dummy_nc_file: {"result_file_ds": str(result_file)}}

        _, result = process_dataset(
            "dataset1",
            {"dataset1": [dummy_nc_file]},
            ["cons", "cont"],
            {"cons": {}, "cont": {}},
            files_to_check_dict,
            set(),
            str(tmp_env["progress"]),
        )

        assert "errors" in result["cons"]
        assert result["cont"] == {"continued": {}}

    def test_collection_check_runtime_error_covers_all_datasets(self):
        """An all-dataset failure is associated with every dataset in scope."""

        def failing_collection_check(*args, **kwargs):
            raise RuntimeError("all-dataset failure")

        ds_map = {
            "dataset1": ["file-1.nc", "file-2.nc"],
            "dataset2": ["file-3.nc"],
        }
        aggregator = QAResultAggregator()

        result = run_dataset_collection_check(
            aggregator,
            "cons",
            failing_collection_check,
            ds_map,
            {},
            {},
        )

        assert result is None
        error_group = aggregator.summary["error"][
            "[Consistency] failing_collection_check"
        ]
        error_msg = next(iter(error_group))
        assert "all-dataset failure" in error_msg
        assert error_group[error_msg]["dataset1"] == ["file-1.nc", "file-2.nc"]
        assert error_group[error_msg]["dataset2"] == ["file-3.nc"]

    def test_inter_dataset_runtime_error_discards_reference_result(
        self, tmp_env, dummy_nc_file
    ):
        """A failed inter-dataset check returns no partial reference metadata."""
        invalid_consistency_file = tmp_env["results"] / "invalid.json"
        invalid_consistency_file.write_text("not valid JSON")
        ds_map = {"dataset1": [dummy_nc_file]}
        files_to_check_dict = {
            dummy_nc_file: {"consistency_file": str(invalid_consistency_file)}
        }
        aggregator = QAResultAggregator()

        result = run_dataset_collection_check(
            aggregator,
            "cons",
            inter_dataset_consistency_checks,
            ds_map,
            files_to_check_dict,
            {},
        )

        assert result is None
        error_group = aggregator.summary["error"][
            "[Consistency] inter_dataset_consistency_checks"
        ]
        error_msg = next(iter(error_group))
        assert "con_checks.py" in error_msg
        assert "function/method 'inter_dataset_consistency_checks'" in error_msg
        assert error_group[error_msg]["dataset1"] == [dummy_nc_file]

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
