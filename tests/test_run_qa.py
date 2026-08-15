import csv
import json
import os
import re
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest

from esgf_qa import checker_registry, cli, run_qa, workflow
from esgf_qa._constants import (
    checker_dict,
    checker_supporting_consistency_checks,
)
from esgf_qa.checker_registry import (
    CheckerMetadata,
    format_checker_version,
    get_checker_metadata,
    normalize_checker_specs,
)
from esgf_qa.cli import parse_options, prepare_run
from esgf_qa.discovery import (
    _checker_options_for_file,
    discover_files,
    get_dsid,
    write_excluded_files,
)
from esgf_qa.resume import (
    invalidate_nonreusable_dataset_results,
    track_checked_datasets,
    verify_options_dict,
)
from esgf_qa.run_qa import get_default_result_dir


def test_main_enables_cf_appendix_a_checks(monkeypatch, tmp_path):
    """The main workflow passes Appendix A under Compliance Checker's CF key."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.nc").touch()
    output_dir = tmp_path / "output"
    captured_options = {}

    monkeypatch.setattr(
        cli, "get_installed_checker_versions", lambda: {"cf": ["latest"]}
    )

    def capture_process_file(
        file_path,
        checkers,
        checker_options,
        files_to_check_dict,
        processed_files,
        progress_file,
    ):
        captured_options.update(checker_options)
        return file_path, {"cf": {"errors": {}}}

    monkeypatch.setattr(workflow, "process_file", capture_process_file)
    monkeypatch.setattr(
        workflow,
        "get_checker_metadata",
        lambda checkers: {"cf": CheckerMetadata("test")},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["esgqa", "-t", "cf", "-o", str(output_dir), str(input_dir)],
    )

    run_qa.main()

    assert captured_options["cf"]["enable_appendix_a_checks"] is True
    assert "cf:" not in captured_options


def test_new_file_invalidates_cached_dataset_result(tmp_path):
    cached_file = tmp_path / "cached.nc"
    new_file = tmp_path / "new.nc"
    cached_result = tmp_path / "cached-result.json"
    cached_result.write_text(json.dumps({"cf": {"errors": {}}}))
    files_to_check_dict = {
        str(cached_file): {
            "result_file": str(cached_result),
            "consistency_file": str(tmp_path / "cached-consistency.json"),
        },
        str(new_file): {
            "result_file": str(tmp_path / "new-result.json"),
            "consistency_file": str(tmp_path / "new-consistency.json"),
        },
    }
    processed_datasets = {"dataset1"}

    invalidate_nonreusable_dataset_results(
        {"dataset1": [str(cached_file), str(new_file)]},
        ["cf"],
        files_to_check_dict,
        {str(cached_file)},
        processed_datasets,
    )

    assert processed_datasets == set()


def test_main_retains_info_when_resuming(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.nc").touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "tables").mkdir()
    (output_dir / "progress.txt").touch()
    (output_dir / "progress_datasets.txt").touch()
    (output_dir / ".resume_info").write_text(
        json.dumps(
            {
                "parent_dir": str(input_dir),
                "info": "original-info",
                "tests": ["cf"],
            }
        )
    )

    monkeypatch.setattr(
        cli, "get_installed_checker_versions", lambda: {"cf": ["latest"]}
    )
    monkeypatch.setattr(
        workflow,
        "get_checker_metadata",
        lambda checkers: {"cf": CheckerMetadata("test")},
    )
    monkeypatch.setattr(
        workflow,
        "process_file",
        lambda file_path, *args: (file_path, {"cf": {"errors": {}}}),
    )
    monkeypatch.setattr(sys, "argv", ["esgqa", "-r", "-o", str(output_dir)])

    run_qa.main()

    resume_info = json.loads((output_dir / ".resume_info").read_text())
    assert resume_info["info"] == "original-info"
    result_file = next(
        path
        for path in output_dir.glob("qa_result_*.json")
        if not path.name.endswith(".cluster.json")
    )
    result = json.loads(result_file.read_text())
    assert result["info"]["id"].startswith("original-info ")


def test_main_rejects_invalid_stored_checker_options(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "tables").mkdir()
    (output_dir / "progress.txt").touch()
    (output_dir / ".resume_info").write_text(
        json.dumps(
            {
                "parent_dir": str(input_dir),
                "info": "original-info",
                "tests": ["cf"],
                "checker_options": [],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["esgqa", "-r", "-o", str(output_dir)])

    with pytest.raises(Exception, match="checker_options"):
        run_qa.main()


def test_main_rejects_empty_input_before_writing_inventory(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="No NetCDF files found"):
        run_qa.main(["-o", str(output_dir), str(input_dir)])

    assert not (output_dir / "files_to_check.json").exists()
    assert not (output_dir / "excluded_files.json").exists()


def test_empty_input_with_filters_writes_exclusion_report(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="No NetCDF files found"):
        run_qa.main(["-w", "1950", "-o", str(output_dir), str(input_dir)])

    report = json.loads((output_dir / "excluded_files.json").read_text())
    assert report["summary"] == {
        "discovered": 0,
        "selected": 0,
        "blacklisted": 0,
        "not_whitelisted": 0,
    }


def test_main_reports_when_filters_exclude_every_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "tas_195001-195012.nc").touch()
    (input_dir / "tas_196001-196012.nc").touch()
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="No files remain to check") as error:
        run_qa.main(
            [
                "-w",
                "not-present",
                "-o",
                str(output_dir),
                str(input_dir),
            ]
        )

    report_path = output_dir / "excluded_files.json"
    assert str(report_path) in str(error.value)
    report = json.loads(report_path.read_text())
    assert report["summary"] == {
        "discovered": 2,
        "selected": 0,
        "blacklisted": 0,
        "not_whitelisted": 2,
    }
    assert report["not_whitelisted"] == sorted(
        str(path) for path in input_dir.glob("*.nc")
    )


def test_path_filters_are_restored_when_resuming(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    default_output = str(tmp_path / "unused-default")

    config = prepare_run(
        default_output,
        [
            "-w",
            "1950",
            "-w",
            "historical",
            "-b",
            "ICON-ESM",
            "-o",
            str(output_dir),
            str(input_dir),
        ],
    )
    resume_info = json.loads((output_dir / ".resume_info").read_text())

    assert config.whitelist == ["1950", "historical"]
    assert config.blacklist == ["ICON-ESM"]
    assert resume_info["whitelist"] == config.whitelist
    assert resume_info["blacklist"] == config.blacklist

    resumed = prepare_run(
        default_output,
        ["-r", "-o", str(output_dir)],
    )

    assert resumed.whitelist == config.whitelist
    assert resumed.blacklist == config.blacklist


@pytest.mark.parametrize("filter_option", ["-w", "-b"])
def test_resume_rejects_new_path_filters(tmp_path, filter_option):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    prepare_run(
        str(tmp_path / "unused-default"),
        ["-o", str(output_dir), str(input_dir)],
    )

    with pytest.raises(SystemExit):
        prepare_run(
            str(tmp_path / "unused-default"),
            ["-r", filter_option, "fragment", "-o", str(output_dir)],
        )


@pytest.mark.parametrize("filter_option", ["-w", "-b"])
def test_empty_path_filter_is_rejected(tmp_path, filter_option):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(SystemExit):
        prepare_run(
            str(tmp_path / "unused-default"),
            [filter_option, "", str(input_dir)],
        )


@pytest.mark.parametrize(
    "option_args", [[], ["-O", "mip:tables"], ["-O", "mip:tables:"]]
)
def test_main_rejects_mip_without_table_path(monkeypatch, tmp_path, option_args):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        cli, "get_installed_checker_versions", lambda: {"mip": ["latest"]}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "esgqa",
            "-t",
            "mip",
            *option_args,
            "-o",
            str(output_dir),
            str(input_dir),
        ],
    )

    with pytest.raises(Exception, match="tables.*path.*explicitly selected"):
        run_qa.main()


def test_main_rejects_mip_and_eerie_together(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        cli, "get_installed_checker_versions", lambda: {"mip": ["latest"]}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "esgqa",
            "-t",
            "mip",
            "-t",
            "eerie",
            "-o",
            str(output_dir),
            str(input_dir),
        ],
    )

    with pytest.raises(Exception, match="Cannot run both 'mip'.*'eerie'"):
        run_qa.main()


@pytest.mark.parametrize(
    "checker, checker_args, expected_tables",
    [
        (
            "mip",
            ["-t", "eerie", "-O", "eerie:tables:/custom/eerie/Tables"],
            "/custom/eerie/Tables",
        ),
        ("wcrp_cmip6plus", ["-t", "wcrp_cmip6plus"], None),
    ],
)
def test_main_configures_checker_consistency_output(
    monkeypatch, tmp_path, checker, checker_args, expected_tables
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.nc").touch()
    output_dir = tmp_path / "output"
    captured = {}

    monkeypatch.setattr(
        cli,
        "get_installed_checker_versions",
        lambda: {"mip": ["latest"], "wcrp_cmip6plus": ["1.0", "latest"]},
    )

    def capture_process_file(file_path, checkers, checker_options, *args):
        captured["checkers"] = checkers
        captured["options"] = checker_options
        return file_path, {checker: {"errors": {}}}

    monkeypatch.setattr(workflow, "process_file", capture_process_file)
    monkeypatch.setattr(
        workflow,
        "get_checker_metadata",
        lambda checkers: {checker: CheckerMetadata("test")},
    )
    monkeypatch.setattr(workflow, "run_dataset_collection_check", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["esgqa", *checker_args, "-o", str(output_dir), str(input_dir)],
    )

    run_qa.main()

    assert captured["checkers"] == [checker]
    assert captured["options"][checker]["consistency_output"].endswith(".json")
    if expected_tables is not None:
        assert captured["options"][checker]["tables"] == expected_tables
    assert checker in checker_supporting_consistency_checks
    assert checker in checker_dict


# Test get_default_result_dir
def test_get_default_result_dir(tmpdir):
    """
    Test the get_default_result_dir function.
    """
    os.chdir(tmpdir)
    cwd = re.escape(os.getcwd())
    result_dir = get_default_result_dir()
    result_dir2 = get_default_result_dir()
    # Assert that the result directories are the same
    #  (they depend on when the library was imported /
    #   the program was executed)
    assert result_dir == result_dir2
    # Example: /path/to/cwd/esgf-qa-results_20251103-1209_bf5ae0fafabf6cc03e71180efe3e468c
    assert re.match(
        rf"^{cwd}/esgf-qa-results_\d{{8}}-\d{{4}}_[a-f0-9]{{32}}$", result_dir
    )


def test_get_dsid():
    """
    Test the get_dsid function.
    """
    project_id = "my_project"
    files_to_check_dict = {
        f"/path/to/{project_id}/drs/elements/until/file1_1950-1960.nc": {
            "id_dir": f"/path/to/{project_id}/drs/elements/until",
            "id_fn": "file1",
        },
        f"/path/to/{project_id}/drs2/elements2/until2/file2_1955-1960.nc": {
            "id_dir": f"/path/to/{project_id}/drs2/elements2/until2",
            "id_fn": "file2",
        },
    }
    dataset_files_map_ext = {
        f"/path/to/{project_id}/drs/elements/until": {
            "file1": ["file1_1950-1960.nc"],
        },
        f"/path/to/{project_id}/drs2/elements2/until2": {
            "file2": ["file2_1955-1960.nc"],
        },
    }
    file_path = f"/path/to/{project_id}/drs/elements/until/file1_1950-1960.nc"
    dsid = get_dsid(files_to_check_dict, dataset_files_map_ext, file_path, [project_id])
    assert dsid == "my_project.drs.elements.until"


def test_checker_options_for_file_uses_consistency_checker_registry(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    consistency_file = str(tmp_path / "consistency.json")
    cli_options = {
        "custom_checker": {"custom_option": "value"},
        "cf": {"cf_option": "value", "enable_appendix_a_checks": False},
        "cc6": {"tables_dir": "/wrong/path", "cc6_option": "value"},
        "wcrp_cmip7": {
            "consistency_output": "/wrong/output.json",
            "plugin_option": "value",
        },
    }

    options = _checker_options_for_file(
        "/input/first.nc",
        "/input/first.nc",
        consistency_file,
        str(tmp_path),
        cli_options,
        time_checks_only=True,
        resume=False,
    )

    assert options["custom_checker"] == {"custom_option": "value"}
    assert options["cf"] == {
        "cf_option": "value",
        "enable_appendix_a_checks": True,
    }
    for checker in checker_supporting_consistency_checks:
        assert options[checker]["consistency_output"] == consistency_file
    assert options["wcrp_cmip7"]["plugin_option"] == "value"
    assert options["cc6"]["cc6_option"] == "value"
    assert options["cc6"]["tables_dir"] == str(tables_dir)
    assert options["cc6"]["force_table_download"] is True
    assert options["cc6"]["time_checks_only"] is True
    assert options["mip"]["time_checks_only"] is True
    assert cli_options["wcrp_cmip7"]["consistency_output"] == "/wrong/output.json"


def test_discovery_filters_literal_path_fragments_with_blacklist_precedence(
    tmp_path,
):
    input_dir = tmp_path / "input"
    selected_filename = input_dir / "other-model" / "tas_195001-195012.nc"
    selected_path = input_dir / "historical" / "tas_196001-196012.nc"
    blacklisted_file = input_dir / "ICON-ESM" / "tas_195001-195012.nc"
    blacklisted_filename = input_dir / "other-model" / "tas_blocked_1950.nc"
    not_whitelisted_file = input_dir / "other-model" / "tas_196001-196012.nc"
    for file_path in (
        selected_filename,
        selected_path,
        blacklisted_file,
        blacklisted_filename,
        not_whitelisted_file,
    ):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    result_dir = tmp_path / "output"
    result_dir.mkdir()
    (result_dir / "tables").mkdir()
    config = SimpleNamespace(
        parent_dir=str(input_dir),
        result_dir=str(result_dir),
        whitelist=["1950", "historical"],
        blacklist=["ICON-ESM", "blocked"],
        checkers=["cf"],
        checker_options=defaultdict(dict),
        time_checks_only=False,
        resume=False,
        processed_files=set(),
        processed_datasets=set(),
    )

    inventory = discover_files(config)
    report_path = write_excluded_files(inventory, config)

    assert inventory.discovered_file_count == 5
    assert inventory.files == sorted([str(selected_filename), str(selected_path)])
    assert inventory.blacklisted_files == {
        str(blacklisted_file): ["ICON-ESM"],
        str(blacklisted_filename): ["blocked"],
    }
    assert inventory.not_whitelisted_files == [str(not_whitelisted_file)]
    with open(report_path) as report_file:
        report = json.load(report_file)
    assert report["filters"] == {
        "whitelist": ["1950", "historical"],
        "blacklist": ["ICON-ESM", "blocked"],
    }
    assert report["summary"] == {
        "discovered": 5,
        "selected": 2,
        "blacklisted": 2,
        "not_whitelisted": 1,
    }


def test_excluded_file_report_is_written_when_filters_exclude_nothing(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    selected_file = input_dir / "tas_195001-195012.nc"
    selected_file.touch()
    result_dir = tmp_path / "output"
    result_dir.mkdir()
    (result_dir / "tables").mkdir()
    config = SimpleNamespace(
        parent_dir=str(input_dir),
        result_dir=str(result_dir),
        whitelist=["1950"],
        blacklist=[],
        checkers=["cf"],
        checker_options=defaultdict(dict),
        time_checks_only=False,
        resume=False,
        processed_files=set(),
        processed_datasets=set(),
    )

    inventory = discover_files(config)
    report_path = write_excluded_files(inventory, config)

    assert report_path == str(result_dir / "excluded_files.json")
    report = json.loads((result_dir / "excluded_files.json").read_text())
    assert report["summary"]["selected"] == 1
    assert report["blacklisted"] == {}
    assert report["not_whitelisted"] == []


def test_get_checker_metadata():
    """
    Test function get_checker_metadata.

    Verifies that checker metadata is returned without mutating global state.
    """
    checkers = ["cf:1.6", "cc6:latest", "wcrp_cmip6:latest"]
    metadata = get_checker_metadata(checkers)

    assert "cf" in metadata
    assert "cc6" in metadata
    assert "wcrp_cmip6" in metadata

    for checker_metadata in metadata.values():
        assert checker_metadata.checker_version
    assert metadata["cf"].checker_version == "1.6"
    assert metadata["cf"].package_name == "compliance-checker"
    assert metadata["cc6"].package_name == "cc-plugin-cc6"


def test_checker_metadata_falls_back_without_entry_point(monkeypatch):
    checker_v1 = SimpleNamespace(_cc_spec_version="1.0")
    checker_v2 = SimpleNamespace(_cc_spec_version="2.0")

    class FakeCheckSuite:
        def __init__(self, options=None):
            self.checkers = {"demo:1.0": checker_v1, "demo:2.0": checker_v2}

        def load_all_available_checkers(self):
            pass

    monkeypatch.setattr(checker_registry, "CheckSuite", FakeCheckSuite)
    monkeypatch.setattr(checker_registry, "entry_points", lambda **kwargs: [])

    metadata = get_checker_metadata(["demo"])

    assert metadata["demo"] == CheckerMetadata("2.0")


def test_checker_metadata_skips_broken_entry_point(monkeypatch):
    checker = SimpleNamespace(
        _cc_spec="demo",
        _cc_spec_version="2.0",
    )

    class FakeCheckSuite:
        def __init__(self, options=None):
            self.checkers = {"demo:2.0": checker}

        def load_all_available_checkers(self):
            pass

    class BrokenEntryPoint:
        dist = SimpleNamespace(name="broken-package", version="0.1")

        def load(self):
            raise RuntimeError("broken plugin")

    class WorkingEntryPoint:
        dist = SimpleNamespace(name="demo-package", version="3.4")

        def load(self):
            return checker

    monkeypatch.setattr(checker_registry, "CheckSuite", FakeCheckSuite)
    monkeypatch.setattr(
        checker_registry,
        "entry_points",
        lambda **kwargs: [BrokenEntryPoint(), WorkingEntryPoint()],
    )

    metadata = get_checker_metadata(["demo:2.0"])

    assert metadata["demo"] == CheckerMetadata("2.0", "demo-package", "3.4")


def test_format_checker_version_includes_providing_package(monkeypatch):
    monkeypatch.setitem(checker_dict, "wcrp_cmip7", "CMIP7")
    metadata = {
        "wcrp_cmip7": CheckerMetadata("1.0", "cc-plugin-wcrp", "2.3.4.dev3+gc324abc")
    }

    assert format_checker_version("wcrp_cmip7", metadata) == (
        "CMIP7 wcrp_cmip7:1.0 (cc-plugin-wcrp 2.3.4.dev3+gc324abc)"
    )


def test_format_checker_version_includes_cf_package(monkeypatch):
    monkeypatch.setitem(checker_dict, "cf", "CF-Conventions")
    metadata = {
        "cf": CheckerMetadata("1.11", "compliance-checker", "6.1.1.dev69+gc4067cca7")
    }

    assert format_checker_version("cf", metadata) == (
        "CF-Conventions cf:1.11 (compliance-checker 6.1.1.dev69+gc4067cca7)"
    )


def test_track_checked_datasets(tmpdir):
    """
    Test the track_checked_datasets function.
    """
    # Create a temporary file
    checked_datasets_file = tmpdir.join("checked_datasets.csv")
    # Call the track_checked_datasets function
    checked_datasets = ["dataset1", "dataset2"]
    track_checked_datasets(str(checked_datasets_file), checked_datasets)
    # Check that the file was created and contains the expected data
    with open(checked_datasets_file) as file:
        reader = csv.reader(file)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == ["dataset1"]
        assert rows[1] == ["dataset2"]

    # Call the track_checked_datasets function again
    checked_datasets = ["dataset3"]
    track_checked_datasets(str(checked_datasets_file), checked_datasets)
    # Check that the file was updated and contains the expected data
    with open(checked_datasets_file) as file:
        reader = csv.reader(file)
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0] == ["dataset1"]
        assert rows[1] == ["dataset2"]
        assert rows[2] == ["dataset3"]


def test_verify_options_dict():
    """
    Test the _verify_options_dict function.
    """
    # Test case 1: empty options dictionary
    options = {}
    assert verify_options_dict(options) is True

    # Test case 2: options dictionary with one key-value pair
    options = {"checker_type": {"opt1": "value"}}
    assert verify_options_dict(options) is True

    # Test case 3: options dictionary with nested structure
    options = {
        "checker_type1": {"opt1": "value1", "opt2": 123},
        "checker_type2": {"opt1": "value2", "opt3": False},
    }
    assert verify_options_dict(options) is True

    # Test case 4: options dictionary with invalid value type
    options = {"checker_type": {"opt1": "value", "opt2": 123, "opt3": {}}}
    assert verify_options_dict(options) is False

    # Test case 5: options dictionary with non-dict value
    options = {"checker_type": "opt1"}
    assert verify_options_dict(options) is False
    options = {"checker_type": ["opt1", "opt2"]}
    assert verify_options_dict(options) is False

    # Test case 6: options dictionary with empty dict as value
    options = {"checker_type": {"opt1": {}}}
    assert verify_options_dict(options) is False


def test_parse_options():
    """Test the option parser"""
    # Simple test checker_type:checker_opt
    opt_dict = parse_options(["cf:enable_appendix_a_checks"])
    assert opt_dict == defaultdict(dict, {"cf": {"enable_appendix_a_checks": True}})
    assert verify_options_dict(opt_dict) is True
    # Test case checker_type:checker_opt:checker_val
    opt_dict = parse_options(
        ["type:opt:val", "type:opt2:val:2", "cf:enable_appendix_a_checks"],
    )
    assert opt_dict == defaultdict(
        dict,
        {
            "type": {"opt": "val", "opt2": "val:2"},
            "cf": {"enable_appendix_a_checks": True},
        },
    )
    assert verify_options_dict(opt_dict) is True


def test_latest_and_omitted_versions_are_equivalent_in_internal_specs():
    checkers_versions = {"cf": "latest", "cc6": "latest", "wcrp_cmip6": "1.7"}
    checkers = normalize_checker_specs(checkers_versions)

    assert "cf" in checkers
    assert "cc6" in checkers
    assert "wcrp_cmip6:1.7" in checkers
    assert "cf:latest" not in checkers
    assert "cc6:latest" not in checkers
