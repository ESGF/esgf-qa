import csv
import json
import os
import re
import sys
from collections import defaultdict

import pytest

from esgf_qa import run_qa
from esgf_qa._constants import (
    checker_dict,
    checker_package_versions,
    checker_release_versions,
    checker_supporting_consistency_checks,
)
from esgf_qa.run_qa import (
    _invalidate_nonreusable_dataset_results,
    _verify_options_dict,
    format_checker_version,
    get_checker_release_versions,
    get_default_result_dir,
    get_dsid,
    normalize_checker_specs,
    parse_options,
    track_checked_datasets,
)


def test_main_enables_cf_appendix_a_checks(monkeypatch, tmp_path):
    """The main workflow passes Appendix A under Compliance Checker's CF key."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.nc").touch()
    output_dir = tmp_path / "output"
    captured_options = {}

    monkeypatch.setattr(
        run_qa, "get_installed_checker_versions", lambda: {"cf": ["latest"]}
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

    def set_checker_release_versions(checkers):
        monkeypatch.setitem(run_qa.checker_release_versions, "cf", "test")

    monkeypatch.setattr(run_qa, "process_file", capture_process_file)
    monkeypatch.setattr(
        run_qa, "get_checker_release_versions", set_checker_release_versions
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

    _invalidate_nonreusable_dataset_results(
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
        run_qa, "get_installed_checker_versions", lambda: {"cf": ["latest"]}
    )
    monkeypatch.setattr(
        run_qa,
        "process_file",
        lambda file_path, *args: (file_path, {"cf": {"errors": {}}}),
    )

    def set_checker_release_versions(checkers):
        monkeypatch.setitem(run_qa.checker_release_versions, "cf", "test")

    monkeypatch.setattr(
        run_qa, "get_checker_release_versions", set_checker_release_versions
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


@pytest.mark.parametrize(
    "option_args", [[], ["-O", "mip:tables"], ["-O", "mip:tables:"]]
)
def test_main_rejects_mip_without_table_path(monkeypatch, tmp_path, option_args):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        run_qa, "get_installed_checker_versions", lambda: {"mip": ["latest"]}
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
        run_qa, "get_installed_checker_versions", lambda: {"mip": ["latest"]}
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
        run_qa,
        "get_installed_checker_versions",
        lambda: {"mip": ["latest"], "wcrp_cmip6plus": ["1.0", "latest"]},
    )

    def capture_process_file(file_path, checkers, checker_options, *args):
        captured["checkers"] = checkers
        captured["options"] = checker_options
        return file_path, {checker: {"errors": {}}}

    def set_checker_release_versions(checkers):
        monkeypatch.setitem(run_qa.checker_release_versions, checker, "test")

    monkeypatch.setattr(run_qa, "process_file", capture_process_file)
    monkeypatch.setattr(
        run_qa, "get_checker_release_versions", set_checker_release_versions
    )
    monkeypatch.setattr(run_qa, "run_dataset_collection_check", lambda *args: None)
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


def test_get_checker_release_versions():
    """
    Test function get_checker_release_versions.

    Verifies that known checkers update the global checker_release_versions
    dictionary with the correct version values.
    """
    # reset globals
    checker_package_versions.clear()
    checker_release_versions.clear()

    # instantiate a real CheckSuite with empty options
    checkers = ["cf:1.6", "cc6:latest", "wcrp_cmip6:latest"]
    get_checker_release_versions(checkers)

    # check that the dictionary is filled correctly
    assert "cf" in checker_release_versions
    assert "cc6" in checker_release_versions
    assert "wcrp_cmip6" in checker_release_versions

    # ensure non-empty version strings (format check)
    for version in checker_release_versions.values():
        assert isinstance(version, str)
        assert len(version) > 0
    assert checker_release_versions["cf"] == "1.6"
    assert checker_package_versions["cf"][0] == "compliance-checker"
    assert checker_package_versions["cc6"][0] == "cc-plugin-cc6"


def test_format_checker_version_includes_providing_package(monkeypatch):
    monkeypatch.setitem(checker_dict, "wcrp_cmip7", "CMIP7")
    monkeypatch.setitem(checker_release_versions, "wcrp_cmip7", "1.0")
    monkeypatch.setitem(
        checker_package_versions,
        "wcrp_cmip7",
        ("cc-plugin-wcrp", "2.3.4.dev3+gc324abc"),
    )

    assert format_checker_version("wcrp_cmip7") == (
        "CMIP7 wcrp_cmip7:1.0 (cc-plugin-wcrp 2.3.4.dev3+gc324abc)"
    )


def test_format_checker_version_includes_cf_package(monkeypatch):
    monkeypatch.setitem(checker_dict, "cf", "CF-Conventions")
    monkeypatch.setitem(checker_release_versions, "cf", "1.11")
    monkeypatch.setitem(
        checker_package_versions,
        "cf",
        ("compliance-checker", "6.1.1.dev69+gc4067cca7"),
    )

    assert format_checker_version("cf") == (
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
    assert _verify_options_dict(options) is True

    # Test case 2: options dictionary with one key-value pair
    options = {"checker_type": {"opt1": "value"}}
    assert _verify_options_dict(options) is True

    # Test case 3: options dictionary with nested structure
    options = {
        "checker_type1": {"opt1": "value1", "opt2": 123},
        "checker_type2": {"opt1": "value2", "opt3": False},
    }
    assert _verify_options_dict(options) is True

    # Test case 4: options dictionary with invalid value type
    options = {"checker_type": {"opt1": "value", "opt2": 123, "opt3": {}}}
    assert _verify_options_dict(options) is False

    # Test case 5: options dictionary with non-dict value
    options = {"checker_type": "opt1"}
    assert _verify_options_dict(options) is False
    options = {"checker_type": ["opt1", "opt2"]}
    assert _verify_options_dict(options) is False

    # Test case 6: options dictionary with empty dict as value
    options = {"checker_type": {"opt1": {}}}
    assert _verify_options_dict(options) is False


def test_parse_options():
    """Test the option parser"""
    # Simple test checker_type:checker_opt
    opt_dict = parse_options(["cf:enable_appendix_a_checks"])
    assert opt_dict == defaultdict(dict, {"cf": {"enable_appendix_a_checks": True}})
    assert _verify_options_dict(opt_dict) is True
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
    assert _verify_options_dict(opt_dict) is True


def test_latest_and_omitted_versions_are_equivalent_in_internal_specs():
    checkers_versions = {"cf": "latest", "cc6": "latest", "wcrp_cmip6": "1.7"}
    checkers = normalize_checker_specs(checkers_versions)

    assert "cf" in checkers
    assert "cc6" in checkers
    assert "wcrp_cmip6:1.7" in checkers
    assert "cf:latest" not in checkers
    assert "cc6:latest" not in checkers
