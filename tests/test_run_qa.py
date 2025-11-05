import csv
import os
import re
from collections import defaultdict

from esgf_qa._constants import (
    checker_dict,
    checker_dict_ext,
    checker_release_versions,
)
from esgf_qa.run_qa import (
    _verify_options_dict,
    get_checker_release_versions,
    get_default_result_dir,
    get_dsid,
    parse_options,
    track_checked_datasets,
)


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
    dsid = get_dsid(files_to_check_dict, dataset_files_map_ext, file_path, project_id)
    assert dsid == "my_project.drs.elements.until"


def test_get_checker_release_versions():
    """
    Test function get_checker_release_versions.

    Verifies that known checkers update the global checker_release_versions
    dictionary with the correct version values.
    """
    # reset globals
    checker_release_versions.clear()
    checker_dict.clear()
    checker_dict_ext.clear()

    # prepare minimal fake environment
    checker_dict.update({"cf": "", "cc6": "", "wcrp_cmip6": ""})
    checker_dict_ext.update({**checker_dict, "cons": ""})

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
