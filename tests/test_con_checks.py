import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from esgf_qa import con_checks as cc
from esgf_qa._constants import deltdic
from esgf_qa.cluster_results import QAResultAggregator
from esgf_qa.con_checks import (
    _group_datasets,
    _group_key,
    compare_dicts,
    compare_nested_dicts,
    printtimedelta,
    truncate_str,
)
from esgf_qa.consistency_config import GroupRule, load_consistency_config


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


@pytest.mark.parametrize(
    "data",
    [
        {"global_attributes": {}, "global_attributes_non_required": {}},
        {
            "global_attributes": {"realm": "unset"},
            "global_attributes_non_required": {},
        },
    ],
)
def test_group_missing_value_policies(data):
    together = GroupRule((("realm", "modeling_realm"),), "together")
    assert (
        _group_key("ds1", data, "project", together)[0]
        == _group_key("ds2", data, "project", together)[0]
    )

    separate = GroupRule((("realm", "modeling_realm"),), "separate")
    assert (
        _group_key("ds1", data, "project", separate)[0]
        != _group_key("ds2", data, "project", separate)[0]
    )

    error = GroupRule((("realm", "modeling_realm"),), "error")
    with pytest.raises(ValueError, match="required by group 'project'"):
        _group_key("ds1", data, "project", error)


def test_group_labels_disambiguate_values_containing_separator():
    group = GroupRule((("first",), ("second",)))
    consistency_data = {
        "ds1": {"global_attributes": {"first": "a/b", "second": "c"}},
        "ds2": {"global_attributes": {"first": "a", "second": "b/c"}},
    }

    groups, labels = _group_datasets(["ds1", "ds2"], consistency_data, "project", group)

    assert len(groups) == 2
    assert set(labels.values()) == {'["a/b", "c"]', '["a", "b/c"]'}


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
    return dict(temp_files)


@pytest.mark.parametrize(
    "different_grid,changed_lat", [(False, False), (False, True), (True, True)]
)
def test_cordex_shared_non_scalar_coordinates(tmp_path, different_grid, changed_lat):
    ds_map, files = {}, {}
    for index in range(2):
        path = str(tmp_path / f"ds{index}.nc")
        with cc.Dataset(path, "w") as dataset:
            dataset.createDimension("lat", 1)
            dataset.createVariable("lat", "f8", ("lat",))[:] = (
                index if changed_lat else 0
            )
            dataset.createVariable("height", "f8").assignValue(2 if index == 0 else 10)
        summary = tmp_path / f"ds{index}.json"
        data = {
            "global_attributes": {
                "domain_id": "EUR-12",
                "grid": f"grid-{index if different_grid else 0}",
            },
            "global_attributes_non_required": {},
            "global_attributes_dtypes": {},
            "dimensions": {"lat": 1},
            "coordinates": {
                "lat": str(index if changed_lat else 0),
                "height": str(index),
                f"only_in_{index}": "checksum",
            },
        }
        summary.write_text(json.dumps(data))
        ds_map[f"ds{index}"] = [path]
        files[path] = {"consistency_file": str(summary)}
    config = load_consistency_config(checkers=["cc6"])
    results, _ = cc.inter_dataset_consistency_checks(ds_map, files, config)
    findings = results["ds1"].get("Coordinates (Inter-Dataset)")
    assert bool(findings) == (changed_lat and not different_grid)
    if findings:
        assert all(
            "height" not in message and "only_in" not in message
            for message in findings["msgs"]
        )


@pytest.mark.parametrize(
    "shared_only,scalar_files,expected_difference",
    [
        (True, (0, 1), False),
        (True, (0,), False),
        (True, (1,), False),
        (True, (), True),
        (False, (0, 1), True),
    ],
)
def test_coordinate_comparison_scalar_policy(
    tmp_path, monkeypatch, shared_only, scalar_files, expected_difference
):
    """Scalars on either side are excluded only when explicitly configured."""
    from hashlib import md5

    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
schema_version = 1
[profile]
name = "coordinate-policy-test"
version = "1.0"
[groups.project]
attributes = [["domain_id"], ["grid"]]
on_missing = "separate"
[checks.within_group.coordinates]
enabled = true
severity = "M"
group = "project"
shared_non_scalar_only = {str(shared_only).lower()}
"""
    )
    ds_map, files = {}, {}
    for index in range(2):
        path = str(tmp_path / f"file{index}.nc")
        with cc.Dataset(path, "w") as dataset:
            dataset.createDimension("level", 1)
            dimensions = () if index in scalar_files else ("level",)
            coordinate = dataset.createVariable("height", "f8", dimensions)
            coordinate[...] = 2 if index == 0 else 10
            checksum = md5(coordinate[:].tobytes()).hexdigest()
        summary = tmp_path / f"file{index}.json"
        summary.write_text(
            json.dumps(
                {
                    "global_attributes": {"domain_id": "EUR-12", "grid": "same grid"},
                    "coordinates": {"height": checksum},
                }
            )
        )
        ds_map[f"ds{index}"] = [path]
        files[path] = {"consistency_file": str(summary)}

    # Without the new option, comparisons must still work from summaries alone.
    if not shared_only:

        def unexpected_metadata_read(*args, **kwargs):
            pytest.fail("Disabled scalar policy must not open NetCDF files")

        monkeypatch.setattr(cc, "Dataset", unexpected_metadata_read)

    results, _ = cc.inter_dataset_consistency_checks(
        ds_map, files, load_consistency_config(path=profile)
    )
    finding = results["ds1"].get("Coordinates (Inter-Dataset)")
    assert bool(finding) == expected_difference
    if finding:
        assert finding["weight"] == 2
        assert any("height" in message for message in finding["msgs"])


@pytest.mark.parametrize("scope", ["within_group", "across_all"])
@pytest.mark.parametrize(
    "names,ids,expected",
    [
        (("height", "height"), ("height2m", "height2m"), True),
        (("height", "height"), ("height2m", "height10m"), False),
        (("height", "altitude"), ("height2m", "height2m"), True),
        (("height", "height"), (None, None), False),
        (("height", "height"), ("height2m", None), False),
        (("height", "height"), (None, "height2m"), False),
    ],
)
def test_coordinate_ids_with_scalar_fallback(tmp_path, scope, names, ids, expected):
    """IDs enable scalar comparisons; missing IDs retain CORDEX's fallback."""
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
schema_version = 1
[profile]
name = "ids"
version = "1.0"
[groups.project]
attributes = [["grid"]]
on_missing = "separate"
[checks.{scope}.coordinates]
enabled = true
severity = "M"
{ 'group = "project"' if scope == 'within_group' else '' }
shared_non_scalar_only = true
"""
    )
    datasets, files = {}, {}
    # The first dataset deliberately lacks this coordinate: it must not
    # prevent the two later datasets from being compared with each other.
    for index in range(3):
        path = str(tmp_path / f"ds{index}.nc")
        data = {"global_attributes": {"grid": "same"}, "coordinates": {}}
        with cc.Dataset(path, "w") as dataset:
            if index:
                name, identity = names[index - 1], ids[index - 1]
                dataset.createVariable(name, "f8").assignValue(index)
                data["coordinates"][name] = str(index)
                if identity:
                    data["coordinate_ids"] = {name: identity}
        summary = tmp_path / f"ds{index}.json"
        summary.write_text(json.dumps(data))
        datasets[f"ds{index}"] = [path]
        files[path] = {"consistency_file": str(summary)}
    results, _ = cc.inter_dataset_consistency_checks(
        datasets, files, load_consistency_config(path=profile)
    )
    assert not results["ds1"].get("Coordinates (Inter-Dataset)")
    assert bool(results["ds2"].get("Coordinates (Inter-Dataset)")) == expected


@pytest.mark.parametrize(
    "case,expected_datasets",
    [
        ("different_meanings", set()),
        ("same_meaning", {"ds2"}),
        ("equal_values", set()),
        ("renamed_coordinate", {"ds2"}),
        ("different_groups", set()),
        ("excluded", set()),
        ("legacy_reference", {"ds2"}),
        ("legacy_current", {"ds2"}),
        ("late_coordinate", {"ds2"}),
    ],
)
def test_coordinate_identity_reference_matrix(
    tmp_path, monkeypatch, case, expected_datasets
):
    """Independent coordinate references must not mix meanings or grid groups."""
    profile = tmp_path / "identities.toml"
    profile.write_text(
        """
schema_version = 1
[profile]
name = "identities"
version = "1.0"
[groups.project]
attributes = [["grid_label"]]
on_missing = "separate"
[checks.within_group.coordinates]
enabled = true
severity = "M"
group = "project"
"""
        + ('allow_differences = ["sector"]\n' if case == "excluded" else "")
    )

    # The first dataset has soil pools; later datasets have basin labels.
    # A single reference for the whole group would miss the ds1/ds2 difference.
    records = [
        {
            "coordinates": {"sector": "soil-checksum"},
            "coordinate_ids": {"sector": "soilpools"},
        },
        {
            "coordinates": {"sector": "basin-checksum"},
            "coordinate_ids": {"sector": "basin"},
        },
        {
            "coordinates": {"sector": "changed-checksum"},
            "coordinate_ids": {"sector": "basin"},
        },
    ]
    if case == "different_meanings":
        records[2]["coordinate_ids"]["sector"] = "vegetationtype"
    elif case == "equal_values":
        records[2]["coordinates"]["sector"] = "basin-checksum"
    elif case == "renamed_coordinate":
        records[2] = {
            "coordinates": {"labels": "changed-checksum"},
            "coordinate_ids": {"labels": "basin"},
        }
    elif case in {"legacy_reference", "legacy_current", "late_coordinate"}:
        records[0] = {"coordinates": {}}
        if case == "legacy_reference":
            records[1].pop("coordinate_ids")
        elif case == "legacy_current":
            records[2].pop("coordinate_ids")

    datasets, files = {}, {}
    for index, record in enumerate(records):
        record["global_attributes"] = {
            "grid_label": (
                "other" if case == "different_groups" and index == 2 else "g142"
            )
        }
        summary = tmp_path / f"ds{index}.json"
        summary.write_text(json.dumps(record))
        path = str(tmp_path / f"ds{index}.nc")
        datasets[f"ds{index}"] = [path]
        files[path] = {"consistency_file": str(summary)}

    def unexpected_read(*args, **kwargs):
        pytest.fail(
            "Identity comparisons must work from summaries without NetCDF reads"
        )

    monkeypatch.setattr(cc, "Dataset", unexpected_read)
    results, _ = cc.inter_dataset_consistency_checks(
        datasets, files, load_consistency_config(path=profile)
    )
    findings = {
        dataset_id: result["Coordinates (Inter-Dataset)"]
        for dataset_id, result in results.items()
        if result.get("Coordinates (Inter-Dataset)")
    }
    assert set(findings) == expected_datasets
    for finding in findings.values():
        assert finding["weight"] == 2
        assert len(finding["msgs"]) == 1
        name = "labels" if case == "renamed_coordinate" else "sector"
        assert next(iter(finding["msgs"])).endswith(": " + name)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("different_ids", False),
        ("same_ids", True),
        ("renamed", True),
        ("legacy", True),
        ("padding", False),
        ("missing", False),
    ],
)
def test_dimension_id_comparison(tmp_path, case, expected):
    profile = tmp_path / "dimensions.toml"
    profile.write_text(
        """
schema_version = 1
[profile]
name = "dimensions"
version = "1.0"
[checks.across_all.dimensions]
enabled = true
severity = "M"
"""
    )
    records = [
        {"dimensions": {}},
        {
            "dimensions": {"type": 3, "label_width": 16},
            "dimension_ids": {"type": "basin"},
            "string_length_dimensions": ["label_width"],
        },
        {
            "dimensions": {"type": 5, "label_width": 32},
            "dimension_ids": {"type": "basin"},
            "string_length_dimensions": ["label_width"],
        },
    ]
    if case == "different_ids":
        records[2]["dimension_ids"]["type"] = "soilpools"
    elif case == "renamed":
        records[2]["dimensions"]["categories"] = records[2]["dimensions"].pop("type")
        records[2]["dimension_ids"] = {"categories": "basin"}
    elif case == "legacy":
        records[2].pop("dimension_ids")
    elif case == "padding":
        records[2]["dimensions"]["type"] = 3
    elif case == "missing":
        records[2] = {"dimensions": {}}
    datasets, files = {}, {}
    for index, record in enumerate(records):
        path = str(tmp_path / f"ds{index}.json")
        Path(path).write_text(json.dumps(record))
        datasets[f"ds{index}"] = [path]
        files[path] = {"consistency_file": path}
    results, _ = cc.inter_dataset_consistency_checks(
        datasets, files, load_consistency_config(path=profile)
    )
    assert not results["ds1"].get("Dimensions (Inter-Dataset)")
    finding = results["ds2"].get("Dimensions (Inter-Dataset)")
    assert bool(finding) == expected
    if finding:
        assert all("label_width" not in message for message in finding["msgs"])


@pytest.mark.parametrize("checker", ["wcrp_cmip7", "wcrp_cmip6", "wcrp_cordex_cmip6"])
@pytest.mark.parametrize(
    "areacella_types,expected",
    [
        # fx areacella has no external_variables, which is allowed.
        ({"Conventions": "str", "license": "str"}, False),
        # Other attributes are still flagged when missing or of another type.
        ({"external_variables": "str", "license": "str"}, True),
        ({"Conventions": "int", "license": "str"}, True),
    ],
    ids=["missing_external_variables", "missing_other", "changed_type"],
)
def test_inter_dataset_types_allow_missing_external_variables(
    tmp_path, checker, areacella_types, expected
):
    rule = load_consistency_config(checkers=[checker]).check(
        "across_all", "global_attribute_types"
    )
    assert rule.enabled and "external_variables" in rule.allow_differences
    # Restrict execution to the type rule with the shipped policy.
    profile = tmp_path / "types.toml"
    profile.write_text(
        f"""
schema_version = 1
[profile]
name = "types"
version = "1.0"
[checks.across_all.global_attribute_types]
enabled = true
severity = "M"
allow_differences = {json.dumps(list(rule.allow_differences))}
"""
    )
    records = {
        "tas": {"Conventions": "str", "external_variables": "str", "license": "str"},
        "pr": {"Conventions": "str", "external_variables": "str", "license": "str"},
        "areacella": areacella_types,
    }
    datasets, files = {}, {}
    for name, dtypes in records.items():
        path = str(tmp_path / f"{name}.json")
        Path(path).write_text(json.dumps({"global_attributes_dtypes": dtypes}))
        datasets[name] = [path]
        files[path] = {"consistency_file": path}

    results, _ = cc.inter_dataset_consistency_checks(
        datasets, files, load_consistency_config(path=profile)
    )

    # Which dataset is the reference is chosen automatically; check all.
    findings = [
        tests["Global attributes data types (Inter-Dataset)"]
        for tests in results.values()
        if tests.get("Global attributes data types (Inter-Dataset)")
    ]
    assert bool(findings) == expected
    for finding in findings:
        assert finding["weight"] == 2
        assert all(message.endswith(": Conventions") for message in finding["msgs"])


@pytest.mark.parametrize("checker", ["wcrp_cmip7", "wcrp_cmip6", "wcrp_cordex_cmip6"])
@pytest.mark.parametrize(
    "case,expected",
    [
        ("same_table", True),
        ("different_tables", False),
        ("missing_table", False),
        ("missing_info", False),
        ("equal_info", False),
    ],
)
def test_optional_table_info_group(tmp_path, checker, case, expected):
    config = load_consistency_config(checkers=[checker])
    # Restrict execution to the table rule while retaining its shipped policy.
    profile = tmp_path / "table.toml"
    profile.write_text(
        """
schema_version = 1
[profile]
name = "table"
version = "1.0"
[groups.table]
attributes = [["table_id"]]
on_missing = "separate"
[checks.within_group.selected_global_attributes.instances.table]
enabled = true
severity = "L"
group = "table"
compare_only = ["table_info"]
compare_only_when_present = true
"""
    )
    rule = config.check("within_group", "selected_global_attributes:table")
    assert rule.enabled and rule.severity == 1 and rule.group == "table"
    for name in (
        "required_global_attributes",
        "non_required_global_attributes",
        "global_attribute_types",
    ):
        assert "table_info" in config.check("across_all", name).allow_differences
    records = [
        {
            "table_id": "Amon"
        },  # No info in the first dataset must not hide later differences.
        {"table_id": "Amon", "table_info": "version one"},
        {"table_id": "Amon", "table_info": "version two"},
    ]
    if case == "different_tables":
        records[2]["table_id"] = "Omon"
    elif case == "missing_table":
        for record in records:
            record.pop("table_id")
    elif case == "missing_info":
        records[2].pop("table_info")
    elif case == "equal_info":
        records[2]["table_info"] = "version one"
    datasets, files = {}, {}
    for index, attributes in enumerate(records):
        path = str(tmp_path / f"ds{index}.json")
        Path(path).write_text(
            json.dumps({"global_attributes_non_required": attributes})
        )
        datasets[f"ds{index}"] = [path]
        files[path] = {"consistency_file": path}
    results, _ = cc.inter_dataset_consistency_checks(
        datasets, files, load_consistency_config(path=profile)
    )
    finding = results["ds2"].get(
        "Group-specific global attributes (Inter-Dataset) [table]"
    )
    assert bool(finding) == expected
    if finding:
        assert finding["weight"] == 1
        assert all(message.endswith(": table_info") for message in finding["msgs"])


@pytest.mark.parametrize("fail_open", [False, True])
def test_compatibility_suppresses_only_time_range_fallback(monkeypatch, fail_open):
    message = (
        "Unable to decode time axis into full numpy.datetime64[ns] objects, "
        "continuing using cftime.datetime objects instead, reason: dates out of range. "
        "To silence this warning use a coarser resolution 'time_unit' or specify 'use_cftime=True'."
    )
    calls = []

    @contextmanager
    def open_dataset(*args, **kwargs):
        calls.append(kwargs["compat"])
        warnings.warn(message, cc.xr.SerializationWarning)
        warnings.warn("Another serialization issue", cc.xr.SerializationWarning)
        warnings.warn(message, UserWarning)
        if fail_open:
            raise ValueError("Invalid time units")
        yield None

    monkeypatch.setattr(cc.xr, "open_mfdataset", open_dataset)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        results = cc.compatibility_checks("dataset", {"dataset": ["file.nc"]}, {}, {})
        # The suppression must not escape the compatibility checks.
        warnings.warn(message, cc.xr.SerializationWarning)
    assert calls == ["override", "no_conflicts"]
    assert len(recorded) == 5
    assert (
        sum(str(item.message) == "Another serialization issue" for item in recorded)
        == 2
    )
    assert sum(item.category is UserWarning for item in recorded) == 2
    assert str(recorded[-1].message) == message
    for result in results.values():
        if fail_open:
            assert result["msgs"] == {"Invalid time units": ["file.nc"]}
        else:
            assert not result.get("msgs")


class TestConChecks:
    @pytest.mark.parametrize("table_enabled", [True, False])
    def test_named_attribute_instances_run_independently(self, tmp_path, table_enabled):
        profile = tmp_path / "instances.toml"
        profile.write_text(
            f"""
schema_version = 1
[profile]
name = "instances"
version = "1.0"
[groups.grid]
attributes = [["grid_label"]]
on_missing = "separate"
[groups.table]
attributes = [["table_id"]]
on_missing = "separate"
[checks.within_group.selected_global_attributes]
severity = "H"
group = "grid"
compare_only = ["nominal_resolution"]
[checks.within_group.selected_global_attributes.instances.table]
enabled = {str(table_enabled).lower()}
severity = "L"
group = "table"
compare_only = ["table_info"]
compare_only_when_present = true
"""
        )
        datasets, files = {}, {}
        for index in range(3):
            attrs = {
                "grid_label": "g142",
                "table_id": "Amon",
                "nominal_resolution": str(index),
            }
            if index:
                attrs["table_info"] = f"version {index}"
            path = str(tmp_path / f"ds{index}.json")
            Path(path).write_text(json.dumps({"global_attributes": attrs}))
            datasets[f"ds{index}"] = [path]
            files[path] = {"consistency_file": path}
        results, _ = cc.inter_dataset_consistency_checks(
            datasets, files, load_consistency_config(path=profile)
        )
        unnamed = "Group-specific global attributes (Inter-Dataset)"
        named = unnamed + " [table]"
        assert results["ds2"][unnamed]["weight"] == 3
        assert not results["ds1"].get(named)
        assert bool(results["ds2"].get(named)) == table_enabled
        if table_enabled:
            assert results["ds2"][named]["weight"] == 1
            assert all(
                "table_info" in message for message in results["ds2"][named]["msgs"]
            )

    def test_string_width_changes_within_dataset_remain_strict(
        self, temp_files, ds_map, files_to_check_dict
    ):
        for index, path in enumerate(temp_files["ds1"]):
            data = json.loads(Path(path).read_text())
            data["dimensions"]["label_width"] = 16 + index
            data["string_length_dimensions"] = ["label_width"]
            Path(path).write_text(json.dumps(data))
        results = cc.consistency_checks(
            "ds1",
            ds_map,
            files_to_check_dict,
            load_consistency_config(checkers=["cc6"]),
        )
        assert any(
            "label_width" in message for message in results["Dimensions"]["msgs"]
        )

    def test_cordex_scalar_changes_within_dataset_still_fail(
        self, temp_files, ds_map, files_to_check_dict
    ):
        for index, path in enumerate(temp_files["ds1"]):
            with open(path) as stream:
                data = json.load(stream)
            data["coordinates"]["height"] = str(2 if index == 0 else 10)
            with open(path, "w") as stream:
                json.dump(data, stream)
        config = load_consistency_config(checkers=["cc6"])
        results = cc.consistency_checks("ds1", ds_map, files_to_check_dict, config)
        assert any("height" in message for message in results["Coordinates"]["msgs"])

    def test_consistency_checks(self, ds_map, files_to_check_dict):
        results = cc.consistency_checks("ds1", ds_map, files_to_check_dict, {})
        assert isinstance(results, dict)
        assert "Required global attributes" in results

    def test_consistency_checks_aggregate_multiple_severities(
        self, temp_files, ds_map, files_to_check_dict
    ):
        """Real consistency results retain Required and Suggested failures."""
        changed_file = temp_files["ds1"][1]
        with open(changed_file) as file:
            changed_data = json.load(file)
        changed_data["global_attributes"]["title"] = "different required value"
        changed_data["global_attributes_non_required"][
            "notes"
        ] = "different suggested value"
        with open(changed_file, "w") as file:
            json.dump(changed_data, file)

        results = cc.consistency_checks("ds1", ds_map, files_to_check_dict, {})
        aggregator = QAResultAggregator()
        aggregator.update_ds({"cons": results}, "ds1")

        assert any(
            name.endswith("Required global attributes")
            for name in aggregator.summary["fail"][3]
        )
        assert any(
            name.endswith("Non-required global attributes")
            for name in aggregator.summary["fail"][1]
        )

    def test_config_changes_severity_disables_check_and_allows_difference(
        self, tmp_path, temp_files, ds_map, files_to_check_dict
    ):
        changed_file = temp_files["ds1"][1]
        with open(changed_file) as stream:
            changed_data = json.load(stream)
        changed_data["global_attributes"].update(
            title="different", project_specific="allowed"
        )
        changed_data["global_attributes_non_required"]["notes"] = "different"
        with open(changed_file, "w") as stream:
            json.dump(changed_data, stream)

        profile = tmp_path / "configured.toml"
        profile.write_text(
            """
schema_version = 1
[profile]
name = "configured"
version = "1"
extends = "default"
[checks.within_dataset.required_global_attributes]
severity = "L"
allow_differences = ["creation_date", "history", "tracking_id", "project_specific"]
[checks.within_dataset.non_required_global_attributes]
enabled = false
"""
        )
        config = load_consistency_config(path=profile)

        results = cc.consistency_checks("ds1", ds_map, files_to_check_dict, config)

        assert results["Required global attributes"]["weight"] == 1
        assert any(
            "title" in message
            for message in results["Required global attributes"]["msgs"]
        )
        assert all(
            "project_specific" not in message
            for message in results["Required global attributes"]["msgs"]
        )
        assert "Non-required global attributes" not in results

    def test_disabled_compatibility_check_is_not_executed(
        self, monkeypatch, tmp_path, ds_map, files_to_check_dict
    ):
        profile = tmp_path / "configured.toml"
        profile.write_text(
            """
schema_version = 1
[profile]
name = "configured"
version = "1"
extends = "default"
[checks.within_dataset.compatibility_exact]
enabled = false
"""
        )
        config = load_consistency_config(path=profile)
        calls = []

        class Opened:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def open_mfdataset(*args, **kwargs):
            calls.append(kwargs)
            return Opened()

        monkeypatch.setattr(cc.xr, "open_mfdataset", open_mfdataset)

        results = cc.compatibility_checks("ds1", ds_map, files_to_check_dict, config)

        assert [call["compat"] for call in calls] == ["override"]
        assert list(results) == [
            "xarray open_mfdataset (compat='override', join='outer')"
        ]

    def test_continuity_checks(self, ds_map, files_to_check_dict):
        results = cc.continuity_checks("ds1", ds_map, files_to_check_dict, {})
        assert isinstance(results, dict)
        assert "Time continuity" in results

    def test_continuity_checks_accept_zero_as_a_time_value(
        self, temp_files, ds_map, files_to_check_dict
    ):
        first_file, second_file = temp_files["ds1"]
        for file, values in (
            (first_file, {"timen": 0, "boundn": 0}),
            (second_file, {"time0": 2, "bound0": 2}),
        ):
            with open(file) as stream:
                data = json.load(stream)
            data["time_info"].update(values)
            with open(file, "w") as stream:
                json.dump(data, stream)

        messages = cc.continuity_checks("ds1", ds_map, files_to_check_dict, {})[
            "Time continuity"
        ]["msgs"]

        assert any(message.startswith("Gap in time axis") for message in messages)
        assert any(message.startswith("Gap in time bounds") for message in messages)

    def test_continuity_checks_report_unsupported_frequency(
        self, temp_files, ds_map, files_to_check_dict
    ):
        second_file = temp_files["ds1"][1]
        with open(second_file) as stream:
            data = json.load(stream)
        data["time_info"].update(
            {"time0": 1, "timen": 2, "bound0": 1, "boundn": 2, "frequency": "fortnight"}
        )
        with open(second_file, "w") as stream:
            json.dump(data, stream)

        messages = cc.continuity_checks("ds1", ds_map, files_to_check_dict, {})[
            "Time continuity"
        ]["msgs"]

        assert messages["Unsupported frequency 'fortnight'"] == [second_file]

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

    def test_dataset_coverage_uses_first_file_start_month(self):
        ds_map = {"ds1": ["first.nc", "last.nc"], "ds2": ["other.nc"]}
        files_to_check_dict = {
            "first.nc": {"ts": "20000201-20000630"},
            "last.nc": {"ts": "20000701-20011231"},
            "other.nc": {"ts": "20010101-20011231"},
        }

        results = cc.dataset_coverage_checks(ds_map, files_to_check_dict, {})

        assert not any(
            "Time series starts" in message
            for dataset_results in results.values()
            for message in dataset_results["Time coverage"]["msgs"]
        )

    @pytest.mark.parametrize("single_label", [False, True])
    @pytest.mark.parametrize("label", ["2015", "201501", "20150101", "201501010000"])
    def test_single_timestamp_coverage_keeps_its_year(self, label, single_label):
        ds_map = {"single": ["single.nc"], "monthly": ["monthly.nc"]}
        files = {
            "single.nc": {"ts": label if single_label else f"{label}-{label}"},
            "monthly.nc": {"ts": "201501-201512"},
        }
        assert not cc.dataset_coverage_checks(ds_map, files, {})

    @pytest.mark.parametrize("split_files", [False, True])
    def test_january_endpoint_still_matches_previous_monthly_year(self, split_files):
        subdaily = ["a.nc", "b.nc"] if split_files else ["a.nc"]
        ds_map = {"subdaily": subdaily, "monthly": ["monthly.nc"]}
        files = {
            "monthly.nc": {"ts": "201401-201412"},
            "a.nc": {"ts": "201401010000-201501010000"},
        }
        if split_files:
            files["a.nc"]["ts"] = "201401010000-201412312300"
            # The last file alone is a single point, but the dataset is not.
            files["b.nc"] = {"ts": "201501010000-201501010000"}
        assert not cc.dataset_coverage_checks(ds_map, files, {})

    @pytest.mark.parametrize("timestamp", ["abcd0101-20001231", "abcd0101", "201501-"])
    def test_dataset_coverage_reports_malformed_timestamp(self, timestamp):
        ds_map = {"ds1": ["broken.nc"]}
        files_to_check_dict = {"broken.nc": {"ts": timestamp}}

        results = cc.dataset_coverage_checks(ds_map, files_to_check_dict, {})

        assert results["ds1"]["Time coverage"]["msgs"] == {
            "Time coverage cannot be inferred.": ["broken.nc"]
        }

    def test_inter_dataset_consistency_checks(self, ds_map, files_to_check_dict):
        results, ref_ds = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, {}
        )
        assert isinstance(results, dict)
        assert isinstance(ref_ds, dict)
        assert "general_reference" in ref_ds

    def test_cmip7_grouping_uses_branding_suffix_grid_label_and_region(
        self, temp_files, ds_map, files_to_check_dict
    ):
        for dataset_id, branding_suffix in (("ds1", "brand-a"), ("ds2", "brand-b")):
            reference_file = temp_files[dataset_id][0]
            with open(reference_file) as stream:
                data = json.load(stream)
            data["global_attributes"].update(
                branding_suffix=branding_suffix,
                grid_label="gn",
                region="glb",
            )
            if dataset_id == "ds2":
                data["dimensions"]["lat"] = 6
            with open(reference_file, "w") as stream:
                json.dump(data, stream)

        config = load_consistency_config(checkers=["wcrp_cmip7"])
        results, references = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, config
        )

        assert "Dimensions (Inter-Dataset)" not in results["ds2"]
        assert references["group_references"]["project"] == {
            "brand-a/gn/glb": "ds1",
            "brand-b/gn/glb": "ds2",
        }

        second_reference = temp_files["ds2"][0]
        with open(second_reference) as stream:
            data = json.load(stream)
        data["global_attributes"].update(branding_suffix="brand-a", region="30S-90S")
        with open(second_reference, "w") as stream:
            json.dump(data, stream)

        results, references = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, config
        )

        assert "Dimensions (Inter-Dataset)" not in results["ds2"]
        assert references["group_references"]["project"] == {
            "brand-a/gn/glb": "ds1",
            "brand-a/gn/30S-90S": "ds2",
        }

        data["global_attributes"]["region"] = "glb"
        with open(second_reference, "w") as stream:
            json.dump(data, stream)

        results, references = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, config
        )

        assert results["ds2"]["Dimensions (Inter-Dataset)"]["weight"] == 2
        assert references["group_references"]["project"] == {"brand-a/gn/glb": "ds1"}

    def test_time_coverage_can_be_limited_to_configured_groups(
        self, tmp_path, temp_files, ds_map, files_to_check_dict
    ):
        for dataset_id, branding_suffix in (("ds1", "brand-a"), ("ds2", "brand-b")):
            reference_file = temp_files[dataset_id][0]
            with open(reference_file) as stream:
                data = json.load(stream)
            data["global_attributes"]["branding_suffix"] = branding_suffix
            with open(reference_file, "w") as stream:
                json.dump(data, stream)
        for file_path in ds_map["ds2"]:
            files_to_check_dict[file_path]["ts"] = "20010101-20011231"

        profile = tmp_path / "grouped-coverage.toml"
        profile.write_text(
            """
schema_version = 1
[profile]
name = "grouped-coverage"
version = "1"
extends = "default"
[groups.project]
attributes = [["branding_suffix"]]
on_missing = "separate"
[checks.across_all.time_coverage]
enabled = false
[checks.within_group.time_coverage]
enabled = true
severity = "H"
group = "project"
"""
        )
        config = load_consistency_config(path=profile)

        results = cc.dataset_coverage_checks(ds_map, files_to_check_dict, config)
        assert results == {}

        second_reference = temp_files["ds2"][0]
        with open(second_reference) as stream:
            data = json.load(stream)
        data["global_attributes"]["branding_suffix"] = "brand-a"
        with open(second_reference, "w") as stream:
            json.dump(data, stream)

        results = cc.dataset_coverage_checks(ds_map, files_to_check_dict, config)
        assert results["ds2"]["Time coverage"]["weight"] == 3

    @pytest.mark.parametrize("frequency", ["dec", "cen"])
    def test_long_frequency_tolerances_enclose_nominal_delta(self, frequency):
        assert deltdic[f"{frequency}min"] <= deltdic[frequency]
        assert deltdic[frequency] <= deltdic[f"{frequency}max"]

    def test_missing_consistency_outputs_are_skipped(
        self, tmp_path, ds_map, files_to_check_dict
    ):
        """Consistency checks continue when their expected artifacts are absent."""
        for file in files_to_check_dict:
            files_to_check_dict[file]["consistency_file"] = str(
                tmp_path / f"missing-{os.path.basename(file)}"
            )

        assert cc.consistency_checks("ds1", ds_map, files_to_check_dict, {}) == {}
        continuity_results = cc.continuity_checks(
            "ds1", ds_map, files_to_check_dict, {}
        )
        assert continuity_results["Time continuity"]["msgs"] == {}
        inter_results, reference_datasets = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, {}
        )
        assert inter_results == {}
        assert reference_datasets == {}

    def test_replacement_reference_file_is_reported(
        self, tmp_path, ds_map, files_to_check_dict
    ):
        """Only deviations from the default reference selection are reported."""
        default_reference = sorted(ds_map["ds1"])[0]
        replacement_reference = sorted(ds_map["ds1"])[1]
        files_to_check_dict[default_reference]["consistency_file"] = str(
            tmp_path / "missing-reference.json"
        )

        _, reference_datasets = cc.inter_dataset_consistency_checks(
            ds_map, files_to_check_dict, {}
        )

        assert reference_datasets["general_reference"] == "ds1"
        assert reference_datasets["reference_file_exceptions"] == {
            "ds1": {
                "default_reference_file": default_reference,
                "used_reference_file": replacement_reference,
            }
        }
