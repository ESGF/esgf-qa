import json

import pytest

from esgf_qa import cli
from esgf_qa.consistency_config import (
    ConsistencyConfigError,
    load_consistency_config,
    load_resolved_consistency_config,
    write_resolved_consistency_config,
)


def write_profile(tmp_path, contents):
    path = tmp_path / "profile.toml"
    path.write_text(contents)
    return path


def test_default_profile_preserves_existing_rules():
    config = load_consistency_config(checkers=["wcrp_cmip6"])

    assert config.profile_name == "default"
    assert config.check("within_dataset", "required_global_attributes").severity == 3
    assert (
        config.check("within_dataset", "non_required_global_attributes").severity == 1
    )
    assert config.check("within_dataset", "dimensions").allow_differences == ("time",)
    assert config.check("within_group", "dimensions").group == "project"
    assert config.check("across_all", "time_coverage").enabled is True


def test_named_attribute_instances_inheritance_and_roundtrip(tmp_path):
    path = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "custom"
version = "1.0"
extends = "cmip7"
[checks.within_group.selected_global_attributes.instances.table]
enabled = false
[checks.across_all.selected_global_attributes.instances.model]
severity = "H"
compare_only = ["source_id"]
compare_only_when_present = true
""",
    )
    config = load_consistency_config(path=path)
    assert config.enabled("within_group", "selected_global_attributes")
    assert not config.enabled("within_group", "selected_global_attributes:table")
    assert config.check(
        "within_group", "selected_global_attributes:table"
    ).compare_only == ("table_info",)
    assert config.any_enabled("across_all", {"selected_global_attributes"})
    assert config.requires_consistency_output
    archived = tmp_path / "resolved.json"
    archived.write_text(config.serialized)
    restored = load_resolved_consistency_config(archived)
    assert restored.serialized == config.serialized
    assert (
        restored.check("across_all", "selected_global_attributes:model").severity == 3
    )


@pytest.mark.parametrize("scope", ["within_group", "across_all"])
def test_named_instance_alone_requires_summary_and_survives_reload(tmp_path, scope):
    group_setting = 'group = "table"' if scope == "within_group" else ""
    path = write_profile(
        tmp_path,
        f"""
schema_version = 1
[profile]
name = "named-only"
version = "1.0"
[groups.table]
attributes = [["table_id"]]
[checks.{scope}.selected_global_attributes.instances.table]
{group_setting}
compare_only = ["table_info"]
compare_only_when_present = true
""",
    )
    config = load_consistency_config(path=path)
    assert not config.enabled(scope, "selected_global_attributes")
    assert config.any_enabled(scope, {"selected_global_attributes"})
    assert config.requires_consistency_output
    archived = tmp_path / "resolved.json"
    archived.write_text(config.serialized)
    assert load_resolved_consistency_config(archived).serialized == config.serialized


@pytest.mark.parametrize(
    "section,settings",
    [
        (
            "within_dataset.selected_global_attributes.instances.bad",
            'compare_only = ["x"]',
        ),
        ("across_all.dimensions.instances.bad", 'severity = "M"'),
        (
            "across_all.selected_global_attributes",
            'compare_only = ["x"]\ncompare_only_when_present = "yes"',
        ),
        ("across_all.dimensions", "compare_only_when_present = true"),
    ],
)
def test_invalid_named_attribute_configuration(tmp_path, section, settings):
    path = write_profile(
        tmp_path,
        f"""
schema_version = 1
[profile]
name = "invalid"
version = "1.0"
[checks.{section}]
{settings}
""",
    )
    with pytest.raises(ConsistencyConfigError):
        load_consistency_config(path=path)


def test_cmip7_profile_groups_by_branding_suffix_grid_label_and_region():
    config = load_consistency_config(checkers=["cf", "wcrp_cmip7:1.0"])

    assert config.profile_name == "CMIP7"
    assert config.groups["project"].attributes == (
        ("branding_suffix",),
        ("grid_label",),
        ("region",),
    )
    assert config.groups["project"].on_missing == "separate"
    assert config.check("within_group", "selected_global_attributes").compare_only == (
        "branding_suffix",
        "grid_label",
        "region",
        "nominal_resolution",
    )
    for check in ("required_global_attributes", "non_required_global_attributes"):
        allowed = config.check("across_all", check).allow_differences
        for attribute in ("branding_suffix", "grid_label", "region"):
            assert attribute in allowed


@pytest.mark.parametrize(
    "checkers",
    [
        ["cc6"],
        ["wcrp_cordex_cmip6:1.0"],
        ["cf", "cc6", "wcrp_cordex_cmip6"],
    ],
)
def test_cordex_profile_selection_and_rules(checkers):
    config = load_consistency_config(checkers=checkers)
    assert config.profile_name == "CORDEX-CMIP6"
    assert config.groups["project"].attributes == (
        ("domain_id",),
        ("grid",),
    )
    assert config.groups["project"].on_missing == "separate"
    assert (
        "grid"
        in config.check("within_group", "selected_global_attributes").compare_only
    )
    assert config.check("within_group", "coordinates").shared_non_scalar_only
    assert not config.check("within_dataset", "coordinates").shared_non_scalar_only
    for scope in ("required_global_attributes", "non_required_global_attributes"):
        assert "domain_id" in config.check("across_all", scope).allow_differences
        assert (
            "domain_id" not in config.check("within_dataset", scope).allow_differences
        )
    assert config.check("within_dataset", "dimensions").severity == 3
    assert config.check("within_group", "coordinates").severity == 2
    assert config.check("across_all", "time_coverage").severity == 1


def test_custom_profile_extends_cordex(tmp_path):
    path = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "custom-cordex"
version = "1"
extends = "cordex_cmip6"
[checks.within_group.coordinates]
severity = "H"
""",
    )
    config = load_consistency_config(path=path, checkers=["cc6"])
    assert config.profile_name == "custom-cordex"
    assert config.groups["project"].attributes[0] == ("domain_id",)
    assert config.check("within_group", "coordinates").severity == 3


def test_profile_selection_rejects_multiple_major_projects():
    with pytest.raises(ConsistencyConfigError, match="multiple major projects"):
        load_consistency_config(checkers=["wcrp_cmip6", "wcrp_cmip7:1.0"])


@pytest.mark.parametrize(
    "check", ["required_global_attributes", "non_required_global_attributes"]
)
def test_cmip7_variable_identity_may_differ_across_datasets(check):
    config = load_consistency_config(checkers=["wcrp_cmip7"])
    attributes = {
        "branded_variable",
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
        "nominal_resolution",
    }
    assert attributes <= set(config.check("across_all", check).allow_differences)
    assert not attributes.intersection(
        config.check("within_dataset", check).allow_differences
    )


def test_custom_profile_extends_default_and_overrides_rules(tmp_path):
    profile = write_profile(
        tmp_path,
        """
schema_version = 1

[profile]
name = "test-project"
version = "2"
extends = "default"

[checks.within_dataset.required_global_attributes]
severity = "LOW"
allow_differences = ["history", "project_specific"]

[checks.within_dataset.compatibility_exact]
enabled = false
""",
    )

    config = load_consistency_config(path=profile)

    assert config.profile_name == "test-project"
    assert config.check("within_dataset", "required_global_attributes").severity == 1
    assert (
        config.check("within_dataset", "required_global_attributes").severity_name
        == "L"
    )
    assert config.check(
        "within_dataset", "required_global_attributes"
    ).allow_differences == ("history", "project_specific")
    assert config.check("within_dataset", "compatibility_exact").enabled is False
    assert config.check("within_dataset", "compatibility_override").enabled is True
    assert "extends" not in config.resolved["profile"]


def test_custom_profile_can_refine_cmip7(tmp_path):
    profile = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "cmip7-trial"
version = "1"
extends = "cmip7"
[groups.project]
attributes = [["branding_suffix"], ["grid_label"], ["region"], ["nominal_resolution"]]
""",
    )

    config = load_consistency_config(path=profile)

    assert config.groups["project"].attributes == (
        ("branding_suffix",),
        ("grid_label",),
        ("region",),
        ("nominal_resolution",),
    )
    assert config.groups["project"].on_missing == "separate"


def test_profile_with_only_compatibility_checks_needs_no_consistency_output(tmp_path):
    profile = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "compatibility-only"
version = "1"
[checks.within_dataset.compatibility_override]
enabled = true
severity = "H"
""",
    )

    config = load_consistency_config(path=profile)

    assert config.requires_consistency_output is False


def test_resolved_profile_makes_implicit_defaults_explicit(tmp_path):
    profile = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "minimal"
version = "1"
[groups.project]
attributes = [["realm"]]
[checks.within_dataset.dimensions]
""",
    )

    config = load_consistency_config(path=profile)

    assert config.resolved["groups"]["project"]["on_missing"] == "together"
    assert config.resolved["checks"]["within_dataset"]["dimensions"] == {
        "enabled": True,
        "severity": "M",
        "allow_differences": [],
    }
    assert config.resolved["checks"]["within_dataset"]["time_continuity"] == {
        "enabled": False,
        "severity": "M",
    }
    assert set(config.resolved["checks"]) == {
        "within_dataset",
        "within_group",
        "across_all",
    }


@pytest.mark.parametrize(
    "body, match",
    [
        ("schema_version = 2\n[profile]\nname='x'\nversion='1'", "schema_version"),
        ("schema_version = true\n[profile]\nname='x'\nversion='1'", "schema_version"),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[checks.within_dataset.unknown] \nseverity='H'",
            "Unknown consistency check",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[checks.within_dataset.dimensions] \nseverity='critical'",
            "severity",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[checks.within_group.dimensions] \nseverity='H'\ngroup='missing'",
            "unknown group",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[groups.project]\nattributes=[['realm']]\non_missing='guess'",
            "on_missing",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[groups.project]\nattributes=[[]]",
            "must not be empty",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\n"
            "[groups.project]\nattributes=[['realm']]\non_missing=[]",
            "on_missing",
        ),
        (
            "schema_version = 1\n[profile]\nname='x'\nversion='1'\nextends=[]",
            "profile.extends",
        ),
    ],
)
def test_invalid_profiles_are_rejected(tmp_path, body, match):
    profile = write_profile(tmp_path, body)

    with pytest.raises(ConsistencyConfigError, match=match):
        load_consistency_config(path=profile)


def test_resolved_profile_round_trip_and_hash_validation(tmp_path):
    config = load_consistency_config(checkers=["wcrp_cmip7"])
    metadata = write_resolved_consistency_config(config, tmp_path)
    archived = tmp_path / metadata["file"]

    restored = load_resolved_consistency_config(
        archived, expected_sha256=metadata["sha256"]
    )
    assert restored == config

    archived.write_text(
        archived.read_text().replace('version = "1.0"', 'version = "2"')
    )
    with pytest.raises(ConsistencyConfigError, match="does not match"):
        load_resolved_consistency_config(archived, expected_sha256=metadata["sha256"])


def test_severity_aliases_have_canonical_resolved_form(tmp_path):
    long_profile = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "severity-alias"
version = "1"
[checks.within_dataset.dimensions]
severity = "LOW"
""",
    )
    short_profile = tmp_path / "short.toml"
    short_profile.write_text(long_profile.read_text().replace('"LOW"', '"L"'))

    long_config = load_consistency_config(path=long_profile)
    short_config = load_consistency_config(path=short_profile)

    assert long_config == short_config
    assert long_config.serialized == short_config.serialized
    assert long_config.sha256 == short_config.sha256


def test_prepare_run_archives_and_restores_custom_profile(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    source = write_profile(
        tmp_path,
        """
schema_version = 1
[profile]
name = "custom"
version = "1"
extends = "default"
[checks.within_dataset.time_continuity]
severity = "L"
""",
    )
    monkeypatch.setattr(
        cli,
        "get_installed_checker_versions",
        lambda: {"wcrp_cmip7": ["1.0"]},
    )

    config = cli.prepare_run(
        str(tmp_path / "unused"),
        [
            "-t",
            "wcrp_cmip7:1.0",
            "--consistency-config",
            str(source),
            "-o",
            str(output_dir),
            str(input_dir),
        ],
    )
    resume_info = json.loads((output_dir / ".resume_info").read_text())

    assert config.consistency_config.profile_name == "custom"
    assert resume_info["consistency_config"] == config.consistency_config.metadata
    assert (output_dir / "consistency_config.toml").is_file()

    source.unlink()
    resumed = cli.prepare_run(str(tmp_path / "unused"), ["-r", "-o", str(output_dir)])
    assert resumed.consistency_config == config.consistency_config


def test_prepare_run_selects_cmip7_profile_automatically(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "get_installed_checker_versions",
        lambda: {"wcrp_cmip7": ["1.0"]},
    )

    config = cli.prepare_run(
        str(tmp_path / "unused"),
        [
            "-t",
            "wcrp_cmip7:1.0",
            "-o",
            str(tmp_path / "output"),
            str(input_dir),
        ],
    )

    assert config.consistency_config.profile_name == "CMIP7"


def test_resume_rejects_profile_metadata_mismatch(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        cli,
        "get_installed_checker_versions",
        lambda: {"wcrp_cmip7": ["1.0"]},
    )
    cli.prepare_run(
        str(tmp_path / "unused"),
        ["-t", "wcrp_cmip7:1.0", "-o", str(output_dir), str(input_dir)],
    )
    resume_path = output_dir / ".resume_info"
    resume_info = json.loads(resume_path.read_text())
    resume_info["consistency_config"]["profile"] = "not-cmip7"
    resume_path.write_text(json.dumps(resume_info))

    with pytest.raises(ValueError, match="name or version"):
        cli.prepare_run(str(tmp_path / "unused"), ["-r", "-o", str(output_dir)])
