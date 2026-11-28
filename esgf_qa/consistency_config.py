"""Load and validate project-specific consistency-check configuration."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import toml

from esgf_qa._constants import checker_projects

SEVERITIES = {
    "H": 3,
    "HIGH": 3,
    "M": 2,
    "MEDIUM": 2,
    "L": 1,
    "LOW": 1,
}

SCOPES = ("within_dataset", "within_group", "across_all")
COMPARISON_CHECKS = {
    "required_global_attributes",
    "non_required_global_attributes",
    "global_attribute_types",
    "selected_global_attributes",
    "variable_attributes",
    "variable_attribute_types",
    "dimensions",
    "coordinates",
}
CHECKS_BY_SCOPE = {
    "within_dataset": COMPARISON_CHECKS
    | {
        "time_continuity",
        "compatibility_override",
        "compatibility_exact",
    },
    "within_group": COMPARISON_CHECKS | {"time_coverage"},
    "across_all": COMPARISON_CHECKS | {"time_coverage"},
}


class ConsistencyConfigError(ValueError):
    """Raised when a consistency-check TOML profile is invalid."""


@dataclass(frozen=True)
class GroupRule:
    """Attributes used to place datasets into one comparison group."""

    attributes: tuple[tuple[str, ...], ...]
    on_missing: str = "together"


@dataclass(frozen=True)
class CheckRule:
    """Resolved configuration for one consistency check."""

    enabled: bool = False
    severity: int = 2
    severity_name: str = "M"
    group: str | None = None
    allow_differences: tuple[str, ...] = ()
    compare_only: tuple[str, ...] = ()
    shared_non_scalar_only: bool = False
    compare_only_when_present: bool = False


@dataclass(frozen=True)
class ConsistencyConfig:
    """Validated and fully resolved consistency-check profile."""

    profile_name: str
    profile_version: str
    groups: dict[str, GroupRule]
    checks: dict[str, dict[str, CheckRule]]
    resolved: dict = field(repr=False, compare=False)

    def check(self, scope: str, name: str) -> CheckRule:
        """Return a configured rule or a disabled rule when it is absent."""
        return self.checks.get(scope, {}).get(name, CheckRule())

    def enabled(self, scope: str, name: str) -> bool:
        """Return whether one check is enabled."""
        return self.check(scope, name).enabled

    def any_enabled(self, scope: str, names=None) -> bool:
        """Return whether any configured check in a scope is enabled."""
        selected = self.checks.get(scope, {})
        if names is not None:
            selected = {
                name: rule
                for name, rule in selected.items()
                if name.split(":", 1)[0] in names
            }
        return any(rule.enabled for rule in selected.values())

    @property
    def requires_consistency_output(self) -> bool:
        """Return whether enabled checks consume per-file consistency metadata."""
        if self.enabled("within_dataset", "time_continuity"):
            return True
        if self.any_enabled("within_dataset", COMPARISON_CHECKS):
            return True
        if self.any_enabled("within_group"):
            return True
        return self.any_enabled("across_all", COMPARISON_CHECKS)

    @property
    def serialized(self) -> str:
        """Return the deterministic TOML representation archived with a run."""
        return toml.dumps(self.resolved)

    @property
    def sha256(self) -> str:
        """Return the hash of the resolved TOML representation."""
        return hashlib.sha256(self.serialized.encode("utf-8")).hexdigest()

    @property
    def metadata(self) -> dict[str, str]:
        """Return compact provenance stored in resume and result metadata."""
        return {
            "file": "consistency_config.toml",
            "profile": self.profile_name,
            "version": self.profile_version,
            "sha256": self.sha256,
        }


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _builtin_text(profile):
    resource = resources.files("esgf_qa").joinpath(
        "resources", "consistency", f"{profile}.toml"
    )
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConsistencyConfigError(
            f"Unknown built-in consistency profile '{profile}'."
        ) from error


def _parse_toml(text, source):
    try:
        data = toml.loads(text)
    except toml.TomlDecodeError as error:
        raise ConsistencyConfigError(
            f"Could not parse consistency configuration '{source}': {error}"
        ) from error
    if not isinstance(data, dict):
        raise ConsistencyConfigError(
            f"Consistency configuration '{source}' must contain a TOML table."
        )
    return data


def _read_profile(path=None, builtin=None):
    if path is not None:
        source = Path(path).expanduser()
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise ConsistencyConfigError(
                f"Could not read consistency configuration '{source}': {error}"
            ) from error
        return _parse_toml(text, source)
    profile = builtin or "default"
    return _parse_toml(_builtin_text(profile), f"built-in profile '{profile}'")


def _check_unknown_keys(data, allowed, context):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConsistencyConfigError(
            f"Unknown key(s) in {context}: {', '.join(unknown)}."
        )


def _string_list(value, context, allow_empty=True):
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ConsistencyConfigError(f"{context} must be a list of non-empty strings.")
    if not allow_empty and not value:
        raise ConsistencyConfigError(f"{context} must not be empty.")
    if len(value) != len(set(value)):
        raise ConsistencyConfigError(f"{context} must not contain duplicates.")
    return tuple(value)


def _validate_group(name, data):
    if not isinstance(name, str) or not name:
        raise ConsistencyConfigError("Group names must be non-empty strings.")
    if not isinstance(data, dict):
        raise ConsistencyConfigError(f"Group '{name}' must be a TOML table.")
    _check_unknown_keys(data, {"attributes", "on_missing"}, f"group '{name}'")
    attributes = data.get("attributes")
    if not isinstance(attributes, list) or not attributes:
        raise ConsistencyConfigError(
            f"Group '{name}' must define a non-empty 'attributes' list."
        )
    alternatives = []
    for index, values in enumerate(attributes):
        alternatives.append(
            _string_list(
                values,
                f"groups.{name}.attributes[{index}]",
                allow_empty=False,
            )
        )
    on_missing = data.get("on_missing", "together")
    if not isinstance(on_missing, str) or on_missing not in {
        "together",
        "separate",
        "error",
    }:
        raise ConsistencyConfigError(
            f"groups.{name}.on_missing must be 'together', 'separate', or 'error'."
        )
    return GroupRule(tuple(alternatives), on_missing)


def _validate_check(scope, name, data, groups):
    if name not in CHECKS_BY_SCOPE[scope]:
        raise ConsistencyConfigError(
            f"Unknown consistency check 'checks.{scope}.{name}'."
        )
    if not isinstance(data, dict):
        raise ConsistencyConfigError(
            f"Check 'checks.{scope}.{name}' must be a TOML table."
        )
    _check_unknown_keys(
        data,
        {
            "enabled",
            "severity",
            "group",
            "allow_differences",
            "compare_only",
            "shared_non_scalar_only",
            "compare_only_when_present",
        },
        f"check 'checks.{scope}.{name}'",
    )
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConsistencyConfigError(
            f"checks.{scope}.{name}.enabled must be true or false."
        )
    severity_name = data.get("severity", "M")
    if not isinstance(severity_name, str) or severity_name.upper() not in SEVERITIES:
        raise ConsistencyConfigError(
            f"checks.{scope}.{name}.severity must be H, M, L, HIGH, MEDIUM, or LOW."
        )
    severity_name = severity_name.upper()
    severity = SEVERITIES[severity_name]
    # Store the short spelling in resolved profiles so aliases such as "LOW"
    # and "L" produce identical archived configuration and provenance hashes.
    severity_name = {3: "H", 2: "M", 1: "L"}[severity]
    group = data.get("group")
    if scope == "within_group":
        if group is not None and (not isinstance(group, str) or not group):
            raise ConsistencyConfigError(
                f"checks.{scope}.{name}.group must be a non-empty string."
            )
        if enabled and group is None:
            raise ConsistencyConfigError(
                f"Enabled check 'checks.{scope}.{name}' must name a group."
            )
        if group is not None and group not in groups:
            raise ConsistencyConfigError(
                f"checks.{scope}.{name} references unknown group '{group}'."
            )
    elif group is not None:
        raise ConsistencyConfigError(
            f"checks.{scope}.{name}.group is only valid for within_group checks."
        )
    allow_differences = _string_list(
        data.get("allow_differences", []),
        f"checks.{scope}.{name}.allow_differences",
    )
    compare_only = _string_list(
        data.get("compare_only", []), f"checks.{scope}.{name}.compare_only"
    )
    if name not in COMPARISON_CHECKS and (allow_differences or compare_only):
        raise ConsistencyConfigError(
            f"checks.{scope}.{name} does not support attribute comparison options."
        )
    if compare_only and name != "selected_global_attributes":
        raise ConsistencyConfigError(
            f"checks.{scope}.{name}.compare_only is only supported by "
            "selected_global_attributes."
        )
    if name == "selected_global_attributes" and enabled and not compare_only:
        raise ConsistencyConfigError(
            f"checks.{scope}.{name} must define a non-empty compare_only list."
        )
    present_only = data.get("compare_only_when_present", False)
    if not isinstance(present_only, bool):
        raise ConsistencyConfigError("compare_only_when_present must be true or false.")
    if "compare_only_when_present" in data and (
        name != "selected_global_attributes" or scope == "within_dataset"
    ):
        raise ConsistencyConfigError(
            "compare_only_when_present is only supported for cross-dataset selected_global_attributes."
        )
    shared_non_scalar_only = data.get("shared_non_scalar_only", False)
    if not isinstance(shared_non_scalar_only, bool):
        raise ConsistencyConfigError("shared_non_scalar_only must be true or false.")
    if "shared_non_scalar_only" in data and (
        name != "coordinates" or scope == "within_dataset"
    ):
        raise ConsistencyConfigError(
            "shared_non_scalar_only is only supported for cross-dataset coordinates."
        )
    return CheckRule(
        enabled=enabled,
        severity=severity,
        severity_name=severity_name,
        group=group,
        allow_differences=allow_differences,
        compare_only=compare_only,
        shared_non_scalar_only=shared_non_scalar_only,
        compare_only_when_present=present_only,
    )


def _resolved_config_data(profile, groups, checks):
    """Return a canonical profile with every effective default made explicit."""
    resolved_groups = {
        name: {
            "attributes": [list(alternatives) for alternatives in rule.attributes],
            "on_missing": rule.on_missing,
        }
        for name, rule in sorted(groups.items())
    }
    resolved_checks = {}
    for scope in SCOPES:
        scope_checks = {}
        for key in sorted(checks[scope]):
            name, _, instance = key.partition(":")
            rule = checks[scope][key]
            data = {
                "enabled": rule.enabled,
                "severity": {3: "H", 2: "M", 1: "L"}[rule.severity],
            }
            if rule.group is not None:
                data["group"] = rule.group
            if name in COMPARISON_CHECKS:
                data["allow_differences"] = list(rule.allow_differences)
            if name == "selected_global_attributes":
                data["compare_only"] = list(rule.compare_only)
                if scope != "within_dataset":
                    data["compare_only_when_present"] = rule.compare_only_when_present
            if name == "coordinates" and scope != "within_dataset":
                data["shared_non_scalar_only"] = rule.shared_non_scalar_only
            if instance:
                scope_checks.setdefault(name, {}).setdefault("instances", {})[
                    instance
                ] = data
            else:
                scope_checks[name] = data
        resolved_checks[scope] = scope_checks
    return {
        "schema_version": 1,
        "profile": {
            "name": profile["name"],
            "version": profile["version"],
        },
        "groups": resolved_groups,
        "checks": resolved_checks,
    }


def _validate_config(data):
    _check_unknown_keys(data, {"schema_version", "profile", "groups", "checks"}, "root")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ConsistencyConfigError("schema_version must be 1.")

    profile = data.get("profile")
    if not isinstance(profile, dict):
        raise ConsistencyConfigError("A [profile] table is required.")
    _check_unknown_keys(profile, {"name", "version"}, "[profile]")
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise ConsistencyConfigError("profile.name must be a non-empty string.")
    if not isinstance(profile.get("version"), str) or not profile["version"]:
        raise ConsistencyConfigError("profile.version must be a non-empty string.")

    group_data = data.get("groups", {})
    if not isinstance(group_data, dict):
        raise ConsistencyConfigError("[groups] must contain named group tables.")
    groups = {name: _validate_group(name, value) for name, value in group_data.items()}

    checks_data = data.get("checks", {})
    if not isinstance(checks_data, dict):
        raise ConsistencyConfigError("[checks] must contain check-scope tables.")
    _check_unknown_keys(checks_data, SCOPES, "[checks]")
    configured_checks = {}
    for scope in SCOPES:
        scope_data = checks_data.get(scope, {})
        if not isinstance(scope_data, dict):
            raise ConsistencyConfigError(f"[checks.{scope}] must contain check tables.")
        configured_checks[scope] = {}
        for name, value in scope_data.items():
            if isinstance(value, dict) and "instances" in value:
                if name != "selected_global_attributes" or scope == "within_dataset":
                    raise ConsistencyConfigError(
                        "Named instances are supported for cross-dataset selected_global_attributes only."
                    )
                value = dict(value)
                instances = value.pop("instances")
                if not isinstance(instances, dict):
                    raise ConsistencyConfigError(
                        "instances must contain named check tables."
                    )
                for instance, instance_data in instances.items():
                    if not instance or ":" in instance:
                        raise ConsistencyConfigError(
                            "Instance names must be non-empty and contain no colon."
                        )
                    configured_checks[scope][f"{name}:{instance}"] = _validate_check(
                        scope, name, instance_data, groups
                    )
                if not value:
                    continue
            configured_checks[scope][name] = _validate_check(scope, name, value, groups)
    checks = {
        scope: {
            name: configured_checks[scope].get(name, CheckRule())
            for name in sorted(CHECKS_BY_SCOPE[scope] | configured_checks[scope].keys())
        }
        for scope in SCOPES
    }
    return ConsistencyConfig(
        profile_name=profile["name"],
        profile_version=profile["version"],
        groups=groups,
        checks=checks,
        resolved=_resolved_config_data(profile, groups, checks),
    )


def _resolve_profile(data):
    profile = data.get("profile")
    extends = profile.get("extends") if isinstance(profile, dict) else None
    if extends is None:
        resolved = deepcopy(data)
    else:
        if not isinstance(extends, str) or extends not in {
            "default",
            "cmip7",
            "cordex_cmip6",
        }:
            raise ConsistencyConfigError(
                "profile.extends supports 'default', 'cmip7', or 'cordex_cmip6'."
            )
        resolved = _deep_merge(_resolve_profile(_read_profile(builtin=extends)), data)
    resolved_profile = resolved.get("profile")
    if isinstance(resolved_profile, dict):
        resolved_profile.pop("extends", None)
    return resolved


def validate_project_checker_selection(checkers):
    """Return the selected major project, rejecting ambiguous combinations."""
    selected_projects = {}
    for checker_spec in checkers:
        checker = checker_spec.split(":", 1)[0]
        project = checker_projects.get(checker)
        if project is not None:
            selected_projects.setdefault(project, []).append(checker)
    if len(selected_projects) > 1:
        selections = ", ".join(
            f"{project} ({', '.join(sorted(project_checkers))})"
            for project, project_checkers in sorted(selected_projects.items())
        )
        raise ConsistencyConfigError(
            "ERROR: Cannot combine checkers for multiple major projects in one "
            f"QA run: {selections}. Select checkers for only one major project."
        )
    return next(iter(selected_projects), None)


def load_consistency_config(path=None, checkers=()):
    """Load a custom profile or select the built-in profile for the checkers."""
    selected_project = validate_project_checker_selection(checkers)
    if path is not None:
        data = _read_profile(path=path)
    else:
        builtin = {"CMIP7": "cmip7", "CORDEX-CMIP6": "cordex_cmip6"}.get(
            selected_project, "default"
        )
        data = _read_profile(builtin=builtin)
    return _validate_config(_resolve_profile(data))


def write_resolved_consistency_config(config, result_dir):
    """Archive the complete resolved profile and return its metadata."""
    path = Path(result_dir, "consistency_config.toml")
    path.write_text(config.serialized, encoding="utf-8")
    return config.metadata


def load_resolved_consistency_config(path, expected_sha256=None):
    """Load an archived resolved profile and optionally verify its hash."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConsistencyConfigError(
            f"Could not read consistency configuration '{source}': {error}"
        ) from error
    file_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and file_sha256 != expected_sha256:
        raise ConsistencyConfigError(
            f"Archived consistency configuration '{source}' does not match the "
            "hash stored in .resume_info."
        )
    data = _parse_toml(text, source)
    config = _validate_config(data)
    if config.sha256 != file_sha256:
        raise ConsistencyConfigError(
            f"Archived consistency configuration '{source}' is not in resolved form."
        )
    return config
