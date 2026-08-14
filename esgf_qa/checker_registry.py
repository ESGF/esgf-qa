"""Compliance Checker discovery and provenance helpers."""

from dataclasses import dataclass
from importlib.metadata import entry_points

from compliance_checker.runner import CheckSuite
from packaging import version as pversion

from esgf_qa._constants import checker_dict, checker_dict_ext
from esgf_qa._version import version as esgf_qa_version


@dataclass(frozen=True)
class CheckerMetadata:
    """Resolved version and package provenance for one checker."""

    checker_version: str
    package_name: str | None = None
    package_version: str | None = None


def get_installed_checker_versions():
    """Return installed checker versions grouped by checker name."""
    check_suite = CheckSuite()
    check_suite.load_all_available_checkers()
    installed_versions = {}
    for checker in check_suite.checkers:
        try:
            name, checker_version = checker.split(":")
        except ValueError:
            name, checker_version = checker, "latest"
        if checker_version == "latest":
            continue
        installed_versions.setdefault(name, []).append(checker_version)
    for name, versions in installed_versions.items():
        installed_versions[name] = sorted(versions, key=pversion.parse) + ["latest"]
    return installed_versions


def get_checker_metadata(checkers, checker_options=None):
    """Resolve checker versions and their providing distributions."""
    check_suite = CheckSuite(options=checker_options or {})
    check_suite.load_all_available_checkers()

    checker_packages = {}
    for checker_entry_point in entry_points(group="compliance_checker.suites"):
        try:
            checker_obj = checker_entry_point.load()
        except Exception:
            continue
        checker_name = getattr(checker_obj, "_cc_spec", None) or getattr(
            checker_obj, "name", None
        )
        checker_version = str(
            getattr(checker_obj, "_cc_spec_version", "unknown")
        )
        distribution = getattr(checker_entry_point, "dist", None)
        if checker_name and distribution is not None:
            checker_packages[(checker_name, checker_version)] = (
                distribution.name,
                distribution.version,
            )

    metadata = {}
    for checker in checkers:
        checker_name = checker.split(":", 1)[0]
        if checker_name in checker_dict_ext and checker_name not in checker_dict:
            metadata[checker_name] = CheckerMetadata(
                esgf_qa_version, "esgf-qa", esgf_qa_version
            )
            continue

        checker_obj = check_suite.checkers.get(checker)
        if checker_obj is None:
            prefix = checker_name + ":"
            candidates = [key for key in check_suite.checkers if key.startswith(prefix)]
            if candidates:
                resolved_key = max(
                    candidates,
                    key=lambda key: pversion.parse(key.split(":", 1)[1]),
                )
                checker_obj = check_suite.checkers.get(resolved_key)
        checker_version = (
            str(checker_obj._cc_spec_version)
            if checker_obj is not None
            else "unknown"
        )
        package = checker_packages.get((checker_name, checker_version))
        metadata[checker_name] = CheckerMetadata(
            checker_version,
            package[0] if package else None,
            package[1] if package else None,
        )
    return metadata


def format_checker_version(checker, metadata):
    """Format a checker specification with its providing package version."""
    checker_name = checker.split(":", 1)[0]
    checker_metadata = metadata[checker_name]
    checker_label = (
        f"{checker_dict.get(checker_name, '')} "
        f"{checker_name}:{checker_metadata.checker_version}"
    ).strip()
    if checker_metadata.package_name is not None:
        checker_label += (
            f" ({checker_metadata.package_name} "
            f"{checker_metadata.package_version})"
        )
    return checker_label


def normalize_checker_specs(checkers_versions):
    """Normalize latest versions to Compliance Checker's unversioned form."""
    return sorted(
        checker if checker_version == "latest" else f"{checker}:{checker_version}"
        for checker, checker_version in checkers_versions.items()
    )


# Compatibility name for callers that used the old helper. Unlike the former
# implementation this returns metadata instead of mutating module-level state.
get_checker_release_versions = get_checker_metadata
