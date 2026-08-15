"""Public entry point and high-level orchestration for ``esgqa``."""

import datetime
import hashlib
import os

from esgf_qa.checker_registry import (
    CheckerMetadata,
    format_checker_version,
    get_checker_metadata,
    get_checker_release_versions,
    get_installed_checker_versions,
    normalize_checker_specs,
)
from esgf_qa.cli import RunConfig, parse_options, prepare_run
from esgf_qa.discovery import (
    FileInventory,
    discover_files,
    format_exclusion_counts,
    get_dsid,
    write_excluded_files,
    write_inventory,
)
from esgf_qa.resume import (
    _get_reusable_file_result,
    _invalidate_nonreusable_dataset_results,
    _verify_options_dict,
    reconcile_resume_inventory,
    track_checked_datasets,
)
from esgf_qa.workers import (
    DATASET_CHECKERS,
    _dataset_check_runtime_error,
    _format_dataset_check_runtime_error,
    call_process_dataset,
    call_process_file,
    process_dataset,
    process_file,
    run_compliance_checker,
    run_dataset_collection_check,
)
from esgf_qa.workflow import run_workflow, write_results

_timestamp_with_ms = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
_timestamp_filename = datetime.datetime.strptime(
    _timestamp_with_ms, "%Y%m%d-%H%M%S%f"
).strftime("%Y%m%d-%H%M")
_timestamp_pprint = datetime.datetime.strptime(
    _timestamp_with_ms, "%Y%m%d-%H%M%S%f"
).strftime("%Y-%m-%d %H:%M")


def get_default_result_dir():
    """Return the timestamped default result directory."""
    result_hash = hashlib.md5(_timestamp_with_ms.encode()).hexdigest()
    return os.path.abspath(f"esgf-qa-results_{_timestamp_filename}_{result_hash}")


def main(argv=None):
    """Run the complete QA command-line workflow."""
    config = prepare_run(get_default_result_dir(), argv)
    inventory = discover_files(config)
    excluded_report = write_excluded_files(inventory, config)
    reconcile_resume_inventory(inventory, config)
    if not inventory.files:
        if inventory.discovered_file_count == 0:
            raise FileNotFoundError(
                f"No NetCDF files found under '{config.parent_dir}'."
            )
        raise RuntimeError(
            "No files remain to check: "
            f"{inventory.discovered_file_count} NetCDF files were discovered, "
            f"{format_exclusion_counts(inventory, config)}. "
            f"See '{excluded_report}'."
        )
    write_inventory(inventory, config.result_dir)
    summary, reference_datasets = run_workflow(config, inventory)
    write_results(
        config,
        inventory,
        summary,
        reference_datasets,
        _timestamp_with_ms,
        _timestamp_filename,
        _timestamp_pprint,
    )


__all__ = [
    "CheckerMetadata",
    "DATASET_CHECKERS",
    "FileInventory",
    "RunConfig",
    "_dataset_check_runtime_error",
    "_format_dataset_check_runtime_error",
    "_get_reusable_file_result",
    "_invalidate_nonreusable_dataset_results",
    "_verify_options_dict",
    "call_process_dataset",
    "call_process_file",
    "format_checker_version",
    "get_checker_metadata",
    "get_checker_release_versions",
    "get_default_result_dir",
    "get_dsid",
    "get_installed_checker_versions",
    "main",
    "normalize_checker_specs",
    "parse_options",
    "process_dataset",
    "process_file",
    "reconcile_resume_inventory",
    "run_compliance_checker",
    "run_dataset_collection_check",
    "track_checked_datasets",
]


if __name__ == "__main__":
    main()
