"""Input-file discovery and dataset grouping."""

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

from esgf_qa._constants import (
    checker_supporting_consistency_checks,
    supported_project_ids,
)
from esgf_qa.resume import invalidate_nonreusable_dataset_results


@dataclass
class FileInventory:
    """Files and mappings consumed by the QA workflow."""

    files: list[str]
    file_details: dict
    dataset_files: dict
    directory_datasets: dict
    checker_options: dict
    discovered_file_count: int = 0
    blacklisted_files: dict[str, list[str]] = field(default_factory=dict)
    not_whitelisted_files: list[str] = field(default_factory=list)


def get_dsid(files_to_check_dict, dataset_files_map_ext, file_path, project_ids):
    """Build a dataset identifier from its directory and filename."""
    directory_parts = files_to_check_dict[file_path]["id_dir"].split("/")
    filename_parts = files_to_check_dict[file_path]["id_fn"].split("_")
    dataset_id = ".".join(directory_parts)
    lower_parts = [part.lower() for part in directory_parts]
    for project_id in project_ids:
        if project_id in lower_parts:
            last_index = len(lower_parts) - 1 - lower_parts[::-1].index(project_id)
            dataset_id = ".".join(directory_parts[last_index:])
            break
    directory = files_to_check_dict[file_path]["id_dir"]
    if len(dataset_files_map_ext[directory]) > 1:
        dataset_id += "." + ".".join(filename_parts)
    return dataset_id


def _result_paths(result_dir, dataset_dir, dataset_name, timestamp):
    result_root = result_dir + dataset_dir
    os.makedirs(result_root + "/result", exist_ok=True)
    os.makedirs(result_root + "/consistency-output", exist_ok=True)
    basename = f"{dataset_name}__{timestamp}.json"
    return (
        result_root + "/result/" + basename,
        result_root + "/consistency-output/" + basename,
    )


def _checker_options_for_file(
    file_path,
    first_file,
    consistency_file,
    result_dir,
    cli_options,
    time_checks_only,
    resume,
):
    tables_dir = result_dir + "/tables"
    force_table_download = file_path == first_file and (
        not resume or not os.listdir(tables_dir)
    )
    options = {
        checker: dict(checker_options)
        for checker, checker_options in cli_options.items()
    }

    # This list is the authoritative declaration that a plugin writes the
    # per-file consistency data consumed by ESGF-QA's dataset-level checks.
    for checker in checker_supporting_consistency_checks:
        options.setdefault(checker, {})["consistency_output"] = consistency_file

    options.setdefault("mip", {})["time_checks_only"] = time_checks_only
    options.setdefault("cc6", {}).update(
        tables_dir=tables_dir,
        force_table_download=force_table_download,
        time_checks_only=time_checks_only,
    )
    options.setdefault("cf", {})["enable_appendix_a_checks"] = True
    options.setdefault("wcrp_cordex_cmip6", {}).update(
        tables_dir=tables_dir,
        force_table_download=force_table_download,
    )
    return options


def discover_files(config):
    """Scan the configured data root and construct all workflow mappings."""
    files = []
    file_details = {}
    directory_datasets = {}
    discovered_file_count = 0
    blacklisted_files = {}
    not_whitelisted_files = []
    for root, _, filenames in os.walk(config.parent_dir):
        for filename in filenames:
            if not filename.endswith(".nc"):
                continue
            discovered_file_count += 1
            file_path = os.path.normpath(os.path.join(root, filename))
            blacklist_matches = [
                fragment for fragment in config.blacklist if fragment in file_path
            ]
            if blacklist_matches:
                blacklisted_files[file_path] = blacklist_matches
                continue
            if config.whitelist and not any(
                fragment in file_path for fragment in config.whitelist
            ):
                not_whitelisted_files.append(file_path)
                continue

            dataset_dir = os.path.dirname(file_path)
            stem_parts = os.path.splitext(os.path.basename(file_path))[0].split("_")
            dataset_name = "_".join(
                filter(re.compile(r"^(?!\d{1,}-{0,1}\d{0,}$)").match, stem_parts)
            )
            timestamp = "_".join(filter(re.compile(r"^\d{1,}-?\d*$").match, stem_parts))
            if "_" in timestamp:
                raise Exception(
                    f"Filename contains multiple time stamps: '{file_path}'"
                )
            result_file, consistency_file = _result_paths(
                config.result_dir, dataset_dir, dataset_name, timestamp
            )
            files.append(file_path)
            file_details[file_path] = {
                "id_dir": dataset_dir,
                "id_fn": dataset_name,
                "ts": timestamp,
                "result_file": result_file,
                "consistency_file": consistency_file,
            }
            directory_datasets.setdefault(dataset_dir, {}).setdefault(
                dataset_name, []
            ).append(file_path)

    files.sort()
    dataset_files = {}
    checker_options = defaultdict(dict)
    for file_path in files:
        details = file_details[file_path]
        details["id"] = get_dsid(
            file_details, directory_datasets, file_path, supported_project_ids
        )
        details["result_file_ds"] = (
            config.result_dir
            + "/"
            + details["id_dir"]
            + "/"
            + hashlib.md5(details["id"].encode()).hexdigest()
            + ".json"
        )
        dataset_files.setdefault(details["id"], []).append(file_path)
        checker_options[file_path] = _checker_options_for_file(
            file_path,
            files[0],
            details["consistency_file"],
            config.result_dir,
            config.checker_options,
            config.time_checks_only,
            config.resume,
        )

    invalidate_nonreusable_dataset_results(
        dataset_files,
        config.checkers,
        file_details,
        config.processed_files,
        config.processed_datasets,
    )
    return FileInventory(
        files=files,
        file_details=file_details,
        dataset_files=dataset_files,
        directory_datasets=directory_datasets,
        checker_options=checker_options,
        discovered_file_count=discovered_file_count,
        blacklisted_files=blacklisted_files,
        not_whitelisted_files=not_whitelisted_files,
    )


def format_exclusion_counts(inventory, config):
    """Format exclusion counts for configured filter types."""
    counts = []
    if config.blacklist:
        counts.append(f"{len(inventory.blacklisted_files)} were blacklisted")
    if config.whitelist:
        counts.append(
            f"{len(inventory.not_whitelisted_files)} matched no whitelist fragment"
        )
    return " and ".join(counts)


def write_excluded_files(inventory, config):
    """Write configured filters and every excluded file to JSON."""
    if not config.whitelist and not config.blacklist:
        return None
    output_path = os.path.join(config.result_dir, "excluded_files.json")
    data = {
        "filters": {
            "whitelist": config.whitelist,
            "blacklist": config.blacklist,
        },
        "summary": {
            "discovered": inventory.discovered_file_count,
            "selected": len(inventory.files),
            "blacklisted": len(inventory.blacklisted_files),
            "not_whitelisted": len(inventory.not_whitelisted_files),
        },
        "blacklisted": {
            path: inventory.blacklisted_files[path]
            for path in sorted(inventory.blacklisted_files)
        },
        "not_whitelisted": sorted(inventory.not_whitelisted_files),
    }
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)
    print(
        f"Path filters selected {len(inventory.files)} of "
        f"{inventory.discovered_file_count} discovered NetCDF files: "
        f"{format_exclusion_counts(inventory, config)}."
    )
    print(f"Excluded file information was saved to '{output_path}'.")
    return output_path


def write_inventory(inventory, result_dir):
    """Write discovered file/dataset mappings for inspection and reproducibility."""
    outputs = {
        "files_to_check.json": inventory.files,
        "files_to_check_dict.json": inventory.file_details,
        "dataset_files_map.json": inventory.dataset_files,
        "dataset_files_map_ext.json": inventory.directory_datasets,
    }
    for filename, data in outputs.items():
        with open(os.path.join(result_dir, filename), "w") as file:
            json.dump(data, file, indent=4)
    print("Information on discovered files and dataset grouping was saved to disk:")
    for filename in outputs:
        print(f" - {os.path.join(result_dir, filename)}")
