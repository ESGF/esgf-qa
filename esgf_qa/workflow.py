"""High-level QA execution and result serialization."""

import hashlib
import json
import multiprocessing
import os
import re
import warnings

from compliance_checker import __version__ as cc_version

from esgf_qa._constants import checker_supporting_consistency_checks
from esgf_qa.checker_registry import format_checker_version, get_checker_metadata
from esgf_qa.cluster_results import QAResultAggregator
from esgf_qa.con_checks import dataset_coverage_checks, inter_dataset_consistency_checks
from esgf_qa.workers import (
    call_process_dataset,
    call_process_file,
    process_file,
    run_dataset_collection_check,
)


def _process_count(limit=0, consistency_checks=False):
    count = max(multiprocessing.cpu_count() - 4, 1)
    if consistency_checks:
        # Dataset checks open many files simultaneously, so cap their file and
        # memory pressure independently of the available CPU count.
        count = min(count, 10)
    if limit > 0:
        count = min(count, limit)
    return count


def run_workflow(config, inventory):
    """Run file, dataset, and collection-level QA phases."""
    print(
        f"\nFound {len(inventory.files)} files "
        f"(organized in {len(inventory.dataset_files)} datasets) to check."
    )

    print("\n" + "#" * 50)
    print("# QA Part 1 - Run all compliance-checker checks")
    print("#" * 50 + "\n")
    summary = QAResultAggregator()
    reference_datasets = {}
    process_count = _process_count(config.parallel_processes)
    print(f"Using {process_count} parallel processes for cc checks.\n")

    # Run the first file synchronously so checker setup and any initial table
    # downloads complete before worker processes start using the same resources.
    first_file = inventory.files[0]
    processed_file, first_result = process_file(
        first_file,
        config.checkers,
        inventory.checker_options[first_file],
        inventory.file_details,
        config.processed_files,
        config.progress_file,
    )
    summary.update(
        first_result,
        inventory.file_details[processed_file]["id"],
        processed_file,
    )

    if len(inventory.files) > 1:
        args = [
            (
                file_path,
                config.checkers,
                inventory.checker_options[file_path],
                inventory.file_details,
                config.processed_files,
                config.progress_file,
            )
            for file_path in inventory.files[1:]
        ]
        with multiprocessing.Pool(processes=process_count, maxtasksperchild=10) as pool:
            for processed_file, result in pool.imap_unordered(call_process_file, args):
                summary.update(
                    result,
                    inventory.file_details[processed_file]["id"],
                    processed_file,
                )

    supports_consistency = any(
        checker.split(":", 1)[0] in checker_supporting_consistency_checks
        for checker in config.checkers
    )
    if not supports_consistency:
        warnings.warn(
            "Continuity & consistency checks skipped since no appropriate "
            "checkers were run. Supported checkers: "
            f"{', '.join(checker_supporting_consistency_checks)}"
        )
        return summary, reference_datasets

    print("\n" + "#" * 50)
    print("# QA Part 2 - Run consistency & continuity checks")
    print("#" * 50 + "\n")
    print("# QA Part 2.1 - Continuity & Consistency within each dataset")
    print("#   (Reference is the first file of each dataset timeseries)\n")
    process_count = _process_count(config.parallel_processes, consistency_checks=True)
    print(f"Using {process_count} parallel processes for dataset checks.\n")
    dataset_args = [
        (
            dataset_id,
            inventory.dataset_files,
            ["cons", "cont", "comp"],
            {"cons": {}, "cont": {}, "comp": {}},
            inventory.file_details,
            config.processed_datasets,
            config.dataset_file,
        )
        for dataset_id in sorted(inventory.dataset_files)
        if len(inventory.dataset_files[dataset_id]) > 1
    ]
    if dataset_args:
        with multiprocessing.Pool(processes=process_count, maxtasksperchild=10) as pool:
            for dataset_id, result in pool.imap_unordered(
                call_process_dataset, dataset_args
            ):
                summary.update_ds(result, dataset_id)

    print("\n# QA Part 2.2 - Continuity & Consistency across all datasets\n")
    inter_dataset_results = run_dataset_collection_check(
        summary,
        "cons",
        inter_dataset_consistency_checks,
        inventory.dataset_files,
        inventory.file_details,
        {},
    )
    if inter_dataset_results is not None:
        extra_results, reference_datasets = inter_dataset_results
        for dataset_id, result in extra_results.items():
            summary.update_ds({"cons": result}, dataset_id)

    coverage_results = run_dataset_collection_check(
        summary,
        "cons",
        dataset_coverage_checks,
        inventory.dataset_files,
        inventory.file_details,
        {},
    )
    if coverage_results is not None:
        for dataset_id, result in coverage_results.items():
            summary.update_ds({"cons": result}, dataset_id)
    return summary, reference_datasets


def write_results(
    config,
    inventory,
    summary,
    reference_datasets,
    timestamp_with_ms,
    timestamp_filename,
    timestamp_display,
):
    """Aggregate, cluster, and serialize the final QA results."""
    supports_consistency = any(
        checker.split(":", 1)[0] in checker_supporting_consistency_checks
        for checker in config.checkers
    )
    print("\n" + "#" * 50)
    print(
        f"# QA Part {'3' if supports_consistency else '2'} - "
        "Summarizing and clustering the results"
    )
    print("#" * 50 + "\n")

    summary.sort()
    checker_metadata = get_checker_metadata(config.checkers)
    summary_info = {
        "id": "",
        "date": timestamp_display,
        "files": str(len(inventory.files)),
        "datasets": str(len(inventory.dataset_files)),
        "cc_version": cc_version,
        "checkers": [
            format_checker_version(checker, checker_metadata)
            for checker in config.checkers
        ],
        "parent_dir": config.parent_dir,
    }
    if reference_datasets:
        summary_info["inter_ds_con_checks_ref"] = reference_datasets

    dataset_ids = list(inventory.dataset_files)
    common_prefix = os.path.commonprefix(dataset_ids)
    if common_prefix != dataset_ids[0]:
        common_prefix += "*"
    summary_info["id"] = (
        f"{config.info} ({common_prefix})" if config.info else common_prefix
    )

    full_summary = summary.summary
    full_summary["info"] = summary_info
    file_id = hashlib.md5(timestamp_with_ms.encode()).hexdigest()
    info_slug = re.sub("[^a-z0-9]", "", config.info.lower())[:10]
    prefix = f"qa_result_{info_slug + '_' if info_slug else ''}"
    filename = f"{prefix}{timestamp_filename}_{file_id}.json"
    with open(os.path.join(config.result_dir, filename), "w") as file:
        json.dump(
            full_summary,
            file,
            indent=4,
            ensure_ascii=False,
            sort_keys=False,
        )
    print(f"Saved QC result: {config.result_dir}/{filename}")

    summary.cluster_summary()
    clustered_summary = summary.clustered_summary
    clustered_summary["info"] = summary_info
    filename = f"{prefix}{timestamp_filename}_{file_id}.cluster.json"
    with open(os.path.join(config.result_dir, filename), "w") as file:
        json.dump(
            clustered_summary,
            file,
            indent=4,
            ensure_ascii=False,
            sort_keys=False,
        )
    print(f"Saved QC cluster summary: {config.result_dir}/{filename}")
