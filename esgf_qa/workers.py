"""File- and dataset-level QA worker functions."""

import json
import os
import tempfile
import traceback
from pathlib import Path

from compliance_checker.runner import CheckSuite

from esgf_qa._constants import checker_supporting_consistency_checks
from esgf_qa.con_checks import (
    compatibility_checks,
    consistency_checks,
    continuity_checks,
)
from esgf_qa.resume import get_reusable_dataset_result, get_reusable_file_result

DATASET_CHECKERS = {
    "cons": consistency_checks,
    "cont": continuity_checks,
    "comp": compatibility_checks,
}


def _failed_checker_result(error, stage):
    """Represent a failure outside an individual checker method."""
    return [], {stage: (error, error.__traceback__)}


def _remove_stale_output(path):
    """Remove one known generated output before rebuilding it."""
    Path(path).unlink(missing_ok=True)


def _replace_json(path, data):
    """Atomically replace a JSON result after it has been serialized fully."""
    result_path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=result_path.parent,
            prefix=f".{result_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, ensure_ascii=False, indent=4)
        os.replace(temporary_path, result_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_compliance_checker(file_path, checkers, checker_options=None):
    """Run Compliance Checker for one file, isolating checker-level failures."""
    checker_options = checker_options or {}
    try:
        check_suite = CheckSuite(options=checker_options)
        check_suite.load_all_available_checkers()
    except Exception as error:
        return {
            checker: _failed_checker_result(error, "run_compliance_checker")
            for checker in checkers
        }

    try:
        dataset = check_suite.load_dataset(file_path)
    except Exception as error:
        return {
            checker: _failed_checker_result(error, "load_dataset")
            for checker in checkers
        }

    time_checks_only = checker_options.get("cc6", {}).get(
        "time_checks_only", False
    ) or checker_options.get("mip", {}).get("time_checks_only", False)
    include_checks = (
        ["check_time_continuity", "check_time_bounds", "check_time_range"]
        if time_checks_only
        else None
    )

    results = {}
    close_error = None
    try:
        for checker in checkers:
            checker_include = (
                include_checks if checker.split(":", 1)[0] in {"cc6", "mip"} else None
            )
            try:
                checker_results = check_suite.run_all(
                    dataset,
                    [checker],
                    include_checks=checker_include,
                    skip_checks=[],
                )
                if checker not in checker_results:
                    raise RuntimeError(
                        f"Compliance Checker returned no result for '{checker}'."
                    )
                results[checker] = checker_results[checker]
            except Exception as error:
                results[checker] = _failed_checker_result(error, "run_checker")
    finally:
        if hasattr(dataset, "close"):
            try:
                dataset.close()
            except Exception as error:
                close_error = error

    if close_error is not None:
        for checker in checkers:
            results.setdefault(
                checker, _failed_checker_result(close_error, "close_dataset")
            )[1]["close_dataset"] = (close_error, close_error.__traceback__)
    return results


def _format_compliance_checker_runtime_error(check_method, error_details):
    """Format checker errors without assuming a matching traceback function."""
    try:
        error, traceback_entry = error_details
    except (TypeError, ValueError):
        return f"Exception: {error_details}"

    matching_entry = None
    fallback_entry = None
    current_entry = traceback_entry
    while current_entry is not None:
        fallback_entry = current_entry
        if current_entry.tb_frame.f_code.co_name == check_method:
            matching_entry = current_entry
        current_entry = current_entry.tb_next
    traceback_entry = matching_entry or fallback_entry
    if traceback_entry is None:
        return f"Exception: {error}"

    message = (
        f"Exception: {error} at "
        f"{traceback_entry.tb_frame.f_code.co_filename}:"
        f"{traceback_entry.tb_lineno} in function/method "
        f"'{traceback_entry.tb_frame.f_code.co_name}'."
    )
    affected_variables = [
        value
        for name, value in traceback_entry.tb_frame.f_locals.items()
        if "var" in name and isinstance(value, str)
    ]
    if affected_variables:
        message += f" Potentially affected variables: {', '.join(affected_variables)}."
    return message


def process_file(
    file_path,
    checkers,
    checker_options,
    files_to_check_dict,
    processed_files,
    progress_file,
):
    """Run or reuse file-level checks for one file."""
    consistency_file = files_to_check_dict[file_path]["consistency_file"]
    result_file = files_to_check_dict[file_path]["result_file"]
    result = get_reusable_file_result(
        file_path, checkers, files_to_check_dict, processed_files
    )
    if result is not None:
        print(f"Read result from disk for '{file_path}'.")
        return file_path, result
    if file_path in processed_files:
        print(f"Rerunning incomplete or previously erroneous checks for '{file_path}'.")
    else:
        print(f"Running checks for '{file_path}'.")

    # A rerun must not mistake output from an earlier attempt for freshly
    # generated output when the checker fails before writing its new files.
    _remove_stale_output(result_file)
    _remove_stale_output(consistency_file)
    result = run_compliance_checker(file_path, checkers, checker_options)
    check_results = {}
    for checker_spec in checkers:
        checker = checker_spec.split(":", 1)[0]
        checker_result = {"errors": {}}
        check_results[checker] = checker_result
        for check in result[checker_spec][0]:
            serialized_check = {
                "weight": check.weight,
                "value": check.value,
                "msgs": check.msgs,
                "method": check.check_method,
                "children": check.children,
            }
            previous_result = checker_result.get(check.name)
            if previous_result is None:
                checker_result[check.name] = serialized_check
            elif isinstance(previous_result, list):
                previous_result.append(serialized_check)
            else:
                # A single Compliance Checker method can report the same named
                # check at multiple severities; retain every result record.
                checker_result[check.name] = [previous_result, serialized_check]

        # Error keys are checker method names, whereas normal result keys above
        # are the human-readable check names exposed by Compliance Checker.
        for check_method, error_details in result[checker_spec][1].items():
            checker_result["errors"][check_method] = (
                _format_compliance_checker_runtime_error(check_method, error_details)
            )

    if not os.path.isfile(consistency_file):
        for checker in (
            checker.split(":", 1)[0]
            for checker in checkers
            if checker.split(":", 1)[0] in checker_supporting_consistency_checks
        ):
            check_results[checker]["errors"][
                "consistency_output"
            ] = f"Expected consistency output file was not created: '{consistency_file}'."

    _replace_json(result_file, check_results)
    with open(progress_file, "a") as file:
        file.write(file_path + "\n")
    return file_path, check_results


def _format_dataset_check_runtime_error(error):
    """Format an exception with its most relevant consistency-check frame."""
    frames = traceback.extract_tb(error.__traceback__)
    frame = next(
        (
            frame
            for frame in reversed(frames)
            if os.path.basename(frame.filename) == "con_checks.py"
        ),
        frames[-1] if frames else None,
    )
    if frame is None:
        return f"Exception: {error}"
    return f"Exception: {error} at {frame.filename}:{frame.lineno} in function/method '{frame.name}'."


def _dataset_check_runtime_error(function_name, error, files):
    """Build the dataset-level error structure consumed by the aggregator."""
    return {
        "errors": {
            function_name: {
                "msg": _format_dataset_check_runtime_error(error),
                "files": sorted(files),
            }
        }
    }


def run_dataset_collection_check(
    summary,
    checker,
    checker_fct,
    ds_map,
    files_to_check_dict,
    checker_options,
):
    """Run an all-dataset check and aggregate a runtime error if it fails."""
    try:
        return checker_fct(ds_map, files_to_check_dict, checker_options)
    except Exception as error:
        for dataset_id, files in ds_map.items():
            summary.update_ds(
                {
                    checker: _dataset_check_runtime_error(
                        checker_fct.__name__, error, files
                    )
                },
                dataset_id,
            )
        return None


def process_dataset(
    dataset_id,
    dataset_files_map,
    checkers,
    checker_options,
    files_to_check_dict,
    processed_datasets,
    progress_file,
):
    """Run or reuse dataset-level consistency checks."""
    dataset_files = dataset_files_map[dataset_id]
    result_file = files_to_check_dict[dataset_files[0]]["result_file_ds"]
    result = get_reusable_dataset_result(
        dataset_id, checkers, result_file, processed_datasets
    )
    if result is not None:
        print(f"Read result from disk for '{dataset_id}'.")
        return dataset_id, result
    if dataset_id in processed_datasets:
        print(f"Rerunning previously erroneous checks for '{dataset_id}'.")
    else:
        print(f"Running checks for '{dataset_id}'.")

    _remove_stale_output(result_file)
    result = {}
    for checker_spec in checkers:
        checker = checker_spec.split(":", 1)[0]
        checker_fct = DATASET_CHECKERS.get(checker)
        if checker_fct is None:
            result[checker] = {
                "errors": {
                    checker: {
                        "msg": f"Checker '{checker}' not found.",
                        "files": dataset_files,
                    }
                }
            }
            continue
        try:
            result[checker] = checker_fct(
                dataset_id,
                dataset_files_map,
                files_to_check_dict,
                checker_options[checker],
            )
        except Exception as error:
            result[checker] = _dataset_check_runtime_error(
                checker_fct.__name__, error, dataset_files
            )

    _replace_json(result_file, result)
    with open(progress_file, "a") as file:
        file.write(dataset_id + "\n")
    return dataset_id, result


def call_process_file(args):
    """Unpack multiprocessing arguments for :func:`process_file`."""
    return process_file(*args)


def call_process_dataset(args):
    """Unpack multiprocessing arguments for :func:`process_dataset`."""
    return process_dataset(*args)
