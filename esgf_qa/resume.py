"""Resume metadata and cached-result handling."""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from esgf_qa._constants import checker_supporting_consistency_checks


def verify_options_dict(options):
    """Return whether serialized checker options have the expected shape."""
    if not isinstance(options, dict):
        return False
    try:
        return all(
            isinstance(checker_options, dict)
            and all(
                isinstance(value, (int, float, str, bool, type(None)))
                for value in checker_options.values()
            )
            for checker_options in options.values()
        )
    except AttributeError:
        return False


@dataclass
class ResumeInfo:
    """Configuration persisted alongside a resumable QA run."""

    parent_dir: str
    info: str
    tests: list[str]
    checker_options: dict = field(default_factory=dict)
    include_consistency_checks: bool = False

    @classmethod
    def from_dict(cls, data, result_dir):
        required_keys = {"parent_dir", "info", "tests"}
        if not required_keys.issubset(data):
            raise Exception(
                f"Invalid .resume_info file in '{result_dir}'. It should contain "
                "the keys 'parent_dir', 'info', and 'tests'."
            )
        if not (
            isinstance(data["parent_dir"], str)
            and isinstance(data["info"], str)
            and isinstance(data["tests"], list)
            and all(isinstance(test, str) for test in data["tests"])
            and verify_options_dict(data.get("checker_options", {}))
            and isinstance(data.get("include_consistency_checks", False), bool)
        ):
            raise Exception(
                f"Invalid .resume_info file in '{result_dir}'. 'parent_dir' and "
                "'info' should be strings, 'tests' should be a list of strings, "
                "'checker_options' should be a nested dictionary, and "
                "'include_consistency_checks' should be a boolean."
            )
        return cls(
            parent_dir=data["parent_dir"],
            info=data["info"],
            tests=data["tests"],
            checker_options=data.get("checker_options", {}),
            include_consistency_checks=data.get("include_consistency_checks", False),
        )

    def to_dict(self):
        data = {
            "parent_dir": self.parent_dir,
            "info": self.info,
            "tests": self.tests,
        }
        if self.include_consistency_checks:
            data["include_consistency_checks"] = True
        if self.checker_options:
            data["checker_options"] = self.checker_options
        return data


def load_resume_info(path, result_dir):
    """Load and validate a ``.resume_info`` file."""
    try:
        with open(path) as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise Exception(
            f"Invalid .resume_info file in '{result_dir}'. It needs to be a valid "
            "JSON file."
        ) from error
    return ResumeInfo.from_dict(data, result_dir)


def write_resume_info(path, resume_info):
    """Persist resume configuration."""
    with open(path, "w") as file:
        json.dump(resume_info.to_dict(), file, sort_keys=True, indent=4)


def prepare_result_directory(result_dir, resume, progress_file, resume_info_file):
    """Validate or create the result directory for a QA run."""
    tables_dir = Path(result_dir, "tables")
    if not os.path.exists(result_dir):
        if resume:
            raise FileNotFoundError(
                "Resume is set but specified output_directory does not exist: "
                f"'{result_dir}'."
            )
        os.mkdir(result_dir)
        return
    if not os.listdir(result_dir):
        if resume:
            raise FileNotFoundError(
                f"Resume is set but specified output directory is empty: '{result_dir}'."
            )
        return

    previous_run = (
        os.path.isfile(progress_file)
        and os.path.isfile(resume_info_file)
        and os.path.isdir(tables_dir)
    )
    if resume and not previous_run:
        raise Exception(
            "Resume is set but specified output_directory cannot be identified as "
            "output directory of a previous QA run."
        )
    if not resume and previous_run:
        raise Exception(
            "Specified output directory is not empty but can be identified as output "
            "directory of a previous QA run. Use '-r' or '--resume' together with "
            "'-o' or '--output_dir', or choose a different output directory."
        )
    if not resume:
        raise Exception("Specified output directory is not empty.")


def read_progress(path):
    """Read newline-delimited progress identifiers."""
    with open(path) as file:
        return {line.strip() for line in file}


def get_reusable_file_result(file_path, checkers, files_to_check_dict, processed_files):
    """Return a valid cached result, or ``None`` when the file must be checked."""
    result_file = files_to_check_dict[file_path]["result_file"]
    consistency_file = files_to_check_dict[file_path]["consistency_file"]
    consistency_output_required = any(
        checker.split(":", 1)[0] in checker_supporting_consistency_checks
        for checker in checkers
    )
    if (
        file_path not in processed_files
        or not os.path.isfile(result_file)
        or (consistency_output_required and not os.path.isfile(consistency_file))
    ):
        return None
    try:
        with open(result_file) as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    for checker in checkers:
        checker_result = result.get(checker.split(":", 1)[0])
        if not isinstance(checker_result, dict) or checker_result.get("errors") != {}:
            return None
    return result


def invalidate_nonreusable_dataset_results(
    dataset_files_map,
    checkers,
    files_to_check_dict,
    processed_files,
    processed_datasets,
):
    """Invalidate dataset caches affected by new or incomplete file results."""
    for dataset_id, dataset_files in dataset_files_map.items():
        if any(
            get_reusable_file_result(
                file_path, checkers, files_to_check_dict, processed_files
            )
            is None
            for file_path in dataset_files
        ):
            processed_datasets.discard(dataset_id)


def track_checked_datasets(checked_datasets_file, checked_datasets):
    """Append dataset identifiers to a progress file."""
    with open(checked_datasets_file, "a") as file:
        writer = csv.writer(file)
        for dataset_id in checked_datasets:
            writer.writerow([dataset_id])


# Compatibility aliases for the former private names in run_qa.
_get_reusable_file_result = get_reusable_file_result
_invalidate_nonreusable_dataset_results = invalidate_nonreusable_dataset_results
_verify_options_dict = verify_options_dict
