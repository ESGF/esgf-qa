import argparse
import csv
import datetime
import difflib
import hashlib
import json
import multiprocessing
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import xarray as xr
from compliance_checker import __version__ as cc_version
from compliance_checker.runner import CheckSuite

checker_dict = {
    "cc6": "CORDEX-CMIP6",
    "cf": "CF-Conventions",
}
checker_release_versions = {}

_timestamp_with_ms = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
_timestamp_filename = datetime.datetime.strptime(
    _timestamp_with_ms, "%Y%m%d-%H%M%S%f"
).strftime("%Y%m%d-%H%M")
_timestamp_pprint = datetime.datetime.strptime(
    _timestamp_with_ms, "%Y%m%d-%H%M%S%f"
).strftime("%Y-%m-%d %H:%M")


class QAResultAggregator:
    def __init__(self, checker_dict):
        """
        Initialize the aggregator with an empty summary.
        """
        self.summary = {
            "error": defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            ),  # No weight, just function -> error msg
            "fail": defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            ),  # weight -> test -> msg -> dsid -> filenames
        }
        self.checker_dict = checker_dict

    def update(self, result_dict, dsid, file_name):
        """
        Update the summary with a single result.
        """
        for checker in result_dict:
            for test in result_dict[checker]:
                if test == "errors":
                    for function_name, error_msg in result_dict[checker][
                        "errors"
                    ].items():
                        self.summary["error"][
                            f"[{checker_dict[checker]}] " + function_name
                        ][error_msg][dsid].append(file_name)
                else:
                    score, max_score = result_dict[checker][test]["value"]
                    weight = result_dict[checker][test].get("weight", 3)
                    msgs = result_dict[checker][test].get("msgs", [])
                    if score < max_score:  # test outcome: fail
                        for msg in msgs:
                            self.summary["fail"][weight][
                                f"[{checker_dict[checker]}] " + test
                            ][msg][dsid].append(file_name)

    def sort(self):
        """
        Sort the summary.
        """
        self.summary["fail"] = dict(sorted(self.summary["fail"].items(), reverse=True))
        for key in self.summary["fail"]:
            self.summary["fail"][key] = dict(sorted(self.summary["fail"][key].items()))

        # Sort errors by function name
        for checker in self.summary["error"]:
            self.summary["error"][checker] = dict(
                sorted(self.summary["error"][checker].items())
            )

    def cluster_summary(self, sim_threshold=0.85, min_group_size=2):
        """
        Apply clustering to too extensive parts of the summary.
        """
        new_summary = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )

        for level, weights in self.summary.items():
            if level == "error":
                # There is no weights layer for errors,
                #  so "weights" are actually "tests" in this case
                for test_id, messages in weights.items():
                    # Group messages by ds_id sets
                    dsid_to_messages = defaultdict(list)
                    for msg, dsids in messages.items():
                        for dsid, files in dsids.items():
                            dsid_to_messages[dsid].append((msg, files))

                        for dsid, msg_file_list in dsid_to_messages.items():
                            clustered = QAResultAggregator.cluster_messages(
                                msg_file_list, sim_threshold, min_group_size
                            )
                            for clustered_msg, file_list in clustered:
                                file_summary = QAResultAggregator.summarize_file_list(
                                    file_list
                                )
                                new_summary[level][test_id][clustered_msg][dsid] = [
                                    file_summary
                                ]
            elif level == "fail":
                for weight, tests in weights.items():
                    for test_id, messages in tests.items():
                        # Group messages by ds_id sets
                        dsid_to_messages = defaultdict(list)
                        for msg, dsids in messages.items():
                            for dsid, files in dsids.items():
                                dsid_to_messages[dsid].append((msg, files))

                        for dsid, msg_file_list in dsid_to_messages.items():
                            clustered = QAResultAggregator.cluster_messages(
                                msg_file_list, sim_threshold, min_group_size
                            )
                            for clustered_msg, file_list in clustered:
                                file_summary = QAResultAggregator.summarize_file_list(
                                    file_list
                                )
                                new_summary[level][weight][test_id][clustered_msg][
                                    dsid
                                ] = [file_summary]

        return new_summary

    @staticmethod
    def cluster_messages(msg_file_list, sim_threshold, min_group_size):
        used = [False] * len(msg_file_list)
        result = []

        for i, (msg_i, files_i) in enumerate(msg_file_list):
            if used[i]:
                continue
            similar = [(msg_i, files_i)]
            used[i] = True
            for j in range(i + 1, len(msg_file_list)):
                if used[j]:
                    continue
                msg_j, _ = msg_file_list[j]
                ratio = difflib.SequenceMatcher(None, msg_i, msg_j).ratio()
                if ratio >= sim_threshold:
                    similar.append(msg_file_list[j])
                    used[j] = True

            if len(similar) >= min_group_size:
                template = QAResultAggregator.generalize_message(similar[0][0])
                example = QAResultAggregator.extract_example(similar[0][0])
                clustered_msg = (
                    f"{template} ({len(similar)} occurrences, e.g. {example})"
                )
                all_files = [f for _, files in similar for f in files]
                result.append((clustered_msg, all_files))
            else:
                for msg, files in similar:
                    result.append((msg, files))
        return result

    @staticmethod
    def generalize_message(msg):
        msg = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "{}", msg)
        msg = re.sub(r"\d+", "{}", msg)
        return msg

    @staticmethod
    def extract_example(msg):
        match = re.search(r"'[^']*' \('.*?'\)", msg)
        return match.group(0) if match else "example"

    @staticmethod
    def summarize_file_list(files):
        if not files:
            return ""
        elif len(files) > 1:
            return f"{len(files)} files affected, e.g. '{files[0]}'"
        else:
            return files[0]


def get_default_result_dir():
    global _timestamp
    global _timestamp_with_ms
    hash_object = hashlib.md5(_timestamp_with_ms.encode())
    return (
        os.path.abspath(".")
        + f"/cc-qa-results_{_timestamp_filename}_{hash_object.hexdigest()}"
    )


def get_dsid(files_to_check_dict, dataset_files_map_ext, file_path, project_id):
    dir_id = files_to_check_dict[file_path]["id_dir"].split("/")
    fn_id = files_to_check_dict[file_path]["id_fn"].split("_")
    if project_id in dir_id:
        dsid = ".".join(dir_id[dir_id.index(project_id) :])
    else:
        dsid = ".".join(dir_id)
    if len(dataset_files_map_ext[files_to_check_dict[file_path]["id_dir"]].keys()) > 1:
        dsid += "." + ".".join(fn_id)
    return dsid


def get_checker_release_versions(checkers, checker_options={}):
    global checker_release_versions
    check_suite = CheckSuite(options=checker_options)
    check_suite.load_all_available_checkers()
    for checker in checkers:
        if checker not in checker_release_versions:
            checker_release_versions[checker] = check_suite.checkers.get(
                checker, "unknown version"
            )._cc_spec_version


def run_compliance_checker(file_path, checkers, checker_options={}):
    """
    Run the compliance checker on a file with the specified checkers and options.

    Parameters:
        file_path (str): Path to the file to be checked.
        checkers (list): List of checkers to run.
        checker_options (dict): Dictionary of options for each checker.
                                Example format: {"cf": {"check_dimension_order": True}}
    """
    check_suite = CheckSuite(options=checker_options)
    check_suite.load_all_available_checkers()
    ds = check_suite.load_dataset(file_path)
    return check_suite.run_all(ds, checkers, skip_checks=[])


def run_external_check(file_paths, results_queue):
    dataset = xr.open_mfdataset(file_paths, combine="by_coords")
    # Perform external checks on the dataset
    # ...
    results_queue.put(None)


def track_checked_datasets(checked_datasets_file, checked_datasets):
    with open(checked_datasets_file, "a") as file:
        writer = csv.writer(file)
        for dataset_id in checked_datasets:
            writer.writerow([dataset_id])


def process_file(
    file_path,
    checkers,
    checker_options,
    files_to_check_dict,
    processed_files,
    progress_file,
):
    # Read result from disk if check was run previously
    result_file = files_to_check_dict[file_path]["result_file"]
    consistency_file = files_to_check_dict[file_path]["consistency_file"]
    if (
        file_path in processed_files
        and os.path.isfile(result_file)
        and os.path.isfile(consistency_file)
    ):
        with open(result_file) as file:
            print(f"Read result from disk for '{file_path}'.")
            result = json.load(file)
        # If no runtime errors were registered last time, return results, otherwise rerun checks
        # Potentially add more conditions to rerun checks:
        #  eg. rerun checks if runtime errors occured
        #      rerun checks if lvl 1 checks failed
        #      rerun checks if lvl 1 and 2 checks failed
        #      rerun checks if any checks failed
        #      rerun checks if forced by user
        # if all(result[checker][1] == {} for checker in checkers):
        if all(result[checker]["errors"] == {} for checker in checkers):
            return file_path, result
        else:
            print(f"Rerunning previously erroneous checks for '{file_path}'.")
    else:
        print(f"Running checks for '{file_path}'.")

    # Else run check
    result = run_compliance_checker(file_path, checkers, checker_options)

    # Check result
    check_results = dict()
    # Note: the key in the errors dict is not the same as the check name!
    #       The key is the checker function name, while the check.name
    #       is the description.
    for checker in checkers:
        check_results[checker] = dict()
        check_results[checker]["errors"] = {}
        # print()
        # print("name",result[checker][0][0].name)
        # print("weight", result[checker][0][0].weight)
        # print("value", result[checker][0][0].value)
        # print("msgs", result[checker][0][0].msgs)
        # print("method", result[checker][0][0].check_method)
        # print("children", result[checker][0][0].children)
        # quit()
        for check in result[checker][0]:
            check_results[checker][check.name] = {}
            check_results[checker][check.name]["weight"] = check.weight
            check_results[checker][check.name]["value"] = check.value
            check_results[checker][check.name]["msgs"] = check.msgs
            check_results[checker][check.name]["method"] = check.check_method
            check_results[checker][check.name]["children"] = check.children
        for check_method in result[checker][1]:
            a = result[checker][1][check_method][1]
            while True:
                if a.tb_frame.f_code.co_name == check_method:
                    break
                else:
                    a = a.tb_next
            check_results[checker]["errors"][
                check_method
            ] = f"Exception: {result[checker][1][check_method][0]} at {a.tb_frame.f_code.co_filename}:{a.tb_frame.f_lineno} in function/method '{a.tb_frame.f_code.co_name}'."
            vars = [
                j
                for i, j in a.tb_frame.f_locals.items()
                if "var" in i and isinstance(j, str)
            ]
            if vars:
                check_results[checker]["errors"][
                    check_method
                ] += f" Potentially affected variables: {', '.join(vars)}."

    # Write result to disk
    with open(result_file, "w") as f:
        json.dump(check_results, f, ensure_ascii=False, indent=4)

    # Register file in progress file
    with open(progress_file, "a") as file:
        file.write(file_path + "\n")

    return file_path, check_results


def call_process_file(args):
    return process_file(*args)


def process_dataset(dataset_files, results_queue):
    run_external_check(dataset_files, results_queue)


def main():
    # CLI
    parser = argparse.ArgumentParser(description="Run QA checks")
    parser.add_argument(
        "parent_dir",
        type=str,
        help="Parent directory to scan for files",
        nargs="?",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=get_default_result_dir(),
        help="Directory to store QA results. Needs to be non-existing or empty or from previous QA run.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="append",
        help="The test to run ('cc6' or 'cf', can be specified multiple times) - default: running 'cc6' and 'cf'",
    )
    parser.add_argument(
        "-i",
        "--info",
        type=str,
        help="Informtaion to be included in the QA results identifying the current run, eg. the experiment_id.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Specify to continue a previous QC run. Requires the <output_dir> argument to be set.",
    )
    args = parser.parse_args()

    result_dir = os.path.abspath(args.output_dir)
    parent_dir = os.path.abspath(args.parent_dir) if args.parent_dir else None
    tests = sorted(args.test) if args.test else []
    info = args.info if args.info else ""
    resume = args.resume

    # Progress file to track already checked files
    progress_file = Path(result_dir, "progress.txt")

    # Resume information stored in a json file
    resume_info_file = Path(result_dir, ".resume_info")

    # Deal with result_dir
    if not os.path.exists(result_dir):
        if resume:
            resume = False
            warnings.warn(
                "Resume is set but specified output_directory does not exist. Starting a new QA run..."
            )
        os.mkdir(result_dir)
    elif os.listdir(result_dir) != []:
        if resume:
            required_files = [progress_file, resume_info_file]
            required_paths = [os.path.join(result_dir, p) for p in ["tables"]]
            if not all(os.path.isfile(rfile) for rfile in required_files) or not all(
                os.path.isdir(rpath) and os.listdir(rpath) != []
                for rpath in required_paths
            ):
                raise Exception(
                    "Resume is set but specified output_directory cannot be identified as output_directory of a previous QA run."
                )
        else:
            if "progress.txt" in os.listdir(
                result_dir
            ) and ".resume_info" in os.listdir(result_dir):
                raise Exception(
                    "Specified output_directory is not empty but can be identified as output_directory of a prevous QA run. Use'-r' or '--resume' (together with '-o' or '--output_dir') to continue the previous QA run or choose a different output_directory instead."
                )
            else:
                raise Exception("Specified output_directory is not empty.")
    else:
        if resume:
            resume = False
            warnings.warn(
                "Resume is set but specified output_directory is empty. Starting a new QA run..."
            )
    if resume:
        print(f"Resuming previous QA run in '{result_dir}'")
        with open(os.path.join(result_dir, ".resume_info")) as f:
            try:
                resume_info = json.load(f)
                required_keys = ["parent_dir", "info", "tests"]
                if not all(key in resume_info for key in required_keys):
                    raise Exception(
                        "Invalid .resume_info file. It should contain the keys 'parent_dir', 'info', and 'tests'."
                    )
                if not (
                    isinstance(resume_info["parent_dir"], str)
                    and isinstance(resume_info["info"], str)
                    and isinstance(resume_info["tests"], list)
                    and all(isinstance(test, str) for test in resume_info["tests"])
                ):
                    raise Exception(
                        "Invalid .resume_info file. 'parent_dir' and 'info' should be strings, and 'tests' should be a list of strings."
                    )
            except json.JSONDecodeError:
                raise Exception(
                    "Invalid .resume_info file. It should be a valid JSON file."
                )
            if tests and tests != resume_info["tests"]:
                raise Exception("Cannot resume a previous QA run with different tests.")
            else:
                tests = resume_info["tests"]
            if info and info != resume_info["info"]:
                warnings.warn(
                    f"<info> argument differs from the originally specified <info> argument ('{resume_info['info']}'). Using the new specification."
                )
            if parent_dir and Path(parent_dir) != Path(resume_info["parent_dir"]):
                raise Exception(
                    "Cannot resume a previous QA run with different <parent_dir>."
                )
            else:
                parent_dir = Path(resume_info["parent_dir"])
    else:
        print(f"Storing check results in '{result_dir}'")

    # Deal with tests
    if not tests:
        checkers = ["cc6", "cf"]
        checkers_versions = ["latest", "latest"]
        checker_options = {}
    else:
        test_regex = re.compile(r"^[a-z0-9]+:(latest|[0-9]+(\.[0-9]+)*)$")
        if not all([test_regex.match(test) for test in tests]):
            raise Exception("Invalid test(s) specified.")
        checkers = [test.split(":")[0] for test in tests]
        checkers_versions = [
            (
                test.split(":")[1]
                if len(test.split(":")) == 2 and test.split(":")[1] != ""
                else "latest"
            )
            for test in tests
        ]
        checker_options = {}
        if any(test not in checker_dict.keys() for test in checkers):
            raise Exception(
                f"Invalid test(s) specified. Supported are: {', '.join(checker_dict.keys())}"
            )

    # Does parent_dir exist?
    if parent_dir is None or not os.path.exists(parent_dir):
        raise Exception(f"The specified <parent_dir> '{parent_dir}' does not exist.")

    # Write resume file
    resume_info = {
        "parent_dir": str(parent_dir),
        "info": info,
        "tests": sorted([f"{c}:{v}" for c, v in zip(checkers, checkers_versions)]),
    }
    with open(os.path.join(result_dir, ".resume_info"), "w") as f:
        json.dump(resume_info, f)

    # Ensure progress file exists
    progress_file.touch()

    # list(filter(re.compile(r"^(?!\d{1,}-{0,1}\d{0,}$)").match, os.path.splitext(filename)[0].split("_")))
    # Check if progress file exists and read already processed files
    processed_files = set()
    with open(progress_file) as file:
        for line in file:
            processed_files.add(line.strip())

    # todo: allow black-/whitelisting (parts of) paths for checks
    path_whitelist = []
    path_blacklist = []

    #########################################################
    # Find all files to check and group them in datasets
    #########################################################
    files_to_check = []  # List of files to check
    files_to_check_dict = {}
    dataset_files_map = {}  # Map to store files grouped by their dataset_ids
    dataset_files_map_ext = (
        {}
    )  # allowing files of multiple datasets in a single directory
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.endswith(".nc"):
                file_path = os.path.normpath(os.path.join(root, file))
                dataset_id_dir = os.path.dirname(file_path)
                dataset_id_fn = "_".join(
                    filter(
                        re.compile(r"^(?!\d{1,}-{0,1}\d{0,}$)").match,
                        os.path.splitext(os.path.basename(file_path))[0].split("_"),
                    )
                )
                dataset_timestamp = "_".join(
                    filter(
                        re.compile(r"^\d{1,}-?\d*$").match,
                        os.path.splitext(os.path.basename(file_path))[0].split("_"),
                    )
                )
                os.makedirs(result_dir + dataset_id_dir + "/result", exist_ok=True)
                os.makedirs(
                    result_dir + dataset_id_dir + "/consistency-output", exist_ok=True
                )
                result_file = (
                    result_dir
                    + dataset_id_dir
                    + "/"
                    + "result"
                    + "/"
                    + dataset_id_fn
                    + "__"
                    + dataset_timestamp
                    + ".json"
                )
                consistency_file = (
                    result_dir
                    + dataset_id_dir
                    + "/"
                    + "consistency-output"
                    + "/"
                    + dataset_id_fn
                    + "__"
                    + dataset_timestamp
                    + ".json"
                )
                if "_" in dataset_timestamp:
                    raise Exception(
                        f"Filename contains multiple time stamps: '{file_path}'"
                    )
                if any(file_path.startswith(skip_path) for skip_path in path_blacklist):
                    continue
                if path_whitelist != [] and not any(
                    file_path.startswith(use_path) for use_path in path_whitelist
                ):
                    continue
                files_to_check.append(file_path)
                files_to_check_dict[file_path] = {
                    "id_dir": dataset_id_dir,
                    "id_fn": dataset_id_fn,
                    "ts": dataset_timestamp,
                    "result_file": result_file,
                    "consistency_file": consistency_file,
                }
                if dataset_id_dir in dataset_files_map_ext:
                    if dataset_id_fn in dataset_files_map_ext[dataset_id_dir]:
                        dataset_files_map_ext[dataset_id_dir][dataset_id_fn].append(
                            file_path
                        )
                    else:
                        dataset_files_map_ext[dataset_id_dir][dataset_id_fn] = [
                            file_path
                        ]
                else:
                    dataset_files_map_ext[dataset_id_dir] = {dataset_id_fn: [file_path]}
    files_to_check = sorted(files_to_check)
    for file_path in files_to_check:
        files_to_check_dict[file_path]["id"] = get_dsid(
            files_to_check_dict, dataset_files_map_ext, file_path, "CORDEX-CMIP6"
        )
        if files_to_check_dict[file_path]["id"] in dataset_files_map:
            dataset_files_map[files_to_check_dict[file_path]["id"]].append(file_path)
        else:
            dataset_files_map[files_to_check_dict[file_path]["id"]] = [file_path]
        checker_options[file_path] = {
            "cc6": {
                "consistency_output": files_to_check_dict[file_path][
                    "consistency_file"
                ],
                "tables_dir": result_dir + "/tables",
                "force_table_download": file_path == files_to_check[0] and not resume,
            },
            "cf:": {
                "enable_appendix_a_checks": True,
            },
        }

    if len(files_to_check) == 0:
        raise Exception("No files found to check.")

    print("Files to check:")
    print(json.dumps(files_to_check, indent=4))
    print()
    print("Dataset - Files mapping (extended):")
    print(json.dumps(dataset_files_map_ext, indent=4))
    print()
    print("Dataset - Files mapping:")
    print(json.dumps(dataset_files_map, indent=4))
    print()
    print("Files to check dict:")
    print(json.dumps(files_to_check_dict, indent=4))
    print()

    #########################################################
    # QA Part 1 - Run all compliance-checker checks
    #########################################################

    summary = QAResultAggregator(checker_dict=checker_dict)

    # Run the first process:
    if len(files_to_check) > 0:
        processed_file, result_first = process_file(
            files_to_check[0],
            checkers,
            checker_options[files_to_check[0]],
            files_to_check_dict,
            processed_files,
            progress_file,
        )
        summary.update(
            result_first, files_to_check_dict[processed_file]["id"], processed_file
        )

    # Run the rest of the processes
    if len(files_to_check) > 1:

        # Calculate the number of processes
        num_processes = max(multiprocessing.cpu_count() - 4, 1)

        # Prepare the argument tuples
        args = [
            (
                x,
                checkers,
                checker_options[x],
                files_to_check_dict,
                processed_files,
                progress_file,
            )
            for x in files_to_check[1:]
        ]

        # Use a pool of workers to run jobs in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            # results = [result_first] + pool.starmap(
            #    process_file, args
            # )  # This collects all results in a list
            for processed_file, result in pool.imap_unordered(call_process_file, args):
                summary.update(
                    result, files_to_check_dict[processed_file]["id"], processed_file
                )

    #########################################################
    # QA Part 2 - Run all consistency checks
    #########################################################
    # todo

    #########################################################
    # Summarize and save results
    #########################################################

    # todo: always the latest checker version is used atm, but the
    #       specified version should be used ("tests")
    summary.sort()
    qc_summary = summary.summary
    get_checker_release_versions(checkers)
    qc_summary["info"] = {
        "id": "",
        "date": _timestamp_pprint,
        "files": str(len(files_to_check)),
        "datasets": str(len(dataset_files_map)),
        "cc_version": cc_version,
        "checkers": ", ".join(
            [
                f"{checker_dict.get(checker, '')} {checker}:{checker_release_versions[checker]}"
                for checker in checkers
            ]
        ),
    }
    dsid_common_prefix = os.path.commonprefix(list(dataset_files_map.keys()))
    if dsid_common_prefix != list(dataset_files_map.keys())[0]:
        dsid_common_prefix = dsid_common_prefix + "*"
    if info:
        qc_summary["info"]["id"] = f"{info} ({dsid_common_prefix})"
    else:
        qc_summary["info"]["id"] = f"{dsid_common_prefix}"

    # Save JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = "001"
    filename = f"qc_result_{file_id}_{timestamp}.json"
    with open(os.path.join(result_dir, filename), "w") as f:
        json.dump(qc_summary, f, indent=4, ensure_ascii=False, sort_keys=False)
    print(f"Saved QC result: {result_dir}/{filename}")

    # Save cluster
    qc_summary_clustered = summary.cluster_summary()
    qc_summary_clustered["info"] = qc_summary["info"]
    filename = f"qc_result_{file_id}_{timestamp}.cluster.json"
    with open(os.path.join(result_dir, filename), "w") as f:
        json.dump(
            qc_summary_clustered, f, indent=4, ensure_ascii=False, sort_keys=False
        )
    print(f"Saved QC cluster summary: {result_dir}/{filename}")

    # Save CSV file

    """
    for dataset_id in dataset_files.keys():
        for dataset_files in dataset_files_map[dataset_id]:
            external_check_process = multiprocessing.Process(target=process_dataset, args=(dataset_files, results_queue))
            external_check_processes.append(external_check_process)
            external_check_process.start()

    compliance_check_process.join()
    for external_check_process in external_check_processes:
        external_check_process.join()

    while not results_queue.empty():
        file_path, result = results_queue.get()
        if result is not None:
            all_results.append((file_path, result))
            with open(progress_file, 'a') as file:
                file.write(f"Check completed for file: {file_path}\n")
        else:
            with open(progress_file, 'a') as file:
                file.write(f"Skipped check for file: {file_path}\n")

    # Process all_results as needed
    for file_path, result in all_results:
        if result is not None:
            # Process each result
            pass

    # Additional processing of results if needed
    """


if __name__ == "__main__":
    main()
