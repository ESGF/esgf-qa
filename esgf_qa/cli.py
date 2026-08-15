"""Command-line parsing and run configuration."""

import argparse
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from esgf_qa._constants import checker_supporting_consistency_checks
from esgf_qa.checker_registry import (
    get_installed_checker_versions,
    normalize_checker_specs,
)
from esgf_qa.resume import (
    ResumeInfo,
    load_resume_info,
    prepare_result_directory,
    read_progress,
    verify_options_dict,
    write_resume_info,
)

# EERIE is the MIP checker with this site-specific CMOR table path as its default.
EERIE_TABLES = "/work/bm0021/cmor_tables/eerie_cmor_tables/Tables"


@dataclass
class RunConfig:
    """Validated configuration required by the QA workflow."""

    parent_dir: str
    result_dir: str
    checkers: list[str]
    info: str
    resume: bool
    include_consistency_checks: bool
    checker_options: dict
    parallel_processes: int
    time_checks_only: bool
    progress_file: Path
    dataset_file: Path
    processed_files: set[str]
    processed_datasets: set[str]
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    rerun_all: bool = False


def _nonempty_path_fragment(value):
    """Validate a literal path fragment supplied through the CLI."""
    if not value:
        raise argparse.ArgumentTypeError("path fragments must not be empty")
    return value


def parse_options(opts):
    """Parse ``checker:option[:value]`` CLI options into a nested mapping."""
    options_dict = defaultdict(dict)
    for option in opts:
        try:
            checker_type, checker_option, *checker_value = option.split(":", 2)
            checker_value = checker_value[0] if checker_value else True
        except ValueError as error:
            raise ValueError(
                f"Could not split option '{option}', seems illegally formatted. "
                "The required format is '<checker>:<option_name>[:<option_value>]', "
                "for example 'mip:tables:/path/to/Tables'."
            ) from error
        options_dict[checker_type][checker_option] = checker_value
    return options_dict


def build_parser(default_result_dir):
    """Build the ``esgqa`` argument parser."""
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
        default=default_result_dir,
        help=(
            "Directory to store QA results. Needs to be non-existing or empty "
            "or from a previous QA run."
        ),
    )
    parser.add_argument(
        "-O",
        "--option",
        default=[],
        action="append",
        help=(
            "Additional checker option in "
            "'<checker>:<option_name>[:<option_value>]' format. May be repeated."
        ),
    )
    parser.add_argument(
        "-t",
        "--test",
        action="append",
        help=(
            "Test in '<checker>[:<version>]' format. May be repeated. "
            "The default is 'cf:latest'; omitted versions mean 'latest'."
        ),
    )
    parser.add_argument(
        "-i",
        "--info",
        type=str,
        help="Information identifying the current run, such as an experiment_id.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Continue a previous QA run. Requires --output_dir.",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help=(
            "Rerun all checks from a previous QA run instead of reusing successful "
            "results. Requires --resume."
        ),
    )
    parser.add_argument(
        "-C",
        "--include_consistency_checks",
        action="store_true",
        help="Include consistency checks when the selected checker lacks support.",
    )
    parser.add_argument(
        "-P",
        "--parallel_processes",
        type=int,
        default=0,
        help="Maximum parallel processes. Default: 0 (= number of cores).",
    )
    parser.add_argument(
        "-w",
        "--whitelist",
        action="append",
        default=[],
        type=_nonempty_path_fragment,
        metavar="PATH_FRAGMENT",
        help=(
            "Only check files whose full path contains at least one case-sensitive "
            "literal path fragment. May be repeated."
        ),
    )
    parser.add_argument(
        "-b",
        "--blacklist",
        action="append",
        default=[],
        type=_nonempty_path_fragment,
        metavar="PATH_FRAGMENT",
        help=(
            "Exclude files whose full path contains a case-sensitive literal path "
            "fragment. May be repeated; blacklist matches take precedence."
        ),
    )
    return parser


def _validate_resume_arguments(parser, args):
    if args.rerun_all and not args.resume:
        parser.error("--rerun-all requires -r/--resume.")
    if not args.resume:
        return
    allowed = {
        "output_dir",
        "info",
        "resume",
        "rerun_all",
        "parallel_processes",
    }
    supplied = {
        key for key, value in vars(args).items() if value not in (None, False, [], "")
    }
    invalid = supplied - allowed
    if invalid:
        parser.error(
            "When using -r/--resume, the following arguments are not allowed: "
            f"{', '.join(sorted(invalid))}"
        )


def resolve_checker_specs(tests, checker_options):
    """Validate checker selections and resolve aliases to effective specs."""
    if not tests:
        return ["cf"]

    test_regex = re.compile(r"^[a-zA-Z0-9_-]+(?::(latest|[0-9]+(?:\.[0-9]+)*))?$")
    if not all(test_regex.match(test) for test in tests):
        raise Exception(
            "Invalid test(s) specified. Please use 'checker_name' or 'checker_name:version'."
        )
    checker_names = [test.split(":", 1)[0] for test in tests]
    if len(checker_names) != len(set(checker_names)):
        raise Exception("Cannot specify multiple instances of the same checker.")
    checker_versions = {
        name: (test.split(":", 1)[1] if ":" in test else "latest")
        for name, test in zip(checker_names, tests, strict=True)
    }

    installed_versions = get_installed_checker_versions()
    invalid_checkers = [
        name
        for name in checker_versions
        if name not in installed_versions and name != "eerie"
    ]
    invalid_versions = [
        name
        for name, requested_version in checker_versions.items()
        if name not in {"eerie", "cc6", "mip"}
        and name in installed_versions
        and requested_version not in installed_versions[name]
    ]
    messages = []
    if invalid_checkers:
        messages.append(
            f"The following checkers are not supported or installed: {', '.join(invalid_checkers)}."
        )
    for name in invalid_versions:
        messages.append(
            f"For checker {name}, supported/installed versions are: "
            f"{', '.join(installed_versions[name])}."
        )
    if messages:
        raise ValueError("ERROR: Invalid test(s) specified. " + " ".join(messages))

    for checker in ("cc6", "mip"):
        if checker in checker_versions and checker_versions[checker] != "latest":
            checker_versions[checker] = "latest"
            warnings.warn(
                f"Version of checker '{checker}' must be 'latest'. Using 'latest'."
            )

    mip_explicitly_requested = "mip" in checker_versions
    if mip_explicitly_requested and "eerie" in checker_versions:
        raise Exception(
            "ERROR: Cannot run both 'mip' and its 'eerie' alias at the same time."
        )
    mip_tables = checker_options.get("mip", {}).get("tables")
    if mip_explicitly_requested and (not isinstance(mip_tables, str) or not mip_tables):
        raise Exception(
            "Option 'tables' with a path to CMOR tables must be specified when "
            "checker 'mip' is explicitly selected."
        )

    if "eerie" in checker_versions:
        checker_versions["mip"] = "latest"
        del checker_versions["eerie"]
        if "eerie" in checker_options:
            checker_options["mip"] = checker_options.pop("eerie")
        checker_options["mip"].setdefault("tables", EERIE_TABLES)

    if {"mip", "cc6"}.issubset(checker_versions):
        raise Exception("ERROR: Cannot run both 'cc6' and 'mip' at the same time.")
    return normalize_checker_specs(checker_versions)


def prepare_run(default_result_dir, argv=None):
    """Parse CLI arguments and prepare a validated run configuration."""
    parser = build_parser(default_result_dir)
    args = parser.parse_args(argv)
    _validate_resume_arguments(parser, args)

    result_dir = os.path.abspath(args.output_dir)
    parent_dir = os.path.abspath(args.parent_dir) if args.parent_dir else None
    tests = sorted(args.test) if args.test else []
    info = args.info or ""
    whitelist = list(args.whitelist)
    blacklist = list(args.blacklist)
    checker_options = parse_options(args.option)
    progress_file = Path(result_dir, "progress.txt")
    dataset_file = Path(result_dir, "progress_datasets.txt")
    resume_info_file = Path(result_dir, ".resume_info")

    prepare_result_directory(result_dir, args.resume, progress_file, resume_info_file)
    include_consistency_checks = args.include_consistency_checks
    if args.resume:
        print(f"Resuming previous QA run in '{result_dir}'")
        stored = load_resume_info(resume_info_file, result_dir)
        tests = stored.tests
        parent_dir = stored.parent_dir
        if info and info != stored.info:
            warnings.warn(
                "<info> argument differs from the original value "
                f"('{stored.info}'). Using the new specification."
            )
        elif not info:
            info = stored.info
        checker_options = defaultdict(dict, stored.checker_options)
        include_consistency_checks = stored.include_consistency_checks
        whitelist = stored.whitelist
        blacklist = stored.blacklist
    else:
        print(f"Storing check results in '{result_dir}'")

    checkers = resolve_checker_specs(tests, checker_options)
    if parent_dir is None:
        parser.error("Missing required argument <parent_dir>.")
    if not os.path.exists(parent_dir):
        raise Exception(f"The specified <parent_dir> '{parent_dir}' does not exist.")

    write_resume_info(
        resume_info_file,
        ResumeInfo(
            parent_dir=parent_dir,
            info=info,
            tests=checkers,
            checker_options=dict(checker_options),
            include_consistency_checks=include_consistency_checks,
            whitelist=whitelist,
            blacklist=blacklist,
        ),
    )

    supports_consistency = any(
        checker.split(":", 1)[0] in checker_supporting_consistency_checks
        for checker in checkers
    )
    time_checks_only = include_consistency_checks and not supports_consistency
    if time_checks_only:
        checkers.append("mip")
        checkers.sort()

    Path(result_dir, "tables").mkdir(exist_ok=True)
    if args.rerun_all:
        # Rebuild progress from this invocation. If the forced run is interrupted,
        # a later normal resume will only reuse results completed by this run.
        progress_file.write_text("")
        dataset_file.write_text("")
    else:
        progress_file.touch()
        dataset_file.touch()
    return RunConfig(
        parent_dir=parent_dir,
        result_dir=result_dir,
        checkers=checkers,
        info=info,
        resume=args.resume,
        include_consistency_checks=include_consistency_checks,
        checker_options=checker_options,
        parallel_processes=args.parallel_processes,
        time_checks_only=time_checks_only,
        progress_file=progress_file,
        dataset_file=dataset_file,
        processed_files=read_progress(progress_file),
        processed_datasets=read_progress(dataset_file),
        whitelist=whitelist,
        blacklist=blacklist,
        rerun_all=args.rerun_all,
    )


# Compatibility alias for the former private name in run_qa.
_verify_options_dict = verify_options_dict
