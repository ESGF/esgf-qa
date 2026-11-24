import json
import os
from collections import ChainMap, OrderedDict, defaultdict
from functools import lru_cache

import cftime
import xarray as xr
from netCDF4 import Dataset

from esgf_qa._constants import deltdic
from esgf_qa.consistency_config import (
    COMPARISON_CHECKS,
    ConsistencyConfig,
    load_consistency_config,
)


def level2_factory():
    return defaultdict(list)


def level1_factory():
    return defaultdict(level2_factory)


def level0_factory():
    return defaultdict(level1_factory)


@lru_cache(maxsize=1)
def _default_consistency_config():
    return load_consistency_config()


def _config(checker_options):
    """Resolve configuration while retaining the legacy empty-dict API."""
    if isinstance(checker_options, ConsistencyConfig):
        return checker_options
    return _default_consistency_config()


def _selected_dict(data, keys):
    return {key: data.get(key, "unset") for key in keys}


def printtimedelta(d):
    """Return timedelta (s) as either min, hours, days, whatever fits best."""
    if d > 86000:
        return f"{d/86400.} days"
    if d > 3500:
        return f"{d/3600.} hours"
    if d > 50:
        return f"{d/60.} minutes"
    else:
        return f"{d} seconds"


def truncate_str(s, max_length=16):
    """
    Truncate string if too long.

    Parameters
    ----------
    s : str
        String to truncate.
    max_length : int, optional
        Maximum length of string. Default is 16.

    Returns
    -------
    str
        Truncated string.

    Examples
    --------
    >>> truncate_str("This is a long string", 10)
    'This...string'
    >>> truncate_str("This is a short string", 16)
    'This is a short string'
    """
    if max_length <= 0 or max_length is None or len(s) <= max_length:
        return s

    # Select start and end of string
    words = s.split()
    start = ""
    end = ""

    for i in range(len(words)):
        if len(" ".join(words[: i + 1])) >= 6:
            start = " ".join(words[: i + 1])
            break

    for i in range(len(words) - 1, -1, -1):
        if len(" ".join(words[i:])) >= 6:
            end = " ".join(words[i:])
            break

    # Return truncated string
    if len(start) + len(end) + 3 >= len(s):
        return s
    else:
        return f"{start}...{end}"


def compare_dicts(dict1, dict2, exclude_keys=None):
    """
    Compare two dictionaries and return keys with differing values.

    Parameters
    ----------
    dict1 : dict
        First dictionary to compare.
    dict2 : dict
        Second dictionary to compare.
    exclude_keys : list, optional
        List of keys to exclude from comparison.

    Returns
    -------
    list
        List of keys with differing values.
    """
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)

    # Get all keys that are in either dictionary, excluding the ones to skip
    all_keys = (set(dict1) | set(dict2)) - exclude_keys

    # Collect keys with differing values
    differing_keys = [
        key for key in sorted(all_keys) if dict1.get(key) != dict2.get(key)
    ]

    return differing_keys


def compare_nested_dicts(dict1, dict2, exclude_keys=None):
    """
    Compare two nested dictionaries and return keys with differing values.

    Parameters
    ----------
    dict1 : dict
        First dictionary to compare.
    dict2 : dict
        Second dictionary to compare.
    exclude_keys : list, optional
        List of keys to exclude from comparison.

    Returns
    -------
    dict
        Dictionary of keys with differing values.
    """
    diffs = {}

    all_root_keys = set(dict1) | set(dict2)

    for root_key in sorted(all_root_keys):
        subdict1 = dict1.get(root_key, {})
        subdict2 = dict2.get(root_key, {})

        if not isinstance(subdict1, dict) or not isinstance(subdict2, dict):
            if subdict1 != subdict2:
                diffs[root_key] = []
            continue

        diffs_k = compare_dicts(subdict1, subdict2, exclude_keys)

        if diffs_k:
            diffs[root_key] = diffs_k

    return diffs


def consistency_checks(ds, ds_map, files_to_check_dict, checker_options):
    """
    Consistency checks.

    Runs inter-file consistency checks on a dataset:

        - Global attributes (values and data types)
        - Variable attributes (values and data types)
        - Coordinates (values)
        - Dimensions (names and sizes)

    Parameters
    ----------
    ds : str
        Dataset to process.
    ds_map : dict
        Dictionary mapping dataset IDs to file paths.
    files_to_check_dict : dict
        A special dictionary mapping files to check to datasets.
    checker_options : dict
        Dictionary of checker options.

    Returns
    -------
    dict
        A dictionary containing the results of the consistency checks.
    """
    results = defaultdict(level1_factory)
    config = _config(checker_options)
    filelist = sorted(ds_map[ds])
    consistency_files = OrderedDict(
        (files_to_check_dict[i]["consistency_file"], i)
        for i in filelist
        if os.path.isfile(files_to_check_dict[i]["consistency_file"])
    )

    if not consistency_files:
        return results

    reference_file = next(iter(consistency_files))
    with open(reference_file) as stream:
        reference_data = json.load(stream)

    for file in consistency_files:
        if file == reference_file:
            continue
        with open(file) as stream:
            data = json.load(stream)

        simple_checks = (
            (
                "required_global_attributes",
                "Required global attributes",
                "global_attributes",
                "The following global attributes differ: ",
            ),
            (
                "non_required_global_attributes",
                "Non-required global attributes",
                "global_attributes_non_required",
                "The following non-required global attributes differ: ",
            ),
            (
                "dimensions",
                "Dimensions",
                "dimensions",
                "The following dimensions differ: ",
            ),
            (
                "coordinates",
                "Coordinates",
                "coordinates",
                "The following coordinates differ: ",
            ),
        )
        for name, test, data_key, message in simple_checks:
            rule = config.check("within_dataset", name)
            if not rule.enabled:
                continue
            results[test]["weight"] = rule.severity
            diff_keys = compare_dicts(
                reference_data[data_key],
                data[data_key],
                exclude_keys=rule.allow_differences,
            )
            if diff_keys:
                results[test]["msgs"][message + ", ".join(diff_keys)].append(
                    consistency_files[file]
                )

        rule = config.check("within_dataset", "selected_global_attributes")
        if rule.enabled:
            test = "Selected global attributes"
            results[test]["weight"] = rule.severity
            diff_keys = compare_dicts(
                _selected_dict(
                    _combined_global_attributes(reference_data), rule.compare_only
                ),
                _selected_dict(_combined_global_attributes(data), rule.compare_only),
                exclude_keys=rule.allow_differences,
            )
            if diff_keys:
                message = "The following selected global attributes differ: "
                results[test]["msgs"][message + ", ".join(diff_keys)].append(
                    consistency_files[file]
                )

        rule = config.check("within_dataset", "global_attribute_types")
        if rule.enabled:
            test = "Global attributes data types"
            results[test]["weight"] = rule.severity
            diff_keys = compare_dicts(
                reference_data["global_attributes_dtypes"],
                data["global_attributes_dtypes"],
                exclude_keys=rule.allow_differences,
            )
            # Missing optional attributes are reported by the value checks, not
            # again as inconsistent Python data types.
            diff_keys = [
                key
                for key in diff_keys
                if key in reference_data["global_attributes_dtypes"]
                and key in data["global_attributes_dtypes"]
            ]
            if diff_keys:
                message = (
                    "The following global attributes have inconsistent data types: "
                    + ", ".join(diff_keys)
                )
                results[test]["msgs"][message].append(consistency_files[file])

        nested_checks = (
            (
                "variable_attributes",
                "Variable attributes",
                "variable_attributes",
                "differ",
            ),
            (
                "variable_attribute_types",
                "Variable attributes data types",
                "variable_attributes_dtypes",
                "have inconsistent data types",
            ),
        )
        for name, test, data_key, description in nested_checks:
            rule = config.check("within_dataset", name)
            if not rule.enabled:
                continue
            results[test]["weight"] = rule.severity
            diff_keys = compare_nested_dicts(
                reference_data[data_key],
                data[data_key],
                exclude_keys=rule.allow_differences,
            )
            for variable, differences in diff_keys.items():
                if differences:
                    message = (
                        f"For variable '{variable}' the following variable "
                        f"attributes {description}: " + ", ".join(differences)
                    )
                    results[test]["msgs"][message].append(consistency_files[file])
                elif name == "variable_attributes":
                    message = f"Variable '{variable}' not present."
                    affected = (
                        consistency_files[file]
                        if variable not in data[data_key]
                        else consistency_files[reference_file]
                    )
                    results[test]["msgs"][message].append(affected)

    return results


def continuity_checks(ds, ds_map, files_to_check_dict, checker_options):
    """
    Checks inter-file time and time_bnds continuity for a dataset.

    This check identifies gaps in time or time_bnds between files of a dataset.

    Parameters
    ----------
    ds : str
        Dataset to process.
    ds_map : dict
        Dictionary mapping dataset IDs to file paths.
    files_to_check_dict : dict
        A special dictionary mapping files to check to datasets.
    checker_options : dict
        Dictionary of checker options.

    Returns
    -------
    dict
        Dictionary of results.
    """
    results = defaultdict(level1_factory)
    rule = _config(checker_options).check("within_dataset", "time_continuity")
    if not rule.enabled:
        return results
    filelist = sorted(ds_map[ds])
    consistency_files = OrderedDict(
        (files_to_check_dict[i]["consistency_file"], i)
        for i in filelist
        if os.path.isfile(files_to_check_dict[i]["consistency_file"])
    )

    # Check time and time_bnds continuity
    test = "Time continuity"
    results[test]["weight"] = rule.severity
    timen = None
    boundn = None
    i = 0
    for file in consistency_files.keys():
        with open(file) as fc:
            data = json.load(fc)
            i += 1
            prev_timen = timen
            prev_boundn = boundn
            timen = (
                cftime.num2date(
                    data["time_info"]["timen"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["timen"] is not None
                and data["time_info"]["units"]
                and data["time_info"]["calendar"]
                else None
            )
            boundn = (
                cftime.num2date(
                    data["time_info"]["boundn"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["boundn"] is not None
                and data["time_info"]["units"]
                and data["time_info"]["calendar"]
                else None
            )
            if i == 1:
                continue
            time0 = (
                cftime.num2date(
                    data["time_info"]["time0"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["time0"] is not None
                and data["time_info"]["units"]
                and data["time_info"]["calendar"]
                else None
            )
            bound0 = (
                cftime.num2date(
                    data["time_info"]["bound0"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["bound0"] is not None
                and data["time_info"]["units"]
                and data["time_info"]["calendar"]
                else None
            )
            freq = data["time_info"]["frequency"]
            if (time0 or timen or bound0 or boundn) and not freq:
                err_msg = "Frequency could not be inferred"
                results[test]["msgs"][err_msg].append(consistency_files[file])
                continue
            elif (time0 or timen or bound0 or boundn) and freq not in deltdic:
                err_msg = f"Unsupported frequency '{freq}'"
                results[test]["msgs"][err_msg].append(consistency_files[file])
                continue

            if time0 and prev_timen:
                delt = time0 - prev_timen
                delts = delt.total_seconds()
                if delts > deltdic[freq + "max"] or delts < deltdic[freq + "min"]:
                    err_msg = f"Gap in time axis (between files) - previous {prev_timen} - current {time0} - delta-t {printtimedelta(delts)}"
                    results[test]["msgs"][err_msg].append(consistency_files[file])

            if bound0 and prev_boundn:
                delt_bnd = bound0 - prev_boundn
                delts_bnd = delt_bnd.total_seconds()
                if delts_bnd < -1:
                    err_msg = f"Overlapping time bounds (between files) - previous {prev_boundn} - current {bound0} - delta-t {printtimedelta(delts_bnd)}"
                    results[test]["msgs"][err_msg].append(consistency_files[file])
                if delts_bnd > 1:
                    err_msg = f"Gap in time bounds (between files) - previous {prev_boundn} - current {bound0} - delta-t {printtimedelta(delts_bnd)}"
                    results[test]["msgs"][err_msg].append(consistency_files[file])

    return results


def compatibility_checks(ds, ds_map, files_to_check_dict, checker_options):
    """
    Compatibility checks for a dataset.

    Checks for:

        - xarray open_mfdataset (compat='override', join='outer')
        - xarray open_mfdataset (compat='no_conflicts', join='exact')

    Parameters
    ----------
    ds : str
        Dataset to process.
    ds_map : dict
        Dictionary mapping dataset IDs to file paths.
    files_to_check_dict : dict
        A special dictionary mapping files to check to datasets.
    checker_options : dict
        Dictionary of checker options.

    Returns
    -------
    dict
        Dictionary of results.
    """
    results = defaultdict(level1_factory)
    filelist = sorted(ds_map[ds])

    config = _config(checker_options)

    rule = config.check("within_dataset", "compatibility_override")
    if rule.enabled:
        test = "xarray open_mfdataset (compat='override', join='outer')"
        results[test]["weight"] = rule.severity
        try:
            with xr.open_mfdataset(
                filelist,
                coords="minimal",
                compat="override",
                data_vars="all",
                join="outer",
            ) as ds:
                pass
        except Exception as e:
            results[test]["msgs"][str(e)].extend(filelist)

    # open_mfdataset - no_conflicts
    rule = config.check("within_dataset", "compatibility_exact")
    if rule.enabled:
        test = "xarray open_mfdataset (compat='no_conflicts', join='exact')"
        results[test]["weight"] = rule.severity
        try:
            with xr.open_mfdataset(
                filelist,
                coords="minimal",
                compat="no_conflicts",
                data_vars="all",
                join="exact",
            ) as ds:
                pass
        except Exception as e:
            results[test]["msgs"][str(e)].extend(filelist)

    return results


def _dataset_consistency_data(ds_map, files_to_check_dict):
    """Load the first available consistency output for every dataset."""
    filedict = {}
    consistency_data = {}
    for dataset_id in sorted(ds_map):
        for file_path in sorted(ds_map[dataset_id]):
            consistency_file = files_to_check_dict[file_path]["consistency_file"]
            if not os.path.isfile(consistency_file):
                continue
            with open(consistency_file) as stream:
                consistency_data[dataset_id] = json.load(stream)
            filedict[dataset_id] = file_path
            break
    return filedict, consistency_data


def _combined_global_attributes(data):
    return ChainMap(
        data.get("global_attributes", {}),
        data.get("global_attributes_non_required", {}),
    )


def _group_key(dataset_id, data, group_name, group):
    """Return an internal and human-readable key for one configured group."""
    attributes = _combined_global_attributes(data)
    values = []
    for alternatives in group.attributes:
        value = next(
            (
                attributes[name]
                for name in alternatives
                if attributes.get(name) not in (None, "", "unset")
            ),
            None,
        )
        if value is None:
            if group.on_missing == "error":
                raise ValueError(
                    f"Dataset '{dataset_id}' has none of the attributes "
                    f"{', '.join(alternatives)} required by group '{group_name}'."
                )
            value = (
                f"<missing:{dataset_id}:{'/'.join(alternatives)}>"
                if group.on_missing == "separate"
                else "Default"
            )
        values.append(str(value))
    return tuple(values), "/".join(values)


def _group_datasets(dataset_ids, consistency_data, group_name, group):
    groups = defaultdict(list)
    labels = {}
    for dataset_id in sorted(dataset_ids):
        key, label = _group_key(
            dataset_id,
            consistency_data.get(dataset_id, {}),
            group_name,
            group,
        )
        groups[key].append(dataset_id)
        labels[key] = label
    labels_by_text = defaultdict(list)
    for key, label in labels.items():
        labels_by_text[label].append(key)
    for colliding_keys in labels_by_text.values():
        if len(colliding_keys) > 1:
            for key in colliding_keys:
                labels[key] = json.dumps(key, ensure_ascii=False)
    return groups, labels


def dataset_coverage_checks(ds_map, files_to_check_dict, checker_options):
    """
    Checks consistency of dataset time coverage.

    Variables that differ in their time coverage are reported.

    Parameters
    ----------
    ds_map : dict
        Dictionary mapping dataset IDs to file paths.
    files_to_check_dict : dict
        A special dictionary mapping files to check to datasets.
    checker_options : dict
        Dictionary of checker options.

    Returns
    -------
    dict
        Dictionary of results.
    """
    results = defaultdict(level0_factory)
    config = _config(checker_options)
    configured_rules = []
    for scope in ("within_group", "across_all"):
        rule = config.check(scope, "time_coverage")
        if rule.enabled:
            configured_rules.append((scope, rule))
    if not configured_rules:
        return results

    coverage_start = {}
    coverage_end = {}
    coverage_errors = {}
    for dataset_id in sorted(ds_map):
        files = sorted(ds_map[dataset_id])
        start_year = None
        end_year = None
        start_timestamp = None
        try:
            timestamp = files_to_check_dict[files[0]]["ts"]
            if timestamp:
                start_timestamp = timestamp.split("-")[0]
                start_year = int(start_timestamp[0:4])
                if len(start_timestamp) >= 6 and start_timestamp[4:6] != "01":
                    start_year += 1
                coverage_start[dataset_id] = start_year

            timestamp = files_to_check_dict[files[-1]]["ts"]
            if timestamp:
                # A single label denotes a point; filename syntax is validated
                # separately by the project's DRS checks.
                parts = timestamp.split("-")
                end_timestamp = parts[1] if len(parts) > 1 else parts[0]
                end_year = int(end_timestamp[0:4])
                # Retain the January boundary adjustment for time series, but
                # do not move a single identical start/end label into last year.
                if (
                    len(end_timestamp) >= 6
                    and end_timestamp[4:6] == "01"
                    and end_timestamp != start_timestamp
                ):
                    end_year -= 1
                coverage_end[dataset_id] = end_year

            if start_year is None and end_year is None:
                continue
            if start_year is None:
                coverage_errors[dataset_id] = (
                    "Begin of time coverage cannot be inferred.",
                    [files[0]],
                )
            elif end_year is None:
                coverage_errors[dataset_id] = (
                    "End of time coverage cannot be inferred.",
                    [files[-1]],
                )
        except (IndexError, ValueError):
            affected = [files[0], files[-1]] if len(files) > 1 else [files[0]]
            coverage_errors[dataset_id] = (
                "Time coverage cannot be inferred.",
                affected,
            )

    consistency_data = {}
    if any(scope == "within_group" for scope, _ in configured_rules):
        _, consistency_data = _dataset_consistency_data(ds_map, files_to_check_dict)
    multiple_scopes = len(configured_rules) > 1
    for scope, rule in configured_rules:
        test = "Time coverage"
        if multiple_scopes:
            test += " (Dataset Group)" if scope == "within_group" else " (All Datasets)"

        for dataset_id, (message, files) in coverage_errors.items():
            results[dataset_id][test]["weight"] = rule.severity
            results[dataset_id][test]["msgs"][message] = files

        if scope == "across_all":
            groups = {("all",): sorted(ds_map)}
        else:
            group = config.groups[rule.group]
            groups, _ = _group_datasets(ds_map, consistency_data, rule.group, group)

        for dataset_ids in groups.values():
            starts = [
                coverage_start[dataset_id]
                for dataset_id in dataset_ids
                if dataset_id in coverage_start
            ]
            ends = [
                coverage_end[dataset_id]
                for dataset_id in dataset_ids
                if dataset_id in coverage_end
            ]
            if len(starts) <= 1:
                continue
            expected_start = min(starts)
            expected_end = max(ends) if ends else None
            for dataset_id in dataset_ids:
                files = sorted(ds_map[dataset_id])
                if (
                    dataset_id in coverage_start
                    and coverage_start[dataset_id] != expected_start
                ):
                    results[dataset_id][test]["weight"] = rule.severity
                    results[dataset_id][test]["msgs"][
                        f"Time series starts at '{coverage_start[dataset_id]}' while other time series start at '{expected_start}'"
                    ] = [files[0]]
                if (
                    expected_end is not None
                    and dataset_id in coverage_end
                    and coverage_end[dataset_id] != expected_end
                ):
                    results[dataset_id][test]["weight"] = rule.severity
                    results[dataset_id][test]["msgs"][
                        f"Time series ends at '{coverage_end[dataset_id]}' while other time series end at '{expected_end}'"
                    ] = [files[-1]]

    return results


def inter_dataset_consistency_checks(ds_map, files_to_check_dict, checker_options):
    """
    Inter-dataset consistency checks.

    Groups datasets according to the active consistency profile for checks whose
    scope is ``within_group``.
    Runs inter-dataset consistency checks:

        - Required and non-required global attributes (values and data types)
        - Coordinates (values)
        - Dimensions (names and sizes)

    Parameters
    ----------
    ds_map : dict
        Dictionary mapping dataset IDs to file paths.
    files_to_check_dict : dict
        A special dictionary mapping files to check to datasets.
    checker_options : dict
        Dictionary of checker options.

    Returns
    -------
    dict
        Dictionary of results.
    """
    results = defaultdict(level0_factory)
    config = _config(checker_options)
    if not any(
        config.enabled(scope, name)
        for scope in ("within_group", "across_all")
        for name in COMPARISON_CHECKS
    ):
        return results, {}
    try:
        filedict, consistency_data = _dataset_consistency_data(
            ds_map, files_to_check_dict
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load consistency output: {error}") from error
    if not consistency_data:
        return results, {}

    labels = {
        "required_global_attributes": "Required global attributes",
        "non_required_global_attributes": "Non-required global attributes",
        "global_attribute_types": "Global attributes data types",
        "selected_global_attributes": "Group-specific global attributes",
        "variable_attributes": "Variable attributes",
        "variable_attribute_types": "Variable attributes data types",
        "dimensions": "Dimensions",
        "coordinates": "Coordinates",
    }
    data_keys = {
        "required_global_attributes": "global_attributes",
        "non_required_global_attributes": "global_attributes_non_required",
        "global_attribute_types": "global_attributes_dtypes",
        "variable_attributes": "variable_attributes",
        "variable_attribute_types": "variable_attributes_dtypes",
        "dimensions": "dimensions",
        "coordinates": "coordinates",
    }

    scalar_names = {}
    coordinate_references = {}

    def get_scalar_names(data):
        dataset_id = next(key for key, value in consistency_data.items() if value is data)
        if dataset_id not in scalar_names:
            # Metadata only: do not load any coordinate or data arrays.
            with Dataset(filedict[dataset_id]) as dataset:
                scalar_names[dataset_id] = {
                    name for name, variable in dataset.variables.items()
                    if not variable.dimensions
                }
        return scalar_names[dataset_id]

    def run_comparison(
        scope, name, rule, reference, current, dataset_id, group_label=None
    ):
        test = f"{labels[name]} (Inter-Dataset)"
        if scope == "within_group" and config.enabled("across_all", name):
            test = f"{labels[name]} (Inter-Dataset, Group)"
        elif scope == "across_all" and config.enabled("within_group", name):
            test = f"{labels[name]} (Inter-Dataset, All)"

        if name == "selected_global_attributes":
            reference_values = _selected_dict(
                _combined_global_attributes(reference), rule.compare_only
            )
            current_values = _selected_dict(
                _combined_global_attributes(current), rule.compare_only
            )
            diff_keys = compare_dicts(
                reference_values,
                current_values,
                exclude_keys=rule.allow_differences,
            )
        elif name == "coordinates":
            # A coordinate absent from the group's first dataset still needs
            # its own reference. IDs distinguish, for example, different uses
            # of 'sector', and allow matching coordinates with different names.
            references = coordinate_references.setdefault(
                (scope, rule.group, group_label), {}
            )
            ids = current.get("coordinate_ids") or {}
            diff_keys = []
            for coordinate, checksum in sorted(current["coordinates"].items()):
                if coordinate in rule.allow_differences:
                    continue
                identity = ids.get(coordinate) or None
                for (ref_name, ref_id), (ref_data, ref_checksum) in references.items():
                    if identity is not None and ref_id is not None:
                        if identity != ref_id:
                            continue
                    else:
                        if coordinate != ref_name:
                            continue
                        # Legacy summaries cannot distinguish scalar meanings.
                        # With matching IDs, even scalar values are comparable.
                        if rule.shared_non_scalar_only and (
                            coordinate in get_scalar_names(current)
                            or ref_name in get_scalar_names(ref_data)
                        ):
                            continue
                    if checksum != ref_checksum:
                        diff_keys.append(coordinate)
                        break
                references.setdefault((coordinate, identity), (current, checksum))
        elif name in {"variable_attributes", "variable_attribute_types"}:
            diff_keys = compare_nested_dicts(
                reference[data_keys[name]],
                current[data_keys[name]],
                exclude_keys=rule.allow_differences,
            )
        else:
            diff_keys = compare_dicts(
                reference[data_keys[name]],
                current[data_keys[name]],
                exclude_keys=rule.allow_differences,
            )
        if not diff_keys:
            return

        results[dataset_id][test]["weight"] = rule.severity
        context = f" in group '{truncate_str(group_label)}'" if group_label else ""
        if name in {"variable_attributes", "variable_attribute_types"}:
            description = (
                "have inconsistent data types"
                if name == "variable_attribute_types"
                else "differ"
            )
            for variable, differences in diff_keys.items():
                detail = ", ".join(differences) if differences else "variable missing"
                message = (
                    f"For variable '{variable}' the following variable attributes "
                    f"{description} between datasets{context}: {detail}"
                )
                results[dataset_id][test]["msgs"][message].append(filedict[dataset_id])
            return

        subject = labels[name].lower()
        message = (
            f"The following {subject} differ between datasets{context}: "
            + ", ".join(diff_keys)
        )
        results[dataset_id][test]["msgs"][message].append(filedict[dataset_id])

    enabled_across = {
        name: config.check("across_all", name)
        for name in sorted(COMPARISON_CHECKS)
        if config.enabled("across_all", name)
    }
    reference_datasets = {}
    dataset_ids = sorted(consistency_data)
    if enabled_across:
        main_reference = dataset_ids[0]
        reference_datasets["general_reference"] = main_reference
        for dataset_id in dataset_ids:
            for name, rule in enabled_across.items():
                if dataset_id == main_reference and name != "coordinates":
                    continue
                run_comparison(
                    "across_all",
                    name,
                    rule,
                    consistency_data[main_reference],
                    consistency_data[dataset_id],
                    dataset_id,
                )

    group_references = {}
    enabled_group_rules = {
        name: config.check("within_group", name)
        for name in sorted(COMPARISON_CHECKS)
        if config.enabled("within_group", name)
    }
    for group_name in sorted({rule.group for rule in enabled_group_rules.values()}):
        group = config.groups[group_name]
        grouped, group_labels = _group_datasets(
            dataset_ids, consistency_data, group_name, group
        )
        references = {}
        for key in sorted(grouped):
            grouped_ids = grouped[key]
            reference_id = grouped_ids[0]
            references[group_labels[key]] = reference_id
            for dataset_id in grouped_ids:
                for name, rule in enabled_group_rules.items():
                    if rule.group != group_name:
                        continue
                    if dataset_id == reference_id and name != "coordinates":
                        continue
                    run_comparison(
                        "within_group",
                        name,
                        rule,
                        consistency_data[reference_id],
                        consistency_data[dataset_id],
                        dataset_id,
                        group_labels[key],
                    )
        group_references[group_name] = references
    if group_references:
        reference_datasets["group_references"] = group_references

    print("The following datasets were used as reference:")
    if "general_reference" in reference_datasets:
        print(f" - General reference: {reference_datasets['general_reference']}")
    for group_name, references in group_references.items():
        for key, reference in sorted(references.items()):
            print(f" - '{truncate_str(key)}' ({group_name} group): {reference}")

    reference_file_exceptions = {
        ds: {
            "default_reference_file": sorted(ds_map[ds])[0],
            "used_reference_file": reference_file,
        }
        for ds, reference_file in filedict.items()
        if reference_file != sorted(ds_map[ds])[0]
    }
    if reference_file_exceptions:
        reference_datasets["reference_file_exceptions"] = reference_file_exceptions

    print()

    return results, reference_datasets
