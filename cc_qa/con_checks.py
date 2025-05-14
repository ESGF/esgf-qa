import json
from collections import OrderedDict, defaultdict

import cftime
import xarray as xr

from cc_qa._constants import deltdic


def level2_factory():
    return defaultdict(list)


def level1_factory():
    return defaultdict(level2_factory)


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


def compare_dicts(dict1, dict2, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)

    # Get all keys that are in either dictionary, excluding the ones to skip
    all_keys = (set(dict1) | set(dict2)) - exclude_keys

    # Collect keys with differing values
    differing_keys = [key for key in all_keys if dict1.get(key) != dict2.get(key)]

    return differing_keys


def compare_nested_dicts(dict1, dict2, exclude_keys=None):
    diffs = {}

    all_root_keys = set(dict1) | set(dict2)

    for root_key in all_root_keys:
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
    results = defaultdict(level1_factory)
    filelist = sorted(ds_map[ds])
    consistency_files = OrderedDict(
        (files_to_check_dict[i]["consistency_file"], i) for i in filelist
    )

    # Exclude the following global attributes from comparison
    excl_global_attrs = ["creation_date", "history", "tracking_id"]

    # Exclude the following variable attributes from comparison
    excl_var_attrs = []

    # Exclude the following coordinates from comparison
    excl_coords = []

    # Compare each file with reference
    reference_file = list(consistency_files.keys())[0]
    with open(reference_file) as fr:
        reference_data = json.load(fr)
        for file in consistency_files.keys():
            if file == reference_file:
                continue
            with open(file) as fc:
                data = json.load(fc)

                # Compare required global attributes
                test = "Required global attributes"
                results[test]["weight"] = 3
                diff_keys = compare_dicts(
                    reference_data["global_attributes"],
                    data["global_attributes"],
                    exclude_keys=excl_global_attrs,
                )
                if diff_keys:
                    err_msg = "The following global attributes differ: " + ", ".join(
                        sorted(diff_keys)
                    )
                    results[test]["msgs"][err_msg].append(consistency_files[file])

                # Compare non-required global attributes
                test = "Non-required global attributes"
                results[test]["weight"] = 1
                diff_keys = compare_dicts(
                    reference_data["global_attributes_non_required"],
                    data["global_attributes_non_required"],
                    exclude_keys=excl_global_attrs,
                )
                if diff_keys:
                    err_msg = (
                        "The following non-required global attributes differ: "
                        + ", ".join(sorted(diff_keys))
                    )
                    results[test]["msgs"][err_msg].append(consistency_files[file])

                # Compare global attributes dtypes
                test = "Global attributes data types"
                results[test]["weight"] = 3
                diff_keys = compare_dicts(
                    reference_data["global_attributes_dtypes"],
                    data["global_attributes_dtypes"],
                    exclude_keys=[],
                )
                if diff_keys:
                    diff_keys = [
                        key
                        for key in diff_keys
                        if key in reference_data["global_attributes_dtypes"]
                        and key in data["global_attributes_dtypes"]
                    ]
                    if diff_keys:
                        err_msg = (
                            "The following global attributes have inconsistent data types: "
                            + ", ".join(sorted(diff_keys))
                        )
                        results[test]["msgs"][err_msg].append(consistency_files[file])

                # Compare variable attributes
                test = "Variable attributes"
                results[test]["weight"] = 3
                diff_keys = compare_nested_dicts(
                    reference_data["variable_attributes"],
                    data["variable_attributes"],
                    exclude_keys=excl_var_attrs,
                )
                if diff_keys:
                    for key, diff in diff_keys.items():
                        if diff:
                            err_msg = (
                                f"For variable '{key}' the following variable attributes differ: "
                                + ", ".join(sorted(diff))
                            )
                            results[test]["msgs"][err_msg].append(
                                consistency_files[file]
                            )
                        else:
                            err_msg = f"Variable '{key}' not present."
                            if key not in data["variable_attributes"]:
                                results[test]["msgs"][err_msg].append(
                                    consistency_files[file]
                                )
                            else:
                                results[test]["msgs"][err_msg].append(
                                    consistency_files[reference_file]
                                )

                # Compare variable attributes data types
                test = "Variable attributes data types"
                results[test]["weight"] = 3
                diff_keys = compare_nested_dicts(
                    reference_data["variable_attributes_dtypes"],
                    data["variable_attributes_dtypes"],
                    exclude_keys=[],
                )
                if diff_keys:
                    for key, diff in diff_keys.items():
                        if diff:
                            err_msg = (
                                f"For variable '{key}' the following variable attributes have inconsistent data types: "
                                + ", ".join(sorted(diff))
                            )
                            results[test]["msgs"][err_msg].append(
                                consistency_files[file]
                            )

                # Compare dimensions
                test = "Dimensions"
                results[test]["weight"] = 3
                diff_keys = compare_dicts(
                    reference_data["dimensions"],
                    data["dimensions"],
                    exclude_keys=[],
                )
                if diff_keys:
                    err_msg = "The following dimensions differ: " + ", ".join(
                        sorted(diff_keys)
                    )
                    results[test]["msgs"][err_msg].append(consistency_files[file])

                # Compare coordinates
                test = "Coordinates"
                results[test]["weight"] = 3
                diff_keys = compare_dicts(
                    reference_data["coordinates"],
                    data["coordinates"],
                    exclude_keys=excl_coords,
                )
                if diff_keys:
                    err_msg = "The following coordinates differ: " + ", ".join(
                        sorted(diff_keys)
                    )
                    results[test]["msgs"][err_msg].append(consistency_files[file])

    return results


def continuity_checks(ds, ds_map, files_to_check_dict, checker_options):
    results = defaultdict(level1_factory)
    filelist = sorted(ds_map[ds])
    consistency_files = OrderedDict(
        (files_to_check_dict[i]["consistency_file"], i) for i in filelist
    )

    # Check time and time_bnds continuity
    test = "Time continuity"
    results[test]["weight"] = 3
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
                if data["time_info"]["timen"]
                else None
            )
            boundn = (
                cftime.num2date(
                    data["time_info"]["boundn"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["boundn"]
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
                if data["time_info"]["time0"]
                else None
            )
            bound0 = (
                cftime.num2date(
                    data["time_info"]["bound0"],
                    units=data["time_info"]["units"],
                    calendar=data["time_info"]["calendar"],
                )
                if data["time_info"]["bound0"]
                else None
            )
            freq = data["time_info"]["frequency"]
            if (time0 or timen or bound0 or boundn) and not freq:
                err_msg = "Frequency could not be inferred"
                results[test]["msgs"][err_msg].append(consistency_files[file])
                continue
            elif (time0 or timen or bound0 or boundn) and freq not in deltdic:
                err_msg = f"Unsupported frequency '{freq}'"
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
    results = defaultdict(level1_factory)
    filelist = sorted(ds_map[ds])

    # open_mfdataset - override
    test = "xarray open_mfdataset - override"
    results[test]["weight"] = 3
    try:
        with xr.open_mfdataset(filelist, coords="minimal", compat="override") as ds:
            pass
    except Exception as e:
        results[test]["msgs"][str(e)].extend(filelist)

    # open_mfdataset - no_conflicts
    test = "xarray open_mfdataset - no_conflicts"
    results[test]["weight"] = 3
    try:
        with xr.open_mfdataset(filelist, coords="minimal", compat="no_conflicts") as ds:
            pass
    except Exception as e:
        results[test]["msgs"][str(e)].extend(filelist)

    return results
