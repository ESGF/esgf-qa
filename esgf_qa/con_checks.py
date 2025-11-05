import json
from collections import ChainMap, OrderedDict, defaultdict

import cftime
import xarray as xr

from esgf_qa._constants import deltdic


def level2_factory():
    return defaultdict(list)


def level1_factory():
    return defaultdict(level2_factory)


def level0_factory():
    return defaultdict(level1_factory)


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
        key for key in sorted(list(all_keys)) if dict1.get(key) != dict2.get(key)
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

    for root_key in sorted(list(all_root_keys)):
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
                    exclude_keys=["time"],
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
                if data["time_info"]["boundn"]
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
                if data["time_info"]["time0"]
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
                if data["time_info"]["bound0"]
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

    # open_mfdataset - override
    test = "xarray open_mfdataset (compat='override', join='outer')"
    results[test]["weight"] = 3
    try:
        with xr.open_mfdataset(
            filelist, coords="minimal", compat="override", data_vars="all", join="outer"
        ) as ds:
            pass
    except Exception as e:
        results[test]["msgs"][str(e)].extend(filelist)

    # open_mfdataset - no_conflicts
    test = "xarray open_mfdataset (compat='no_conflicts', join='exact')"
    results[test]["weight"] = 3
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
    test = "Time coverage"

    coverage_start = dict()
    coverage_end = dict()

    # Extract time coverage for each dataset
    for ds in ds_map.keys():
        fl = sorted(ds_map[ds])
        ts0 = None
        tsn = None
        try:
            if files_to_check_dict[fl[0]]["ts"] != "":
                ts0 = files_to_check_dict[fl[0]]["ts"].split("-")[0][0:4]
                # If time interval of timestamp does not start in January, use following year
                if len(files_to_check_dict[fl[-1]]["ts"].split("-")[0]) >= 6:
                    if files_to_check_dict[fl[-1]]["ts"].split("-")[0][4:6] != "01":
                        coverage_start[ds] = int(ts0) + 1
                    else:
                        coverage_start[ds] = int(ts0)
                coverage_start[ds] = int(ts0)
            if files_to_check_dict[fl[-1]]["ts"] != "":
                tsn = files_to_check_dict[fl[-1]]["ts"].split("-")[1][0:4]
                # If time interval of timestamp ends in January, use previous year
                if len(files_to_check_dict[fl[-1]]["ts"].split("-")[1]) >= 6:
                    if files_to_check_dict[fl[-1]]["ts"].split("-")[1][4:6] == "01":
                        coverage_end[ds] = int(tsn) - 1
                    else:
                        coverage_end[ds] = int(tsn)
                else:
                    coverage_end[ds] = int(tsn)
            if ts0 is None and tsn is None:
                continue
            elif ts0 is None:
                results[ds][test]["weight"] = 1
                results[ds][test]["msgs"][
                    "Begin of time coverage cannot be inferred."
                ] = [fl[0]]
                continue
            elif tsn is None:
                results[ds][test]["weight"] = 1
                results[ds][test]["msgs"][
                    "End of time coverage cannot be inferred."
                ] = [fl[-1]]
                continue
        except IndexError or ValueError:
            results[ds][test]["weight"] = 1
            if len(fl) > 1:
                results[ds][test]["msgs"]["Time coverage cannot be inferred."] = [
                    fl[0],
                    fl[-1],
                ]
            else:
                results[ds][test]["msgs"]["Time coverage cannot be inferred."] = [fl[0]]
            continue

    # Compare coverage
    if len(coverage_start.keys()) > 1:
        try:
            scov = min(coverage_start.values())
        except ValueError:
            scov = None
        try:
            ecov = max(coverage_end.values())
        except ValueError:
            ecov = None
        # Get all ds where coverage_start differs
        for ds in coverage_start.keys():
            fl = sorted(ds_map[ds])
            if scov is None:
                pass
            elif coverage_start[ds] != scov:
                results[ds][test]["weight"] = 1
                results[ds][test]["msgs"][
                    f"Time series starts at '{coverage_start[ds]}' while other time series start at '{scov}'"
                ] = [fl[0]]
            if ecov is None:
                pass
            elif ds in coverage_end and coverage_end[ds] != ecov:
                results[ds][test]["weight"] = 1
                results[ds][test]["msgs"][
                    f"Time series ends at '{coverage_end[ds]}' while other time series end at '{ecov}'"
                ] = [fl[-1]]

    return results


def inter_dataset_consistency_checks(ds_map, files_to_check_dict, checker_options):
    """
    Inter-dataset consistency checks.

    Will group datasets by realm and grid for certain checks.
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
    filedict = {}
    consistency_data = {}
    for ds in ds_map.keys():
        filedict[ds] = sorted(ds_map[ds])[0]

    # Exclude the following global attributes from comparison
    excl_global_attrs = [
        "creation_date",
        "history",
        "tracking_id",
        "variable_id",
        "frequency",
        "external_variables",
        "table_id",
        "grid",
        "grid_label",
        "realm",
        "modeling_realm",
    ]

    # Include the following global attributes in the realm-specific comparison
    incl_global_attrs = ["grid", "grid_label", "realm", "modeling_realm"]

    # Consistency data
    for ds, dsfile0 in filedict.items():
        consistency_file = files_to_check_dict[dsfile0]["consistency_file"]
        with open(consistency_file) as f:
            data = json.load(f)
            consistency_data[ds] = data

    # Reference datasets
    ref_ds = dict()

    # Compare each file with reference
    for ds, data in consistency_data.items():
        # Select first dataset as main reference
        if "Main" not in ref_ds:
            ref_ds["Main"] = ds
        # Also group datasets by realm and grid label
        #   for grid / realm specific consistency checks
        realm = ChainMap(
            data["global_attributes"], data["global_attributes_non_required"]
        ).get("realm", None)
        if not realm:
            realm = ChainMap(
                data["global_attributes"], data["global_attributes_non_required"]
            ).get("modeling_realm", None)
        if not realm:
            realm = "Default"
        gridlabel = ChainMap(
            data["global_attributes"], data["global_attributes_non_required"]
        ).get("grid_label", None)
        if not gridlabel:
            gridlabel = ChainMap(
                data["global_attributes"], data["global_attributes_non_required"]
            ).get("grid", None)
        if not gridlabel:
            gridlabel = "Default"
        ref_ds_key = f"{realm}/{gridlabel}"
        if ref_ds_key not in ref_ds:
            ref_ds[ref_ds_key] = ds
            continue
        else:
            reference_data_rg = consistency_data[ref_ds[ref_ds_key]]
            reference_data = consistency_data[ref_ds["Main"]]

            # Compare required global attributes
            test = "Required global attributes (Inter-Dataset)"
            results[ds][test]["weight"] = 2
            diff_keys = compare_dicts(
                reference_data["global_attributes"],
                data["global_attributes"],
                exclude_keys=excl_global_attrs,
            )
            if diff_keys:
                err_msg = (
                    "The following global attributes differ between datasets: "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

            # Compare specific global attributes
            test = "Realm-specific global attributes (Inter-Dataset)"
            results[ds][test]["weight"] = 2
            diff_keys = compare_dicts(
                {
                    k: ChainMap(
                        reference_data_rg["global_attributes"],
                        reference_data_rg["global_attributes_non_required"],
                    ).get(k, "unset")
                    for k in incl_global_attrs
                },
                {
                    k: ChainMap(
                        data["global_attributes"],
                        data["global_attributes_non_required"],
                    ).get(k, "unset")
                    for k in incl_global_attrs
                },
                exclude_keys=[],
            )
            if diff_keys:
                err_msg = (
                    f"The following realm-specific global attributes differ between datasets (realm/grid_label: {truncate_str(ref_ds_key.split('/')[0])}/{truncate_str(ref_ds_key.split('/')[1])}): "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

            # Compare non-required global attributes
            test = "Non-required global attributes (Inter-Dataset)"
            results[ds][test]["weight"] = 1
            diff_keys = compare_dicts(
                reference_data["global_attributes_non_required"],
                data["global_attributes_non_required"],
                exclude_keys=excl_global_attrs,
            )
            if diff_keys:
                err_msg = (
                    "The following non-required global attributes differ between datasets: "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

            # Compare global attributes dtypes
            test = "Global attributes data types (Inter-Dataset)"
            results[ds][test]["weight"] = 2
            diff_keys = compare_dicts(
                reference_data["global_attributes_dtypes"],
                data["global_attributes_dtypes"],
                exclude_keys=[],
            )
            if diff_keys:
                err_msg = (
                    "The following global attributes have inconsistent data types between datasets: "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

            # Compare dimensions
            test = "Dimensions (Inter-Dataset)"
            results[ds][test]["weight"] = 2
            diff_keys = compare_dicts(
                reference_data_rg["dimensions"],
                data["dimensions"],
                exclude_keys=["time", "depth", "lev"],
            )
            if diff_keys:
                err_msg = (
                    "The following dimensions differ between datasets: "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

            # Compare coordinates
            test = "Coordinates (Inter-Dataset)"
            results[ds][test]["weight"] = 2
            diff_keys = compare_dicts(
                reference_data_rg["coordinates"],
                data["coordinates"],
                exclude_keys=[
                    "depth",
                    "depth_bnds",
                    "lev",
                    "lev_bnds",
                    "plev",
                    "height",
                ],
            )
            if diff_keys:
                err_msg = (
                    "The following coordinates differ between datasets: "
                    + ", ".join(sorted(diff_keys))
                )
                results[ds][test]["msgs"][err_msg].append(filedict[ds])

    # List reference datasets
    print("The following datasets were used as reference:")
    print(f" - General reference: {ref_ds['Main']}")
    reference_datasets = {"general_reference": ref_ds["Main"]}
    for key in sorted(list(ref_ds.keys())):
        if key == "Main":
            continue
        else:
            reference_datasets[key] = ref_ds[key]
            print(
                f" - '{truncate_str(key.split('/')[0])}' / '{truncate_str(key.split('/')[1])}' (realm / grid): {ref_ds[key]}"
            )

    print()

    return results, reference_datasets
