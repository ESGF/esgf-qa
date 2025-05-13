import json
from collections import defaultdict


def level2_factory():
    return defaultdict(list)


def level1_factory():
    return defaultdict(level2_factory)


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


def consistency_check(ds, ds_map, files_to_check_dict, checker_options):
    results = defaultdict(level1_factory)
    filelist = sorted(ds_map[ds])
    consistency_files = {
        files_to_check_dict[i]["consistency_file"]: i for i in filelist
    }

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

                # Compare dimensions
                # Compare non-required global attributes

                # Compare required global attributes
                test = "Global attributes"
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
