import difflib
import re
from collections import defaultdict

from esgf_qa._constants import checker_dict, checker_dict_ext


class QAResultAggregator:
    """
    Aggregate, organize, and cluster the results of multiple ESGF-Quality Assurance (QA)
    or Climate Checker (cc) runs.

    This class collects the outcomes of compliance checker (cc) / cc-plugin runs from multiple datasets
    and files, normalizes them into a consistent internal summary structure, and provides
    functionality to sort, cluster, and generalize similar messages.

    Attributes
    ----------
    summary : dict of defaultdict
        Nested dictionary structure that stores the aggregated QA results.
        It contains two top-level keys:
            - ``"error"`` : maps checker functions to error messages → dataset IDs → file names.
            - ``"fail"``  : maps test weights → test names → messages → dataset IDs → file names.
    clustered_summary : dict of defaultdict
        Summary structure produced after clustering messages using
        :meth:`cluster_summary`. Keys and nesting mirror ``summary``, but
        messages are generalized and aggregated/clustered across similar text patterns.
    checker_dict : dict
        Mapping of checker identifiers to human-readable names, used
        for consistent labeling in summaries. Only cc checks.
    checker_dict_ext : dict
        Mapping of checker identifiers to human-readable names, used
        for consistent labeling in summaries. cc checks extended by esgf_qa checks.

    Methods
    -------
    update(result_dict, dsid, file_name)
        Update the summary with a single cc run result (i.e. for one file).
    update_ds(result_dict, dsid)
        Update the summary with results from a single inter-dataset or inter-file checker run
        that come with esgf-qa.
    sort()
        Sort the summary by test weight and test name for consistent output ordering.
    cluster_messages(messages, threshold)
        Cluster similar message strings based on edit-distance similarity.
    generalize_message_group(messages)
        Derive a generalized message template and placeholder map from a list of similar messages.
    merge_placeholders(list_of_strings, dictionary, skip=0)
        Helper to merge adjacent placeholders in message templates where possible.
    cluster_summary(threshold=0.75)
        Cluster and generalize all messages in the current summary using a similarity threshold.

    Examples
    --------
    >>> from esgf_qa._constants import checker_dict
    >>> agg = QAResultAggregator(checker_dict)
    >>> result = {
    ...     "cf": {
    ...         "test_1": {"value": (0, 1), "msgs": ["Missing attribute 'units'"]},
    ...     }
    ... }
    >>> agg.update(result, dsid="dataset_001", file_name="tas_day.nc")
    >>> agg.sort()
    >>> agg.cluster_summary(threshold=0.8)
    >>> agg.clustered_summary["fail"]
    {3: {'[CF-Conventions] test_1': {'Missing attribute {A} (1 occurrences, e.g. A=\'units\')': {...}}}}
    """

    def __init__(self):
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
        self.checker_dict_ext = checker_dict_ext

    def update(self, result_dict, dsid, file_name):
        """
        Update the summary with a single result of a cc-run.

        Parameters
        ----------
        result_dict : dict
            Dictionary containing the results of a single cc-run.
        dsid : str
            Dataset ID.
        file_name : str
            File name.
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

    def update_ds(self, result_dict, dsid):
        """
        Update the summary with a single result of an esgf-qa (inter-file/dataset) run.

        Parameters
        ----------
        result_dict : dict
            Dictionary containing the results of a single esgf-qa (inter-file/dataset) run.
        dsid : str
            Dataset ID.
        """
        for checker in result_dict:
            for test in result_dict[checker]:
                if test == "errors":
                    for function_name, errdict in result_dict[checker][
                        "errors"
                    ].items():
                        for file_name in errdict["files"]:
                            self.summary["error"][
                                f"[{checker_dict_ext[checker]}] " + function_name
                            ][errdict["msg"]][dsid].append(file_name)
                else:
                    weight = result_dict[checker][test].get("weight", 3)
                    fails = result_dict[checker][test].get("msgs", {})
                    for msg, file_names in fails.items():
                        for file_name in file_names:
                            self.summary["fail"][weight][
                                f"[{checker_dict_ext[checker]}] " + test
                            ][msg][dsid].append(file_name)

    def sort(self):
        """
        Sort the summary by test weight and test name for consistent output ordering.

        Modifies the `summary` attribute.
        """
        self.summary["fail"] = dict(sorted(self.summary["fail"].items(), reverse=True))
        for key in self.summary["fail"]:
            self.summary["fail"][key] = dict(sorted(self.summary["fail"][key].items()))

        # Sort errors by function name
        for checker in self.summary["error"]:
            self.summary["error"][checker] = dict(
                sorted(self.summary["error"][checker].items())
            )

    @staticmethod
    def cluster_messages(messages, threshold):
        """
        Cluster messages based on similarity.

        Parameters
        ----------
        messages : list
            List of messages to cluster.
        threshold : float
            Similarity threshold.

        Returns
        -------
        list
            List of clusters.
        """
        clusters = []
        while messages:
            base = messages.pop(0)
            cluster = [base]
            to_remove = []
            for msg in messages:
                ratio = difflib.SequenceMatcher(None, base, msg).ratio()
                if ratio >= threshold:
                    cluster.append(msg)
                    to_remove.append(msg)
            for msg in to_remove:
                messages.remove(msg)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def generalize_message_group(messages):
        """
        Generalize a group of messages.

        Parameters
        ----------
        messages : list
            List of messages to generalize.

        Returns
        -------
        str
            Generalized message.
        dict
            Placeholders.
        """
        if len(messages) == 1:
            return messages[0], {}

        # Split messages into tokens
        split_messages = [re.findall(r"\w+|\W", m) for m in messages]
        transposed = list(zip(*split_messages))
        template = []
        placeholders = {}
        var_index = 0

        for i, tokens in enumerate(transposed):
            unique_tokens = set(tokens)
            if len(unique_tokens) == 1:
                template.append(tokens[0])
            else:
                var_name = chr(ord("A") + var_index)
                template.append(f"{{{var_name}}}")
                placeholders[var_name] = tokens[0]
                var_index += 1

        # Merge placeholders if possible
        template, placeholders = QAResultAggregator.merge_placeholders(
            template, placeholders
        )

        # Return the generalized message and the placeholders
        generalized = "".join(template)
        return generalized, placeholders

    @staticmethod
    def merge_placeholders(list_of_strings, dictionary, skip=0):
        """
        Merge adjacent placeholders in message templates where possible.

        Avoids too many placeholders in a clustered message.

        Parameters
        ----------
        list_of_strings : list
            List of strings.
        dictionary : dict
            Dictionary of placeholders.
        skip : int, optional
            Number of placeholders to skip, by default 0.

        Returns
        -------
        list
            List of strings with placeholders merged.
        dict
            Dictionary of placeholders.
        """

        def find_next_two_placeholders(list_of_strings, skip):
            placeholders = [
                s for s in list_of_strings if s.startswith("{") and s.endswith("}")
            ]
            if len(placeholders) < 2:
                return None, None
            return placeholders[skip] if len(placeholders) >= skip + 1 else None, (
                placeholders[skip + 1] if len(placeholders) >= skip + 2 else None
            )

        def extract_text_between_placeholders(
            list_of_strings, placeholder1, placeholder2
        ):
            idx1 = list_of_strings.index(placeholder1)
            idx2 = list_of_strings.index(placeholder2)
            return "".join(list_of_strings[idx1 + 1 : idx2])

        def merge_two_placeholders(
            placeholder1, placeholder2, text_between, dictionary
        ):
            new_value = (
                dictionary[placeholder1.lstrip("{").rstrip("}")]
                + text_between
                + dictionary[placeholder2.lstrip("{").rstrip("}")]
            )
            dictionary[placeholder1.lstrip("{").rstrip("}")] = new_value
            del dictionary[placeholder2.lstrip("{").rstrip("}")]
            return dictionary

        def update_placeholder_names(list_of_strings, dictionary):
            old_placeholders = sorted(list(dictionary.keys()))
            new_placeholders = [
                chr(ord("A") + i) for i in range(0, len(old_placeholders))
            ]
            new_dictionary = dict(
                zip(new_placeholders, [dictionary[val] for val in old_placeholders])
            )
            for old, new in zip(old_placeholders, new_placeholders):
                list_of_strings = [
                    s.replace("{" + old + "}", "{" + new + "}") for s in list_of_strings
                ]
            return list_of_strings, new_dictionary

        def replace_placeholders_with_new_one(
            list_of_strings, placeholder1, placeholder2
        ):
            idx1 = list_of_strings.index(placeholder1)
            idx2 = list_of_strings.index(placeholder2)
            list_of_strings_new = list_of_strings[:idx1] + [placeholder1]
            if idx2 < len(list_of_strings) + 1:
                list_of_strings_new += list_of_strings[idx2 + 1 :]
            return list_of_strings_new

        if not any(s.startswith("{") and s.endswith("}") for s in list_of_strings):
            return list_of_strings, dictionary

        placeholder1, placeholder2 = find_next_two_placeholders(list_of_strings, skip)
        if placeholder1 is None or placeholder2 is None:
            return list_of_strings, dictionary

        text_between = extract_text_between_placeholders(
            list_of_strings, placeholder1, placeholder2
        )
        if len(text_between) < 5:
            dictionary = merge_two_placeholders(
                placeholder1, placeholder2, text_between, dictionary
            )
            list_of_strings = replace_placeholders_with_new_one(
                list_of_strings, placeholder1, placeholder2
            )
            list_of_strings, dictionary = update_placeholder_names(
                list_of_strings, dictionary
            )
            return QAResultAggregator.merge_placeholders(
                list_of_strings, dictionary, skip
            )
        else:
            return QAResultAggregator.merge_placeholders(
                list_of_strings, dictionary, skip + 1
            )

    def cluster_summary(self, threshold=0.75):
        """
        Cluster messages in the summary into groups of similar messages.

        Drastically reduces number of messages in the summary for datasets accumulating
        large numbers of check failure messages.

        Parameters
        ----------
        threshold : float, optional
            The threshold for similarity between messages, by default 0.75.

        Returns
        -------
        None
            Modifies the `clustered_summary` attribute.
        """
        self.clustered_summary = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        for status in self.summary:
            if status == "error":
                for test_id in self.summary[status]:
                    messages = list(self.summary[status][test_id].keys())
                    # Pass a copy of messages to cluster_messages to generate clusters
                    clusters = QAResultAggregator.cluster_messages(
                        messages[:], threshold
                    )

                    for cluster in clusters:
                        generalized, placeholders = (
                            QAResultAggregator.generalize_message_group(cluster)
                        )
                        example_parts = ", ".join(
                            [
                                (
                                    f"{k}='{v[0]}'"
                                    if isinstance(v, list)
                                    else f"{k}='{v}'"
                                )
                                for k, v in placeholders.items()
                            ]
                        )
                        if example_parts:
                            msg_summary = f"{generalized} ({len(cluster)} occurrences, e.g. {example_parts})"
                        else:
                            msg_summary = f"{generalized}{' (' + str(len(cluster)) + ' occurrences)' if len(cluster) > 1 else ''}"

                        # Gather all ds_ids and filenames across the cluster
                        combined = defaultdict(set)
                        for message in cluster:
                            for ds_id, files in self.summary[status][test_id][
                                message
                            ].items():
                                combined[ds_id].update(files)

                        # Shorten file lists to one example
                        formatted = {
                            ds_id
                            + " ("
                            + str(len(files))
                            + f" file{'s' if len(files) > 1 else ''} affected)": (
                                [f"e.g. '{next(iter(files))}'"]
                                if len(files) > 1
                                else [f"'{next(iter(files))}'"]
                            )
                            for ds_id, files in combined.items()
                        }

                        self.clustered_summary[status][test_id][msg_summary] = formatted
            elif status == "fail":
                for weight in self.summary[status]:
                    for test_id in self.summary[status][weight]:
                        messages = list(self.summary[status][weight][test_id].keys())
                        # Pass a copy of messages to cluster_messages to generate clusters
                        clusters = QAResultAggregator.cluster_messages(
                            messages[:], threshold
                        )

                        for cluster in clusters:
                            generalized, placeholders = (
                                QAResultAggregator.generalize_message_group(cluster)
                            )
                            example_parts = ", ".join(
                                [
                                    (
                                        f"{k}='{v[0]}'"
                                        if isinstance(v, list)
                                        else f"{k}='{v}'"
                                    )
                                    for k, v in placeholders.items()
                                ]
                            )
                            if example_parts:
                                msg_summary = f"{generalized} ({len(cluster)} occurrences, e.g. {example_parts})"
                            else:
                                msg_summary = f"{generalized}{' (' + str(len(cluster)) + ' occurrences)' if len(cluster) > 1 else ''}"

                            # Gather all ds_ids and filenames across the cluster
                            combined = defaultdict(set)
                            for message in cluster:
                                for ds_id, files in self.summary[status][weight][
                                    test_id
                                ][message].items():
                                    combined[ds_id].update(files)

                            # Shorten file lists to one example
                            formatted = {
                                ds_id
                                + " ("
                                + str(len(files))
                                + f" file{'s' if len(files) > 1 else ''} affected)": (
                                    [f"e.g. '{next(iter(files))}'"]
                                    if len(files) > 1
                                    else [f"'{next(iter(files))}'"]
                                )
                                for ds_id, files in combined.items()
                            }

                            self.clustered_summary[status][weight][test_id][
                                msg_summary
                            ] = formatted
