from collections import defaultdict

import pytest

import esgf_qa.cluster_results as esgqacr
from esgf_qa.cluster_results import QAResultAggregator


@pytest.fixture(autouse=True)
def patch_checker_dicts(monkeypatch):
    """
    Patch module-level checker_dict and checker_dict_ext
    to avoid dependency on real ESGF constants.
    """
    mock_checker_dict = {"cf": "CF", "cc6": "C-C6"}
    mock_checker_dict_ext = {"cf": "CF-EXT", "cc6": "C-C6-EXT"}
    monkeypatch.setattr(esgqacr, "checker_dict", mock_checker_dict)
    monkeypatch.setattr(esgqacr, "checker_dict_ext", mock_checker_dict_ext)
    yield


@pytest.fixture
def aggregator():
    """Provide a fresh aggregator instance."""
    return QAResultAggregator()


def test_initial_summary_structure(aggregator):
    """Ensure the summary structure initializes correctly."""
    assert "error" in aggregator.summary
    assert "fail" in aggregator.summary
    assert isinstance(aggregator.summary["fail"], defaultdict)


def test_update_adds_fail_entries(aggregator):
    """Verify that a failed test adds entries to the summary."""
    result_dict = {
        "cf": {
            "check_units": {
                "value": (0, 1),
                "weight": 2,
                "msgs": ["Missing attribute 'units'"],
            }
        }
    }

    aggregator.update(result_dict, dsid="ds1", file_name="file1.nc")

    fail_summary = aggregator.summary["fail"]
    assert 2 in fail_summary
    test_name = "[CF] check_units"
    assert test_name in fail_summary[2]
    assert "Missing attribute 'units'" in fail_summary[2][test_name]
    assert "ds1" in fail_summary[2][test_name]["Missing attribute 'units'"]
    assert "file1.nc" in fail_summary[2][test_name]["Missing attribute 'units'"]["ds1"]


def test_update_adds_error_entries(aggregator):
    """Verify that an error test adds entries to the summary."""
    result_dict = {"cf": {"errors": {"test_func": "Some internal error"}}}

    aggregator.update(result_dict, dsid="dsX", file_name="fX.nc")

    error_summary = aggregator.summary["error"]
    assert "[CF] test_func" in error_summary
    assert "Some internal error" in error_summary["[CF] test_func"]
    assert "dsX" in error_summary["[CF] test_func"]["Some internal error"]
    assert "fX.nc" in error_summary["[CF] test_func"]["Some internal error"]["dsX"]


def test_update_ds_uses_checker_dict_ext(aggregator):
    """Ensure update_ds uses checker_dict_ext for extended checkers."""
    result_dict = {
        "cf": {
            "errors": {
                "check1": {"msg": "Something broke", "files": ["fileA.nc", "fileB.nc"]}
            },
            "test2": {"weight": 3, "msgs": {"Bad value": ["fileC.nc"]}},
        }
    }

    aggregator.update_ds(result_dict, dsid="dataset_42")

    error_summary = aggregator.summary["error"]
    fail_summary = aggregator.summary["fail"]

    # Check both sections populated and use extended prefix
    assert any("[CF-EXT]" in key for key in error_summary.keys())
    assert any("[CF-EXT]" in key for key in fail_summary[3].keys())


def test_sort_orders_failures_by_weight(aggregator):
    """Check that sorting produces a descending order by weight."""
    aggregator.summary["fail"][1]["[CF] test1"] = {}
    aggregator.summary["fail"][5]["[CF] test5"] = {}
    aggregator.sort()
    weights = list(aggregator.summary["fail"].keys())
    assert weights == sorted(weights, reverse=True)


def test_cluster_messages_basic():
    """Cluster messages with small differences using threshold."""
    messages = [
        "Missing value for var1",
        "Missing value for var2",
        "Completely different",
    ]
    clusters = QAResultAggregator.cluster_messages(messages[:], threshold=0.8)

    # Expect two clusters: similar ones together
    assert len(clusters) == 2
    assert any("var1" in msg or "var2" in msg for msg in clusters[0])


def test_generalize_message_group_single():
    """If there is one message, return it unchanged."""
    msg, placeholders = QAResultAggregator.generalize_message_group(["Missing X"])
    assert msg == "Missing X"
    assert placeholders == {}


def test_generalize_message_group_multiple():
    """Generalization should replace differing tokens with placeholders."""
    msgs = ["Missing variable A", "Missing variable B"]
    generalized, placeholders = QAResultAggregator.generalize_message_group(msgs)
    assert "Missing variable" in generalized
    assert "{" in generalized
    assert isinstance(placeholders, dict)
    assert list(placeholders.keys())  # at least one placeholder


def test_merge_placeholders_merges_close():
    """Test merging adjacent placeholders."""
    tokens = ["{A}", "-", "{B}"]
    dictionary = {"A": "foo", "B": "bar"}
    merged_tokens, merged_dict = QAResultAggregator.merge_placeholders(
        tokens, dictionary
    )
    # The placeholders should merge since only one char between them
    assert len(merged_dict) <= 1
    assert "{" in merged_tokens[0]


def test_cluster_summary_produces_clustered_summary(aggregator):
    """Integration-like test for cluster_summary on simple data."""
    result_dict = {
        "cf": {
            "check_attrs": {
                "value": (0, 1),
                "weight": 3,
                "msgs": ["Missing attr 'long_name'", "Missing attr 'standard_name'"],
            }
        }
    }
    aggregator.update(result_dict, dsid="ds1", file_name="file1.nc")
    aggregator.sort()
    aggregator.cluster_summary(threshold=0.7)
    clustered = aggregator.clustered_summary["fail"]

    # should contain weight 3 and a generalized message
    assert 3 in clustered
    test_name = next(iter(clustered[3].keys()))
    assert "[CF]" in test_name
    # at least one generalized message with "Missing attr"
    found_msg_keys = list(clustered[3]["[CF] check_attrs"].keys())
    assert any("Missing attr" in k for k in found_msg_keys)
