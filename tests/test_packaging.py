from importlib.resources import files


def test_web_viewer_resources_are_packaged():
    """The standalone HTML viewer and its fonts are package resources."""
    resources = files("esgf_qa").joinpath("resources")
    viewer = resources.joinpath("display_qc_results.html")

    assert viewer.is_file()
    assert resources.joinpath("fonts/Inter-Regular.woff2").is_file()
    assert resources.joinpath("fonts/Inter-SemiBold.woff2").is_file()
    assert "fonts/Inter-Regular.woff2" in viewer.read_text(encoding="utf-8")
    assert "fonts/Inter-SemiBold.woff2" in viewer.read_text(encoding="utf-8")
    assert "Consistency Configuration" in viewer.read_text(encoding="utf-8")


def test_consistency_profiles_are_packaged():
    """Default and CMIP7 consistency profiles are package resources."""
    profiles = files("esgf_qa").joinpath("resources/consistency")

    assert profiles.joinpath("default.toml").is_file()
    assert profiles.joinpath("cmip7.toml").is_file()
    assert profiles.joinpath("cordex_cmip6.toml").is_file()
