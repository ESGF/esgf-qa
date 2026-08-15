"""Validate the contents of built ESGF-QA distributions."""

import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGE_RESOURCES = {
    "esgf_qa/resources/display_qc_results.html",
    "esgf_qa/resources/fonts/Inter-Regular.woff2",
    "esgf_qa/resources/fonts/Inter-SemiBold.woff2",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def require_single_artifact(dist_dir, pattern, label):
    """Return the single matching artifact or fail with a useful message."""
    artifacts = sorted(dist_dir.glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} matching '{pattern}', found {artifacts}."
        )
    return artifacts[0]


def check_wheel(wheel_path):
    """Check runtime resources, excluded tests, and generated version metadata."""
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        missing = PACKAGE_RESOURCES - names
        if missing:
            raise RuntimeError(f"Wheel is missing package resources: {sorted(missing)}")

        included_tests = sorted(name for name in names if name.startswith("tests/"))
        if included_tests:
            raise RuntimeError(
                f"Wheel unexpectedly contains the top-level tests package: {included_tests}"
            )

        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"Expected one wheel METADATA file, found {metadata_files}."
            )
        metadata = wheel.read(metadata_files[0]).decode("utf-8")
        version = next(
            (
                line.removeprefix("Version: ")
                for line in metadata.splitlines()
                if line.startswith("Version: ")
            ),
            None,
        )
        if version is None or version.split("+", 1)[0] in {"0.0.0", "999"}:
            raise RuntimeError(f"Wheel contains invalid fallback version '{version}'.")


def check_sdist(sdist_path):
    """Check that source distributions contain resources and the test suite."""
    with tarfile.open(sdist_path) as sdist:
        relative_names = {
            name.split("/", 1)[1] for name in sdist.getnames() if "/" in name
        }

    source_tests = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in REPOSITORY_ROOT.joinpath("tests").rglob("*.py")
    }
    required = PACKAGE_RESOURCES | source_tests
    missing = required - relative_names
    if missing:
        raise RuntimeError(f"Source distribution is missing: {sorted(missing)}")


def main():
    """Validate the wheel and sdist in the supplied distribution directory."""
    dist_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheel_path = require_single_artifact(dist_dir, "*.whl", "wheel")
    sdist_path = require_single_artifact(dist_dir, "*.tar.gz", "source distribution")

    check_wheel(wheel_path)
    check_sdist(sdist_path)
    print(f"Validated wheel: {wheel_path}")
    print(f"Validated source distribution: {sdist_path}")


if __name__ == "__main__":
    main()
