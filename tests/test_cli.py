import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


class TestQACommandLine:
    """
    End-to-end pytest test class for esgqa CLI using synthetic CMIP6 and CORDEX-CMIP6 data.
    """

    @classmethod
    def setup_class(cls):
        """
        Generate lightweight synthetic CMIP6 and CORDEX-CMIP6 test datasets.
        """
        cls.test_data_dir = tempfile.mkdtemp(prefix="esgf_qa_testdata_")
        cls.cmip6_dir = os.path.join(cls.test_data_dir, "cmip6")
        cls.cordex_dir = os.path.join(cls.test_data_dir, "cordex_cmip6")
        cls.custom_dir = os.path.join(cls.test_data_dir, "custom")
        os.makedirs(cls.cmip6_dir, exist_ok=True)
        os.makedirs(cls.cordex_dir, exist_ok=True)

        # Generate lightweight CMIP6 test data
        for var in ["tas", "huss"]:
            base_path = (
                Path(cls.cmip6_dir)
                / f"MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/{var}/gn/v20210215"
            )
            base_path.mkdir(parents=True, exist_ok=True)
            for start_year in [1850, 1855]:
                ntime = 60  # 5 years monthly data
                times = np.array(np.arange(ntime), dtype=np.float64)
                lats = np.arange(-90, 91, 10)
                lons = np.arange(0, 360, 10)
                data = np.zeros((len(times), len(lats), len(lons)))
                ds = xr.Dataset(
                    {var: (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons},
                )
                file_name = f"{var}_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{start_year:04d}01-"
                file_name += f"{start_year+4:04d}12.nc"
                ds.to_netcdf(base_path / file_name)

        # Generate lightweight CORDEX-CMIP6 test data
        for var in ["ta600", "tas"]:
            base_path = (
                Path(cls.cordex_dir)
                / f"DD/EUR-12/CLMcom-DWD/MPI-ESM1-2-HR/historical/r1i1p1f1/ICON-CLM-202407-1-1/v1-r1/mon/{var}/v20240920"
            )
            base_path.mkdir(parents=True, exist_ok=True)
            for start_year, end_year in [(1950, 1950), (1951, 1960)]:
                ntime = (end_year - start_year + 1) * 12
                times = np.array(np.arange(ntime), dtype=np.float64)
                rlat = np.arange(0, 41, 10)
                rlon = np.arange(0, 41, 10)
                data = np.zeros((len(times), len(rlat), len(rlon)))
                ds = xr.Dataset(
                    {var: (("time", "rlat", "rlon"), data)},
                    coords={"time": times, "rlat": rlat, "rlon": rlon},
                )
                file_name = f"{var}_EUR-12_MPI-ESM1-2-HR_historical_r1i1p1f1_CLMcom-DWD_ICON-CLM-202407-1-1_v1-r1_mon_{start_year:04d}01-{end_year:04d}12.nc"
                ds.to_netcdf(base_path / file_name)

        # Generate lightweight custom data
        for var in ["temp2", "huss"]:
            base_path = Path(cls.custom_dir) / "model_output"
            base_path.mkdir(parents=True, exist_ok=True)
            for start_year in range(1850, 1860):
                times = np.arange(0, 12)  # 1 years monthly data
                lats = np.arange(-90, 91, 10)
                lons = np.arange(0, 360, 10)
                data = np.zeros((len(times), len(lats), len(lons)))
                ds = xr.Dataset(
                    {var: (("time", "lat", "lon"), data)},
                    coords={"time": times, "lat": lats, "lon": lons},
                )
                file_name = f"{var}_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{start_year:04d}01-"
                file_name += f"{start_year+4:04d}12.nc"
                ds.to_netcdf(base_path / file_name)

    @classmethod
    def teardown_class(cls):
        """Clean up temporary test data."""
        shutil.rmtree(cls.test_data_dir)

    def _run_cli(self, args, expect_error=False, expected_err_msg=None):
        """Run the esgqa CLI and optionally check for errors."""
        cmd = ["python", "-m", "esgf_qa.run_qa"] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if expect_error:
            assert (
                result.returncode != 0
            ), f"Expected error but CLI succeeded:\n{result.stdout}\n{result.stderr}"
            if expected_err_msg:
                combined = result.stdout + "\n" + result.stderr
                assert (
                    expected_err_msg in combined
                ), f"Expected error message '{expected_err_msg}' not found.\nOutput:\n{combined}"
        else:
            assert (
                result.returncode == 0
            ), f"CLI failed unexpectedly:\n{result.stdout}\n{result.stderr}"
        return result.stdout, result.stderr

    @pytest.mark.parametrize(
        "test_args",
        [
            ["-t", "cc6:latest", "-o", "OUTPUT", "cmip6"],
            ["-t", "cc6", "-o", "OUTPUT", "cordex_cmip6"],
            ["-t", "cc6:latest", "-t", "cf", "-o", "OUTPUT", "cordex_cmip6"],
            ["-t", "cf:latest", "-o", "OUTPUT", "cmip6"],
            ["-t", "cf:1.7", "-C", "-o", "OUTPUT", "cmip6"],
            [
                "-t",
                "wcrp_cmip6:latest",
                "-t",
                "cf:1.7",
                "-o",
                "OUTPUT",
                "cmip6",
                "-i",
                "test_info",
            ],
            [
                "-t",
                "wcrp_cordex_cmip6",
                "-t",
                "cf:1.7",
                "-o",
                "OUTPUT",
                "cordex_cmip6",
                "-i",
                "test_info",
            ],
        ],
    )
    def test_cli_runs_successfully(self, test_args, tmp_path):
        temp_dir = tempfile.mkdtemp()
        try:
            result_dir = tmp_path / "results"
            args = [
                (
                    os.path.join(self.test_data_dir, "cmip6")
                    if arg == "cmip6"
                    else (
                        os.path.join(self.test_data_dir, "cordex_cmip6")
                        if arg == "cordex_cmip6"
                        else arg.replace("OUTPUT", str(result_dir))
                    )
                )
                for arg in test_args
            ]
            stdout, stderr = self._run_cli(args)
            output_dir_index = args.index("-o") + 1
            result_dir = args[output_dir_index]
            result_files = os.listdir(result_dir)
            assert any(
                f.startswith("qa_result_") and f.endswith(".json") for f in result_files
            )

            # Check clustered summary if exists
            clustered_files = [
                f for f in result_files if "clustered" in f and f.endswith(".json")
            ]
            for cf in clustered_files:
                with open(os.path.join(result_dir, cf)) as f:
                    data = json.load(f)
                for key in ["error", "fail", "info"]:
                    assert key in data
                info = data["info"]
                for field in [
                    "id",
                    "date",
                    "files",
                    "datasets",
                    "cc_version",
                    "checkers",
                ]:
                    assert field in info
                for sev_dict in [data["fail"], data["error"]]:
                    for _, issues in sev_dict.items():
                        for issue_name, messages in issues.items():
                            for msg, files in messages.items():
                                assert isinstance(files, list)
                                assert (
                                    len(files) == 1
                                ), f"Clustered summary should have one example file for {msg}"
                                assert isinstance(files[0], str)
        finally:
            shutil.rmtree(temp_dir)

    def test_cli_resume_functionality(self):
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        resume_file = os.path.join(output_dir, ".resume_info")
        Path(os.path.join(output_dir, "progress.txt")).touch()
        Path(os.path.join(output_dir, "progress_datasets.txt")).touch()
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        with open(resume_file, "w") as f:
            json.dump(
                {
                    "parent_dir": self.cmip6_dir,
                    "info": "test_resume",
                    "tests": ["cf:latest"],
                },
                f,
            )
        stdout, stderr = self._run_cli(["-r", "-o", output_dir])
        assert "Resuming previous QA run" in stdout
        shutil.rmtree(temp_dir)

    @pytest.mark.parametrize(
        "test_args, expected_err_msg",
        [
            (
                ["-t", "cf:latest", "-o", "some_dir"],
                "Missing required argument <parent_dir>",
            ),
            (
                ["-t", "invalid_checker:latest", "-o", "some_dir", "cmip6"],
                "Invalid test(s) specified",
            ),
            (
                ["-r", "-t", "cf:latest", "-o", "some_dir"],
                "When using -r/--resume, only -o/--output_dir and -i/--info can be set",
            ),
        ],
    )
    def test_cli_fails_on_invalid_arguments(self, test_args, expected_err_msg):
        temp_dir = tempfile.mkdtemp()
        try:
            args = [arg if arg != "cmip6" else self.cmip6_dir for arg in test_args]
            self._run_cli(args, expect_error=True, expected_err_msg=expected_err_msg)
        finally:
            shutil.rmtree(temp_dir)

    def test_cli_produces_valid_json(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_dir = temp_dir / "output"
            output_dir.mkdir()
            self._run_cli(
                ["-t", "cf:latest", "-o", str(output_dir), str(self.cmip6_dir)]
            )
            json_files = list(output_dir.glob("*.json"))
            assert len(json_files) == 2
            with open(json_files[0]) as f:
                data = json.load(f)
            # "info" is the only required field
            assert "info" in data
            # "error" and "fail" are optional, others are not allowed
            assert all([key in ["fail", "info", "error"] for key in data])
            info = data["info"]
            for field in ["id", "date", "files", "datasets", "cc_version", "checkers"]:
                assert field in info
            assert isinstance(data.get("error", {}), dict)
            assert isinstance(data.get("fail", {}), dict)
        finally:
            shutil.rmtree(temp_dir)
