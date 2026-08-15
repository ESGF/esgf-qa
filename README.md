[![PyPI version](https://img.shields.io/pypi/v/esgf-qa.svg)](https://pypi.org/project/esgf-qa/)

# esgf-qa
### Quality Assurance Workflow Based on `compliance-checker` and `cc-plugin-wcrp` (or other cc-plugins)
<img src="https://raw.githubusercontent.com/ESGF/esgf-qa/master/docs/esgf-qa_Logo.png" align="left" width="120">

`esgf-qa` provides a flexible quality assurance (QA) workflow for evaluating dataset compliance using the
[ioos/compliance-checker](https://github.com/ioos/compliance-checker) framework
(including [CF](https://cfconventions.org/) compliance checks)
and any community plugins (`cc-plugin`s), such as
[ESGF/cc-plugin-wcrp](https://github.com/ESGF/cc-plugin-wcrp) and
[euro-cordex/cc-plugin-cc6](https://github.com/euro-cordex/cc-plugin-cc6).

The tool executes file-based quality control (QC) tests through the Compliance Checker,
and, where applicable, performs additional dataset-level checks to test inter-file time-axis continuity
and consistency in variable, coordinate and attribute definitions.
Results from both file- and dataset-level checks are aggregated, summarized, and clustered for easier interpretation.

### Currently supported checkers

While `esgf-qa` has been primarily developed for workflows assessing compliance with WCRP project data specifications
(e.g., CMIP, CORDEX), it can also be used for general CF-compliance testing and generally supports any
`cc-plugin`. It can be easily extended to support any projects following CORDEX- or CMIP-style CMOR table conventions.

| Standard                                                                                             | Checker Name |
| ---------------------------------------------------------------------------------------------------- | ------------ |
| [CF Conventions](https://cfconventions.org/) (shipped with [ioos/compliance-checker](https://github.com/ioos/compliance-checker)) | cf |
| [WCRP CMIP6](https://pcmdi.llnl.gov/CMIP6/):<br><ul><li>[CMIP6 DRS](https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf)</li><li>[CMIP6 CVs](https://github.com/WCRP-CMIP/CMIP6_CVs) (esgvoc)</li></li><li>[cmip6-cmor-tables](https://github.com/PCMDI/cmip6-cmor-tables) (esgvoc)</li></ul> | wcrp_cmip6 |
| [WCRP CMIP6Plus](https://wcrp-cmip.org/cmip-phases/cmip6plus/):<br><ul><li>[CMIP6 DRS](https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf)</li><li>[CMIP6Plus CVs](https://github.com/WCRP-CMIP/CMIP6Plus_CVs) (esgvoc)</li></li><li>[mip-cmor-tables](https://github.com/PCMDI/mip-cmor-tables) (esgvoc)</li></ul> | wcrp_cmip6plus |
| [WCRP CMIP7](https://wcrp-cmip.org/cmip-phases/cmip7/) ([CMIP7 Guidance](https://wcrp-cmip.github.io/cmip7-guidance/docs/)):<br><ul><li>[CMIP7 DRS](https://doi.org/10.5281/zenodo.17250296)</li><li>[CMIP7 CVs](https://github.com/WCRP-CMIP/CMIP7_CVs) (esgvoc)</li></li><li>[cmip7-cmor-tables](https://github.com/WCRP-CMIP/cmip7-cmor-tables) (esgvoc)</li></ul> | wcrp_cmip7 |
| [WCRP CORDEX-CMIP6](https://cordex.org/):<br><ul><li>[CORDEX-CMIP6 Archive Specifications](https://doi.org/10.5281/zenodo.10961069)</li><li>[cordex-cmip6-cv](https://github.com/WCRP-CORDEX/cordex-cmip6-cv) (esgvoc)</li><li>[cordex-cmip6-cmor-tables](https://github.com/WCRP-CORDEX/cordex-cmip6-cmor-tables) (esgvoc)</li></ul> |  wcrp_cordex_cmip6 |
|  [WCRP CORDEX-CMIP6](https://cordex.org/):<br><ul><li>[CORDEX-CMIP6 Archive Specifications](https://doi.org/10.5281/zenodo.10961069)</li><li>[cordex-cmip6-cv](https://github.com/WCRP-CORDEX/cordex-cmip6-cv)</li><li>[cordex-cmip6-cmor-tables](https://github.com/WCRP-CORDEX/cordex-cmip6-cmor-tables)</li></ul>  | cc6 |
| [EERIE](https://eerie-project.eu/):<br>[EERIE CMOR Tables & CV](https://github.com/eerie-project/dreq_tools) | eerie |
| Custom MIP (CMOR/MIP tables have to be specified) | mip |

## Installation

### Pip installation

```shell
$ pip install esgf-qa
```

### Pip installation from source

Clone the repository and `cd` into the repository folder, then:
```shell
$ pip install -e .
```

Optionally install the dependencies for development:
```shell
$ pip install -e .[dev]
```

See the [ioos/compliance-checker](https://github.com/ioos/compliance-checker#installation) for
additional Installation notes if problems arise with the dependencies.

### Installation and setup of `esgvoc`

The `cc-plugin-wcrp` checker plugins require the `esgvoc` software to be installed and setup:
```
pip install esgvoc
```

Run `esgvoc use <project>@latest` for the projects specifications you want to verify against:
```
esgvoc use universe@latest
esgvoc use cmip6@latest cmip6plus@latest cmip7@latest cordex-cmip6@latest
```

Please make sure to keep both, `esgvoc` and the project specifications up-to-date by running the following before conducting a QC-run for a simulation:
```
pip install --upgrade esgvoc
esgvoc update
```

- Test your installation

The following command should now also list the `cc-plugin-wcrp` checks next to all `cc_plugin_cc6` and `compliance_checker` checks:
```
cchecker.py -l
```

The following command should now list the necessary projects with metadata sources for `esgvoc`:
```
esgvoc status
```

The complete test suite is included in the source distribution. After unpacking
it, users can verify the installed code with:

```shell
pip install ".[test]"
pytest
```

Please see the [esgvoc user guide](https://esgf.github.io/esgf-vocab/user/introduction.html) for more information.

## Usage

```shell
$ esgqa [-h] [-P <parallel_processes>] [-o <OUTPUT_DIR>] [-t <TEST>] [-O OPTION] [-i <INFO>] [-r] [--rerun-all] [-C] [-w PATH_FRAGMENT] [-b PATH_FRAGMENT] <parent_dir>
```

- positional arguments:
  - `parent_dir`: Parent directory to scan for netCDF-files to check
- options:
  - `-h, --help`: show this help message and exit
  - `-P, --parallel_processes`: Specify the maximum number of parallel processes. Default: 0 (= number of cores).
  - `-o, --output_dir OUTPUT_DIR`: Directory to store QA results. Needs to be non-existing or empty or from previous QA run. If not specified, will store results in `./cc-qa-check-results/YYYYMMDD-HHmm_<hash>`.
  - `-t, --test TEST`: The test to run (eg. `'wcrp_cmip6:latest'`, `'wcrp_cordex_cmip6:latest'` or `'cf:<version>'`, can be specified multiple times, eg.: `'-t wcrp_cmip6:latest -t cf:1.7'`) - default: running latest CF checks. If the version is omitted, `latest` will be used (`'cf'` and `'cf:latest'` are equivalent).
  - `-O, --option OPTION`: Additional options to be passed to the checkers. Format: `'<checker>:<option_name>[:<option_value>]'`. Multiple invocations possible.
  - `-i, --info INFO`:  Information used to tag the QA results, eg. the simulation id to identify the checked run. Suggested is the original experiment-id you gave the run.
  - `-r, --resume`: Specify to continue a previous QC run. Requires the `<output_dir>` argument to be set.
  - `--rerun-all`: With `--resume`, repeat all checks instead of reusing successful results.
  - `-C, --include_consistency_checks`: Include basic consistency and continuity checks. When using the `wcrp-*`, `cc6`, `mip` or `eerie` checkers, they are included by default.
  - `-w, --whitelist PATH_FRAGMENT`: Only check files whose complete path, including the filename, contains at least one of the specified case-sensitive literal fragments. May be repeated.
  - `-b, --blacklist PATH_FRAGMENT`: Exclude files whose complete path contains any specified case-sensitive literal fragment. May be repeated and takes precedence over the whitelist.

### Example Usage

```shell
$ esgqa -P 8 -t wcrp_cordex_cmip6:latest -t cf:1.11 -o QA_results/IAEVALL02_2025-10-20 -i "IAEVALL02" ESGF_Buff/IAEVALL02/CORDEX-CMIP6
```

To restrict a run to files containing `historical` or `1950`, except for paths
containing `ICON-ESM`:

```shell
$ esgqa -w historical -w 1950 -b ICON-ESM -o QA_results/filtered /path/to/datasets
```

Configured filters are retained when the run is resumed. To use different filters,
start a new run with a different output directory.

To resume at a later date, eg. if the QA run did not finish in time or more files
have been added to the `<parent_dir>`:

```shell
$ esgqa -o QA_results/IAEVALL02_2025-10-20 -r
```

Normal resume does not consider file modification times. Once a file path has
been checked successfully, its checks are only repeated after runtime errors.
Use `--rerun-all` with `--resume` to repeat every check with the stored
configuration regardless of previously successful results:

```shell
$ esgqa -o QA_results/IAEVALL02_2025-10-20 -r --rerun-all
```

On resume, newly selected files and previously selected files that are no longer
found are reported in the terminal and in `resume_inventory_changes.json`.
Results for affected datasets are rerun, and a missing file is checked again if
it later reappears.

For a custom MIP with defined CMOR tables (`"mip"` is not a placeholder but an actual basic checker of the `cc_plugin_cc6`):

```shell
$ esgqa -o /path/to/test/results -t "mip:latest" -O "mip:tables:/path/to/mip_cmor_tables/Tables" /path/to/MIP/datasets/
```

For CF checks and basic time and consistency / continuity checks:
```shell
$ esgqa -o /path/to/test/results -t "cf:1.11" -C /path/to/datasets/to/check
```

## Displaying the check results

The results will be stored in two `json` files:
- `qa_result_*.json`: All failed checks incl. all affected datasets and files are listed. Depending on the number of failed checks and files affected, this file can be quite large in volume (up to GigaBytes).
- `qa_result_*.cluster.json`: The failed checks are clustered and for affected datasets only a single file is referenced as example. This reduces the file size significantly (to usually below 1 MegaByte).

### Web view
The clustered results can be viewed using the following website:

- DKRZ: [https://cmiphub.dkrz.de/info/display_qc_results.html](https://cmiphub.dkrz.de/info/display_qc_results.html).
- IPSL: coming soon

This website runs entirely in the user's browser using JavaScript, without requiring interaction with a web server.
You can select one of the recent QA runs conducted at the respective site or select a local QA run result file to be displayed.

Alternatively, you can open the packaged `display_qc_results.html` file directly
in your browser. Its location in an installed environment can be printed with:

```shell
python -c "from importlib.resources import files; print(files('esgf_qa').joinpath('resources/display_qc_results.html'))"
```

While the web view also supports the full (unclustered) results, it is recommended to not use the web view for files greater than a few MegaBytes.

### `esgqaviewer`
The `esgqaviewer` app can be used to view the result files inside a terminal:
```
esgqaviewer path/to/result.json
```
At the bottom of the viewer, all possible tools are listed. The results can be searched using a full text search for instance.
Mouse controls are enabled by default. Left-click toggles the current node; right-click expands or collapses its subtree. `F2` toggles text-selection mode. After enabling it, drag over the terminal lines and copy them with `Ctrl+Shift+C` (not plain `Ctrl+C`), `Cmd+C` on macOS, or by right-clicking the selected text and using the terminal's context menu. Press `F2` again to disable text selection and restore mouse controls.

### Add results to QA results repository

- DKRZ: [https://cmiphub.dkrz.de/info/display_qc_results.html](https://cmiphub.dkrz.de/info/display_qc_results.html) allows viewing QA results hosted
in the GitLab Repository [qa-results](https://gitlab.dkrz.de/udag/qa-results). You can create a Merge Request in that repository to add your own results.
- IPSL: coming soon
- Feel free to set up repository for QA results for your institute as well. As example implementation can serve: [qa-results](https://gitlab.dkrz.de/udag/qa-results)

# License

This project is licensed under the Apache License 2.0, and includes the Inter font, which is licensed under the SIL Open Font License 1.1. See the [LICENSE](./LICENSE) file for more details.


> [!NOTE]
> **This project was originally developed by [DKRZ](https://www.dkrz.de)** under the name **cc-qa** (see [DKRZ GitLab](https://gitlab.dkrz.de/udag/cc-qa)), with funding from the _German Ministry of Research, Technology and Space_ ([BMFTR](https://www.bmftr.bund.de/en), reference `01LP2326E`).
> It has since been renamed to **esgf-qa** and is now maintained under the **Earth System Grid Federation (ESGF)** organization on GitHub.
>
> If you previously used `cc-qa`, please update your installations as described above.
