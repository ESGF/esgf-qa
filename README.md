[![PyPI version](https://img.shields.io/pypi/v/esgf-qa.svg)](https://pypi.org/project/esgf-qa/)

# esgf-qa
### Quality Assurance Workflow Based on `compliance-checker` and `cc-plugin-wcrp` (or other cc-plugins)
<img src="https://raw.githubusercontent.com/ESGF/esgf-qa/master/docs/esgf-qa_Logo.png" align="left" width="120">

`esgf-qa` provides a flexible quality assurance (QA) workflow for evaluating dataset compliance using the
[ioos/compliance-checker](https://github.com/ioos/compliance-checker) framework
(including [CF](https://cfconventions.org/) compliance checks)
and various community plugins (`cc-plugin`s), such as
[ESGF/cc-plugin-wcrp](https://github.com/ESGF/cc-plugin-wcrp) and
[euro-cordex/cc-plugin-cc6](https://github.com/euro-cordex/cc-plugin-cc6).

The tool executes file-based quality control (QC) tests through the Compliance Checker,
and, where applicable, performs additional dataset-level checks to test inter-file time-axis continuity
and consistency in variable, coordinate and attribute definitions.
Results from both file- and dataset-level checks are aggregated, summarized, and clustered for easier interpretation.

### Currently supported checkers

While `esgf-qa` has been primarily developed for workflows assessing compliance with WCRP project data specifications
(e.g., CMIP, CORDEX), it can also be used for general CF-compliance testing and easily extended to support any
`cc-plugin` and projects following CORDEX- or CMIP-style CMOR table conventions.

| Standard                                                                                             | Checker Name |
| ---------------------------------------------------------------------------------------------------- | ------------ |
| [CF Conventions](https://cfconventions.org/) (shipped with [ioos/compliance-checker](https://github.com/ioos/compliance-checker)) | cf |
| [WCRP CMIP6](https://pcmdi.llnl.gov/CMIP6/):<br><ul><li>[CMIP6 DRS](https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf)</li><li>[CMIP6 CVs](https://github.com/WCRP-CMIP/CMIP6_CVs) (esgvoc)</li></li><li>[cmip6-cmor-tables](https://github.com/PCMDI/cmip6-cmor-tables) (esgvoc)</li></ul> | wcrp_cmip6 |
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
esgvoc config set universe:branch=esgvoc_dev
esgvoc config add cordex-cmip6
esgvoc install
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

## Usage

```shell
$ esgqa [-h] [-o <OUTPUT_DIR>] [-t <TEST>] [-O OPTION] [-i <INFO>] [-r] [-C] <parent_dir>
```

- positional arguments:
  - `parent_dir`: Parent directory to scan for netCDF-files to check
- options:
  - `-h, --help`: show this help message and exit
  - `-o, --output_dir OUTPUT_DIR`: Directory to store QA results. Needs to be non-existing or empty or from previous QA run. If not specified, will store results in `./cc-qa-check-results/YYYYMMDD-HHmm_<hash>`.
  - `-t, --test TEST`: The test to run (`'wcrp_cmip6:latest'`, `'wcrp_cordex_cmip6:latest'` or `'cf:<version>'`, can be specified multiple times, eg.: `'-t wcrp_cmip6:latest -t cf:1.7'`) - default: running latest CF checks `'cf:latest'`.
  - `-O, --option OPTION`: Additional options to be passed to the checkers. Format: `'<checker>:<option_name>[:<option_value>]'`. Multiple invocations possible.
  - `-i, --info INFO`:  Information used to tag the QA results, eg. the simulation id to identify the checked run. Suggested is the original experiment-id you gave the run.
  - `-r, --resume`: Specify to continue a previous QC run. Requires the `<output_dir>` argument to be set.
  - `-C, --include_consistency_checks`: Include basic consistency and continuity checks. When using the `wcrp-*`, `cc6`, `mip` or `eerie` checkers, they are included by default.

### Example Usage

```shell
$ esgqa -t wcrp_cordex_cmip6:latest -t cf:1.11 -o QA_results/IAEVALL02_2025-10-20 -i "IAEVALL02" ESGF_Buff/IAEVALL02/CORDEX-CMIP6
```

To resume at a later date, eg. if the QA run did not finish in time or more files have been added to the `<parent_dir>`
(note, that the last modification date of files is NOT taken into account - once a certain file path has been checked
it will be marked as checked and checks will only be repeated if runtime errors occured):

```shell
$ esgqa -o QA_results/IAEVALL02_2025-10-20 -r
```

For a custom MIP with defined CMOR tables (`"mip"` is not a placeholder but an actual basic checker of the `cc_plugin_cc6`):

```shell
$ esgqa -o /path/to/test/results -t "mip:latest" -O "mip:tables:/path/to/mip_cmor_tables/Tables" /path/to/MIP/datasets/`
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

Alternatively, you can open the included `display_qc_results.html` file directly in your browser.
While the web view also supports the full (unclustered) results, it is recommended to not use the web view for files greater than a few MegaBytes.

### `esgqaviewer`
The `esgqaviewer` app can be used to view the result files inside a terminal:
```
esgqaviewer path/to/result.json
```
At the bottom of the viewer, all possible tools are listed. The results can be searched using a full text search for instance.
A double click with the right mouse button on a node will expand / collapse it and below nodes fully, while a left click will collapse the current node only.

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
