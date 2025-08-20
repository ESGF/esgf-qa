[![PyPI version](https://img.shields.io/pypi/v/cc-qa.svg)](https://pypi.org/project/cc-qa/)

# cc-qa: Quality Assurance Workflow Based on compliance-checker and cc-plugin-cc6

This makes use of the frameworks and [CF](https://cfconventions.org/)-compliance checks of the
[ioos/compliance-checker](https://github.com/ioos/compliance-checker) and extensions coming with
[euro-cordex/cc-plugin-cc6](https://github.com/euro-cordex/cc-plugin-cc6).

This tool is designed to run the desired file-based QC tests with
[ioos/compliance-checker](https://github.com/ioos/compliance-checker) and
[euro-cordex/cc-plugin-cc6](https://github.com/euro-cordex/cc-plugin-cc6),
conduct additional dataset-based checks (such as time axis continuity and
consistency checks) as well as summarizing the test results.

`cc-qa` is mainly aimed at a QA workflow testing compliance with CORDEX-CMIP6 Archive Specifications (see below).
However, it is generally applicable to test for compliance with the CF conventions through application of the IOOS Compliance Checker, and it is easily extendable for any `cc-plugin` and for projects defining CORDEX, CORDEX-CMIP6, CMIP5 or CMIP6 style CMOR-tables.

| Standard                                                                                             | Checker Name |
| ---------------------------------------------------------------------------------------------------- | ------------ |
| [cordex-cmip6-cv](https://github.com/WCRP-CORDEX/cordex-cmip6-cv)         |  cc6         |
| [cordex-cmip6-cmor-tables](https://github.com/WCRP-CORDEX/cordex-cmip6-cmor-tables)|  cc6         |
| [CORDEX-CMIP6 Archive Specifications](https://doi.org/10.5281/zenodo.10961069) | cc6 |
| [CMIP6 DRS](https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf) | wcrp_cmip6 (esgf-qc) |
| [cmip6-cmor-tables](https://github.com/PCMDI/cmip6-cmor-tables) | wcrp_cmip6 (esgf_qc) |
| [CMIP6 CVs](https://github.com/WCRP-CMIP/CMIP6_CVs) | wcrp_cmip6 (esgf_qc) |
| [EERIE CMOR Tables & CV](https://github.com/eerie-project/dreq_tools) | eerie |

## Installation

### Pip installation

```shell
$ pip install cc_qa
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

## Usage

```shell
$ ccqa [-h] [-o <RESULT_DIR>][-t <TEST>] [-i <INFO>] [-r] <parent_dir>
```

- positional arguments:
  - `parent_dir`: Parent directory to scan for netCDF-files to check
- options:
  - `-h, --help`: show this help message and exit
  - `-o, --output_dir OUTPUT_DIR`: Directory to store QA results. Needs to be non-existing or empty. If not specified, will store results in `./cc-qa-check-results/YYYYMMDD-HHmm_<hash>`.
  - `-t, --test TEST`: The test to run ('cc6' or 'cf', can be specified multiple times) - default: running 'cc6' and 'cf'.
  - `-i, --info INFO`:  Information used to tag the QA results, eg. the simulation id to identify the checked run. Suggested is the original experiment-id you gave the run.
  - `-r, --resume`: Specify to continue a previous QC run. Requires the <output_dir> argument to be set.

### Example Usage

```shell
$ ccqa -o /work/bb1364/dkrz/QC_results/IAEVALL02_2025-04-20 -i "IAEVALL02" /work/bb1149/ESGF_Buff/IAEVALL02/CORDEX-CMIP6
```

To resume at a later date, eg. if the QA run did not finish in time or more files have been added to the <parent_dir>
(note, that the last modification date of files is NOT taken into account - once a certain file path has been checked
it will be marked as checked and checks will only be repeated if runtime errors occured):

```shell
$ ccqa -o /work/bb1364/dkrz/QC_results/IAEVALL02_2025-04-20 -r
```

## Displaying the check results

The results will be stored in a single `json` file, which can be viewed using the following website:
[https://cmiphub.dkrz.de/info/display_qc_results.html](https://cmiphub.dkrz.de/info/display_qc_results.html).
This website runs entirely in the user's browser using JavaScript, without requiring interaction with a web server.
Alternatively, you can open the included `display_qc_results.html` file directly in your browser.

### Add results to QA results repository

[https://cmiphub.dkrz.de/info/display_qc_results.html](https://cmiphub.dkrz.de/info/display_qc_results.html) allows viewing QA results hosted
in the GitLab Repository [qa-results](https://gitlab.dkrz.de/udag/qa-results). You can create a Merge Request in that repository to add your own results.


### Install `ESGF-QC` as `compliance_checker` plugin rather than a `compliance_checker` replacement

Since `ESGF-QC` is set up as fork of the IOOS `compliance_checker`, rather than a plugin,
installing `ESGF-QC` and `compliance_checker` at the same time is not possible:
Installing `ESGF-QC` will replace / overwrite any installation of the IOOS `compliance_checker`.

The alternatives are to set up different virtual environments for each tool or to mimic an
`esgf-qc` plugin for `compliance_checker`, enabling to use the `wcrp_cmip6` and other checkers
set up in `esgf-qc` within the `compliance_checker`. As long as there are no incompatibilities 
introduced in `esgf_qc` (which is currently the case) the latter one is a viable option,
 and is explained in the following:

- Clone the `esgf-qc repository`

```
git clone https://github.com/ESGF/esgf-qc.git
```

- Create a `esgf-qc-plugin` directory and softlink the necessary folders

```
mkdir -p esgf-qc-plugin/esgf-qc
cd esgf-qc-plugin
ln -s ../esgf-qc/checks ./esgf-qc/
ln -s ../esgf-qc/wcrp ./esgf-qc/
```

- Link the included `pyproject_esgf-qc-plugin.toml` from this repository

```
ln -s ../cc-qa/pyproject_esgf-qc-plugin.toml ./pyproject.toml
```

- pip install as plugin
```
pip install -e . --no-deps
```

- Set up `esgvoc` as described in the `esgf-qc` readme

```
pip install esgvoc
esgvoc config set universe:branch=esgvoc_dev
esgvoc install
```

- Test your installation

The following command should now also list the `esgf-qc` checks next to all `cc_plugin_cc6` and `compliance_checker` checks:
```
cchecker.py -l
```

# License

This project is licensed under the Apache License 2.0, and includes the Inter font, which is licensed under the SIL Open Font License 1.1. See the [LICENSE](./LICENSE) file for more details.
