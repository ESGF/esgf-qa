0.6.0 (2026-08-16)
------------------

Breaking Changes
^^^^^^^^^^^^^^^^

* The ``info.checkers`` field in result JSON is now a list of readable checker descriptions instead of one comma-separated string. Each description also identifies the Python distribution that supplied the checker and its installed version.
* ``display_qc_results.html`` and its fonts are now packaged under ``esgf_qa/resources`` instead of the repository root. Installed applications should locate the viewer with ``importlib.resources`` as described in the README.
* No other intentional breaking changes.

New Features
^^^^^^^^^^^^

* Added repeatable path filters:

  * ``-w/--whitelist PATH_FRAGMENT`` selects files whose complete path, including the filename, contains at least one supplied literal fragment.
  * ``-b/--blacklist PATH_FRAGMENT`` excludes files whose complete path contains a supplied literal fragment and takes precedence over the whitelist.
  * Filter settings are retained on resume and ``excluded_files.json`` records the selected filters, counts, excluded paths, and blacklist matches.

* Added ``--rerun-all`` for use with ``--resume``. It repeats every file- and dataset-level check with the stored configuration, regardless of previously successful results. Outputs from an earlier attempt are removed immediately before each rerun and completed JSON files are replaced atomically, preventing stale results from being reused after a failed rerun.
* Resume now compares the previous and current selected-file inventories. New and no-longer-found files are reported in the terminal and in ``resume_inventory_changes.json`` - affected dataset-level results are invalidated automatically.
* Added ``wcrp_cmip6plus`` as a supported checker with automatic consistency-output configuration and consistency/continuity checks.
* Added an ``F2`` toggle to ``esgqaviewer`` for switching between TUI mouse controls and terminal text selection. The viewer now displays guidance for left-click, right-click, text selection, and terminal-specific copy commands.
* Result provenance now records the installed distribution and version that provides each checker.

Bug Fixes
^^^^^^^^^

* When one checker function reports the same named check at multiple severity levels, its result is stored as a list of result records instead of silently retaining only one record. Consumers of unclustered result JSON must therefore accept either one result object or a list of result objects for a check.
* Fixed the Compliance Checker options mapping, which incorrectly used ``cf:`` instead of ``cf``. This caused CF options, including the enabled Appendix A checks, to be ignored.
* Checker selection is validated more strictly. Explicitly selecting ``mip`` now always requires ``-O mip:tables:/path/to/Tables`` and ``mip`` cannot be combined with its ``eerie`` alias. Malformed ``-O`` values with a missing checker or option name are rejected with explanatory errors.
* Preserved every severity returned by one Compliance Checker plugin function instead of overwriting results that share the same check name. The same correction applies to consistency-check aggregation.
* Missing expected consistency-output files are now recorded as QA runtime errors. Dataset-level consistency checks skip unavailable outputs, select an available replacement reference file where possible, and report only such reference-file substitutions.
* Exceptions raised by dataset- and collection-level consistency checks are captured in the QA report instead of terminating the complete run. Reports include the failing function, source location, affected datasets, and files.
* Compliance Checker initialization, plugin discovery, dataset loading, individual checker execution, missing checker results, and dataset-closing failures are isolated and recorded as runtime errors. One failing checker no longer prevents other selected checkers from running, and opened datasets are closed even after failures.
* Resume restores the original ``info`` value when none is supplied and validates the correct ``checker_options`` field. Cached file, consistency-output, and dataset JSON is now reused only when it is readable, structurally complete, contains every selected checker, and has no runtime errors.
* Adding a file, removing a file, or encountering an incomplete/error-producing file result now invalidates dependent dataset-level results. A reappearing file is checked again.
* Fixed time-continuity handling for valid zero-valued coordinates, reporting of unsupported frequencies, inverted decade/century tolerances, non-January coverage calculations, and malformed filename timestamps.
* Incomplete filesystem traversal is now reported as a scan error and cannot be mistaken for files disappearing during resume reconciliation.
* Result ordering, clustering input, dataset/file ordering, and clustered example-file selection are deterministic across differing worker-completion and hash iteration orders.
* ``esgqaviewer`` now reports unreadable or invalid JSON input as a clear command-line error and renders list values without numeric indices for scalar entries.

Other Improvements
^^^^^^^^^^^^^^^^^^

* Refactored the former monolithic runner into focused modules for CLI handling, discovery, resume/cache handling, checker metadata, workers, and workflow orchestration.
* Reduced multiprocessing overhead by passing compact per-file or per-dataset tasks and recording progress only in the parent process. Initial checks and multi-file dataset checks run in disposable workers to limit retained netCDF/xarray state, while plugin discovery is cached per worker without sharing ``CheckSuite`` instances between files.
* Expanded regression coverage for CLI validation, resume and cache corruption, checker failures, consistency errors, deterministic output, TUI interaction, and packaged artifacts.
* Reworked CI into code-quality, supported-Python test, package-validation, and optional upstream-head workflows. Migrated Flake8 integration to Ruff and added an enforced coverage threshold.
* Wheels now include ``display_qc_results.html`` and its fonts while excluding the top-level test package. Source distributions retain the tests and provide a ``test`` dependency extra. Release builds validate and smoke-test both distribution formats before publication.
* Updated the checker/project overview and ``esgvoc`` installation and update instructions.

0.5.1 (2026-05-28)
------------------

Bug Fixes
^^^^^^^^^

* Fix for reference dataset blocks accumulating when viewing multiple QC results one after another with ``display_qc_results.html`` (commit caffe55e85408464ec4bf12ac57922d138119f39).
* Compatibility fix for newer ``compliance-checker`` versions (>= 6.1.0): Handling ``:latest`` as supplied checker version (commit 62263ea7dc3c8cb59e282886b4dcb665f95f2771).
* Fixed checker ``wcrp_cmip7`` lacking configuration (checker options) for inter-file and inter-dataset consistency checks (commit 83358b38ad734e19a9dee851980441dbeea65a7d).

0.5.0 (2026-02-06)
------------------

New Features
^^^^^^^^^^^^

* Allowing to limit the number of parallel processes with the ``-P <max_processes>`` command line parameter.
* Generally, any ``cc-plugin`` is now supported.
* Creation of dataset-ids now supports a variety of projects. The list can be updated if needed.
* Information on found files and organization into datasets now stored to disk rather than being output to stdout.

0.4.0 (2025-11-05)
------------------

New Features
^^^^^^^^^^^^

* Allowing checker options to be specified via command line for all checkers.
* Improved support of ``cc-plugin-wcrp``: enabled inter-file/dataset consistency & continuity checks.

Bug Fixes
^^^^^^^^^

* Time continuity check: No longer throwing exception on unsupported time coordinates.

Breaking Changes
^^^^^^^^^^^^^^^^

* No longer allowing respecification of checkers and options when resuming QA run (commit 3d2e082d40aef7c512ce828b1e4600ef81176e37).

0.3.0 (2025-10-17)
------------------

This is the first release of this package under the name `esgf-qa` and versioned/maintained under the ESGF organization
(https://github.com/ESGF/esgf-qa) on GitHub. This project was originally labeled `cc-qa` and versioned via the DKRZ GitLab (https://gitlab.dkrz.de/udag/cc-qa).

New Features
^^^^^^^^^^^^

* Changed app executable from ccqa to esgqa
* Added esgqaviewer app
* Added reference datasets for inter-dataset consistency checks
* Added reference dataset in web result viewer (display_qc_results.html)
* Updated creation of dataset ids from file paths
* Basic support of cc-plugin-wcrp

0.2.0 (2025-08-20)
------------------

New Features
^^^^^^^^^^^^

* Now supporting ESGF-QC and EERIE checkers.
* Added `-C` command line argument to additionally run consistency and time checks when not running the 'mip' or 'cc6' checkers.

Bug Fixes
^^^^^^^^^
* Fixed check for consistent time-span of datasets failing when filename timestamp is not a time range.

0.1.2 (2025-06-16)
------------------

New Features
^^^^^^^^^^^^
* Now printing the respective references for consistency checks (commits d7ebfbd17e1926aa7e3e61acd55b5319cd9ce184 & 4ec6ed82fbecf44aca1680f27b48a1351ec481fd).

Bug Fixes
^^^^^^^^^
* Fixed inter-dataset checks not being reset for each dataset (commits d7ebfbd17e1926aa7e3e61acd55b5319cd9ce184 & 4ec6ed82fbecf44aca1680f27b48a1351ec481fd).
* CLI overhaul (commit 7362826ca8c60efc0a4e0f4a81723ec1f49c006e).

0.1.1 (2025-06-13)
------------------

Bug Fixes
^^^^^^^^^
* Fixed cluster example message ending up scrambled at times (commit babb141203a00325a077da158cfd4e16e13b2af1).

0.1.0 (2025-06-12)
-------------------

* First release.
