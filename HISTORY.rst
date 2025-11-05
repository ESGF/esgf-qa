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
