# Consistency-configuration schema

This document is the reference for the TOML file accepted by
`esgqa --consistency-config FILE`. For a task-oriented introduction and examples,
see [instruction.md](instruction.md).

## Versioning

Three independent identifiers are involved:

| Identifier | Owner | Purpose |
| --- | --- | --- |
| `schema_version` | ESGF-QA | Identifies the structure and meaning of this TOML format. |
| `profile.version` | Profile author | Identifies a particular revision of a project's rules. |
| ESGF-QA package version | ESGF-QA release | Identifies the code that interpreted and executed the rules. |

The only supported schema version is currently the integer `1`. Keep
`schema_version = 1` when adjusting groups, severities, exclusions, or enabled
checks. It is not a user-editable revision counter. ESGF-QA rejects a missing,
non-integer, or unsupported schema version.

Increment `profile.version` whenever the effective project rules change. It is a
required, non-empty string, so a project may use semantic versions, dates, or its
own convention. Changing it does not select a different parser, but it is retained
as provenance and changes the resolved-configuration hash.

A future `schema_version` must be introduced when the configuration language
changes incompatibly, for example when a field is renamed, its type or meaning
changes, or a new scope/check identifier is added that older strict readers cannot
recognize. Supporting such a version requires an explicit implementation change
in ESGF-QA; users should never increment it pre-emptively.

## Top-level structure

```toml
schema_version = 1

[profile]
name = "my-project"
version = "1.0"
extends = "default"

[groups.project]
attributes = [["realm", "modeling_realm"], ["grid_label", "grid"]]
on_missing = "together"

[checks.within_group.dimensions]
enabled = true
severity = "M"
group = "project"
allow_differences = ["time"]
```

The root accepts only `schema_version`, `profile`, `groups`, and `checks`.
Unknown keys are rejected rather than silently ignored.

## `[profile]`

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | non-empty string | yes | Human-readable profile identifier stored with results. |
| `version` | non-empty string | yes | Author-maintained revision of the effective rules. |
| `extends` | `"default"` or `"cmip7"` | no | Packaged profile to merge before applying this file. |

`extends` is accepted in an input profile but removed from the resolved archive.
Tables are merged recursively. Scalars and lists in the child replace inherited
values; lists are not appended. Consequently, an overridden `allow_differences`
list must repeat inherited entries that should remain allowed.

Without `extends`, the profile starts without inherited groups or enabled checks.
An absent check is disabled. An explicitly present check table defaults to enabled
with medium severity.

## `[groups.<name>]`

A named group defines how datasets are partitioned for `within_group` checks.
Group names must be non-empty and may be referenced by multiple checks.

| Field | Type | Required | Default |
| --- | --- | --- | --- |
| `attributes` | non-empty array of non-empty string arrays | yes | none |
| `on_missing` | `"together"`, `"separate"`, or `"error"` | no | `"together"` |

Each inner `attributes` array is one grouping component and lists fallback names
in priority order. For example:

```toml
attributes = [["variant_label", "member_id"], ["grid_label", "grid"]]
```

ESGF-QA first looks for `variant_label` and falls back to `member_id`; it then
does the same for `grid_label` and `grid`. All resulting components form the
group key. Values are read from required and non-required global attributes in
the first available consistency output for each dataset. `null`, an empty string,
and the plugin marker `"unset"` count as missing.

`on_missing` controls a missing component:

- `together` substitutes `Default`, allowing datasets with equivalent remaining
  components to share a group.
- `separate` gives each affected dataset a distinct group.
- `error` reports a QA runtime error naming the dataset and missing alternatives.

## `[checks.<scope>.<name>]`

The supported scopes are:

| Scope | Comparison population |
| --- | --- |
| `within_dataset` | Files in one dataset/time series. |
| `within_group` | Dataset reference files sharing a configured group key. |
| `across_all` | Dataset reference files across the complete run. |

Every check table accepts the following fields, subject to the restrictions below:

| Field | Type | Default | Restrictions |
| --- | --- | --- | --- |
| `enabled` | Boolean | `true` | Applies when the check table is present. |
| `severity` | string | `"M"` | `H`, `M`, `L`, or `HIGH`, `MEDIUM`, `LOW`, case-insensitively. |
| `group` | non-empty string | none | Only for `within_group`; required when that check is enabled and must reference an existing group. |
| `allow_differences` | array of unique, non-empty strings | `[]` | Only for metadata comparison checks. |
| `compare_only` | array of unique, non-empty strings | `[]` | For `selected_global_attributes`; required and non-empty when enabled. |
| `compare_only_when_present` | Boolean | `false` | Cross-dataset `selected_global_attributes` only; skips missing attributes instead of reporting their absence. |

`allow_differences` contains exact names omitted from comparison. For global
attribute, dimension, and coordinate checks these are the corresponding names;
for variable-attribute checks they are attribute names, not variable names.

### Supported checks

These metadata comparison checks are available in all three scopes:

- `required_global_attributes`
- `non_required_global_attributes`
- `global_attribute_types`
- `selected_global_attributes`
- `variable_attributes`
- `variable_attribute_types`
- `dimensions`
- `coordinates`

Only `within_dataset` additionally supports:

- `time_continuity`
- `compatibility_override`
- `compatibility_exact`

Only `within_group` and `across_all` additionally support:

- `time_coverage`

### Named selected-attribute comparisons

Cross-dataset `selected_global_attributes` supports independently configured
instances alongside the existing unnamed check. Each instance has its own
group, severity, enabled flag and selected attributes; settings are not inherited
from the unnamed check. Profile inheritance can override an individual instance.

```toml
[checks.within_group.selected_global_attributes.instances.table]
enabled = true
severity = "L"
group = "table"
compare_only = ["table_info"]
compare_only_when_present = true
```

The group must be declared separately. Bundled profiles define `table` using
`table_id`, keeping datasets with missing IDs separate. With
`compare_only_when_present = true`, each selected attribute gets its own
reference, so an absent value in the first dataset does not prevent later
comparisons. This policy concerns comparison behavior, not whether the project
requires the attribute.

Unknown scopes, check names, and fields are rejected. `allow_differences` and
`compare_only` cannot be assigned to continuity, compatibility, or coverage
checks.

## Resolution, archival, and resume

Before checks run, ESGF-QA resolves inheritance and writes a canonical
`consistency_config.toml` into the result directory. The resolved file:

- removes `profile.extends`;
- explicitly records every supported check, including disabled ones;
- explicitly records default severities, lists, and missing-value policies;
- normalizes severity aliases to `H`, `M`, or `L`;
- orders groups, scopes, and checks deterministically.

ESGF-QA calculates a SHA-256 hash over this resolved TOML. The archive filename,
profile name, profile version, and hash are stored in `.resume_info` and result
metadata. The hash identifies the exact resolved representation; changing even a
provenance value or list order may change it although check behavior can remain
equivalent.

Resume reads the archived file and verifies its hash. It does not re-read the
original custom file or a newer packaged profile. Editing the archived file makes
the run fail validation. To apply changed rules, start a new run and output
directory; `--resume --rerun-all` deliberately continues to use the archived
configuration.
