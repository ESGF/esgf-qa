# Customizing consistency checks

Create a separate TOML file, for example `my-consistency.toml`. Selecting
`wcrp_cmip7` uses the built-in CMIP7 profile automatically; `cc6` and
`wcrp_cordex_cmip6` use `cordex_cmip6` (domain/grid-description grouping). Other projects
currently use `default`. A custom file overrides that selection. Update `profile.version`
when changing your rules; keep `schema_version = 1`.
See [schema.md](schema.md) for the complete field and versioning reference.

Start with this CMIP7 example:

```toml
schema_version = 1

[profile]
name = "my-cmip7"
version = "1.0"
extends = "cmip7"  # Or "default" for the general baseline.

[groups.project]
attributes = [
    ["branding_suffix"],
    ["grid_label"],
    # ["source_id"],  # Uncomment to further subdivide comparison groups.
]
on_missing = "separate"

[checks.within_group.dimensions]
severity = "H"

[checks.within_dataset.required_global_attributes]
allow_differences = ["creation_date", "history", "tracking_id", "comment"]

[checks.within_dataset.compatibility_exact]
enabled = false
```

Run it with a new output directory:

```sh
esgqa -t wcrp_cmip7 --consistency-config my-consistency.toml -o QA_custom /path/to/data
```

## Choosing what to compare

| Scope | Comparisons |
| --- | --- |
| `within_dataset` | Files belonging to the same dataset/time series. |
| `within_group` | Reference files from datasets sharing the configured grouping values. |
| `across_all` | Reference files across the complete selected dataset collection (usually: the complete simulation). |

Grouping does not change how files are assigned to individual datasets. Grouping
values come from the first available consistency-output file of each dataset.
All components must match. Each inner list defines fallback attribute names:
`["variant_label", "member_id"]` uses the first available value. A check selects
its named group using `group = "project"`; the CMIP7 baseline already does this
for grouped attributes, dimensions, and coordinates.

Use `severity = "H"`, `"M"`, or `"L"` (also `HIGH`, `MEDIUM`, `LOW`) and
`enabled = false` to disable a check. These settings affect ESGF-QA checks;
Compliance Checker methods are controlled separately by `-I` and `-s`.
For `cc-plugin-wcrp` checkers (like `wcrp_cmip7` and `wcrp_cordex_cmip6`) using
the `esgqa` command line options `-I` and `-s` to filter checks may have unpredictable
consequences. You are advised to instead update the `cc-plugin-wcrp` checker configuration files
("`cc-plugin-wcrp` toml configuration").

Available metadata checks are `required_global_attributes`,
`non_required_global_attributes`, `global_attribute_types`,
`selected_global_attributes`, `variable_attributes`, `variable_attribute_types`,
`dimensions`, and `coordinates`. All three scopes support them. Additionally,
`within_dataset` supports `time_continuity`, `compatibility_override`, and
`compatibility_exact`; the other two scopes support `time_coverage`.

`allow_differences` excludes exact attribute, dimension, or coordinate names from
that check. For variable-attribute checks it names attributes, not variables.
`selected_global_attributes` requires a non-empty `compare_only` list.

## Inheritance and grouping details

- Omitted settings are inherited from `extends`. Lists **replace** the inherited
  list completely; retain any existing exclusions you still want. Without
  `extends`, omitted checks are disabled. For newly added check tables, specify
  `enabled` and `severity` explicitly.
- Scopes are independent. Allowing an attribute to vary within a dataset does
  not exempt it from group or collection-wide comparisons. A new grouping
  attribute may also need an `across_all` exclusion.
- `on_missing = "separate"` isolates datasets lacking a grouping component.
  `"together"` substitutes `Default` for each missing component, still matching
  the remaining components. `"error"` produces a reported QA runtime error.
  The plugin's literal `"unset"` marker is treated as a missing value.
- Only one major project is permitted per run (eg. CMIP6 or CMIP7).
  CF and UGRID can accompany a project checker;
  `cc6` and `wcrp_cordex_cmip6` can run together.

For example, to compare time coverage within groups instead of across all data,
add both overrides:

```toml
[checks.within_group.time_coverage]
enabled = true
severity = "M"
group = "project"

[checks.across_all.time_coverage]
enabled = false
```

See [default.toml](esgf_qa/resources/consistency/default.toml) and
[cmip7.toml](esgf_qa/resources/consistency/cmip7.toml) for the baseline settings.

## Resuming a run

The run saves its merged profile in `consistency_config.toml` and records its hash
in `.resume_info`. Resume uses that archived profile, even if your original TOML
changes or disappears. Do not edit the archived file: its hash is checked.
To apply configuration changes, start a new run in a new output directory;
`--resume --rerun-all` also retains the archived configuration.

# Coordinate and dimension identities

cc-plugin-wcrp summaries can include `coordinate_ids`, `dimension_ids`, and
`string_length_dimensions`. Inter-dataset comparisons match scientific IDs when
both sides provide them, otherwise falling back to names. Each identity gets its
own reference within the group. Missing coordinates or dimensions in another
dataset are not considered inconsistent. Identified character-string storage
widths are excluded from inter-dataset dimension comparisons, irrespective of
their names. Within-dataset comparisons remain strict.

CORDEX's `shared_non_scalar_only` setting excludes scalars only when matching
coordinate IDs are unavailable.
