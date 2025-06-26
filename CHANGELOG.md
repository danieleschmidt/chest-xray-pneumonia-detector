# Changelog

## v0.1.0
- Add API_USAGE_GUIDE with CLI invocation examples
- Provide `version_cli` for printing the installed package version
- Expose `cxr-version` console script for convenience
- Introduce `dataset_stats` module and `cxr-dataset-stats` CLI
- `cxr-dataset-stats` now accepts a list of file extensions via `--extensions`
- `cxr-dataset-stats` can write counts to CSV with `--csv_output`
- Extensions are normalized to lower case and may omit the leading dot
- JSON and CSV outputs are sorted alphabetically by class name
- Better error messages if the input path does not exist or is not a directory
