# Experiment Notes

This directory intentionally contains only generic experiment helpers.

Cluster-specific launch scripts and environment files were removed during cleanup because they relied on:

- hard-coded absolute paths
- site-specific scheduler settings
- site-specific environment variables
- personal or team-specific W&B defaults

Use the commands documented in the repository root `README.md` as the starting point for new experiments.

The remaining helper:

- `clean.sh`: removes `*.out` and `*.err` files under the current directory
