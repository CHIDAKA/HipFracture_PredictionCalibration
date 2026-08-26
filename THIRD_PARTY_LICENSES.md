# Third-Party Package Licenses

This file records third-party packages referenced by repository code and their
license information.

## Package inventory

| Package | Version | License | Source URL | Notes |
| --- | --- | --- | --- | --- |
| numpy | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/numpy/ | Numerical arrays and vectorized computation |
| pandas | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/pandas/ | DataFrame operations and data IO |
| scikit-learn | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/scikit-learn/ | ML utilities and isotonic regression |
| scikit-survival | Not pinned in repository | GPL-3.0-only | https://pypi.org/project/scikit-survival/ | Survival analysis models and metrics |
| scipy | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/scipy/ | Statistical helper functions |
| matplotlib | Not pinned in repository | Matplotlib License (BSD-style) | https://pypi.org/project/matplotlib/ | Plotting and visualization |
| seaborn | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/seaborn/ | Statistical visualization |
| optuna | Not pinned in repository | MIT | https://pypi.org/project/optuna/ | Hyperparameter optimization |
| duckdb | Not pinned in repository | MIT | https://pypi.org/project/duckdb/ | Embedded analytical SQL engine |
| python-dateutil | Not pinned in repository | BSD-3-Clause | https://pypi.org/project/python-dateutil/ | Date arithmetic and relative deltas |

## Notes

- Versions are listed as "Not pinned in repository" because no lockfile or
  dependency manifest currently defines exact package versions.
- Python standard library modules (for example `os`, `sys`, `argparse`,
  `pickle`, `datetime`, `re`, and `collections`) are intentionally excluded.
