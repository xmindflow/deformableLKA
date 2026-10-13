# D-LKA Net 2D

The upstream D-LKA Net repository is kept intact under [`official/`](official/):

- 2D implementation: [`official/2D/`](official/2D/)
- 3D implementation: [`official/3D/`](official/3D/)
- Original upstream README: [`official/README.md`](official/README.md)
- Paper: [`Beyond_Self-Attention_D-LKA_WACV2024.pdf`](Beyond_Self-Attention_D-LKA_WACV2024.pdf)

Local CPU environment, data/checkpoint links, and setup notes are in
[`docs/DLKA_2D_DATA_AND_ENV.md`](docs/DLKA_2D_DATA_AND_ENV.md). Test cases and
the latest test results are in [`docs/DLKA_2D_TEST_REPORT.md`](docs/DLKA_2D_TEST_REPORT.md).

Run the smoke tests with:

```bash
/Volumes/t7/program/python/environments/dlka2d/bin/python -m pytest -q tests
```
