# CMI-DAIR Template Python Repository

This is a template repository. Below is a checklist of things you should do to use it:

- [ ] Rewrite this `README` file, updating the badges as needed.
- [ ] Update the pre-commit versions in `.pre-commit-config.yaml`.
- [ ] Install the `pre-commit` hooks.
- [ ] Update the `LICENSE` file to your desired license and set the year.
- [ ] Replace "ENTER_YOUR_EMAIL_ADDRESS" in `CODE_OF_CONDUCT.md`
- [ ] Replace "ENTER_YOUR_EMAIL_ADDRESS" in `SECURITY.md` and update the supported versions or, if not relevant, delete this file.
- [ ] Remove the placeholder src and test files, these are there merely to show how the CI works.
- [ ] Update `pyproject.toml`
- [ ] Update the name of `src/APP_NAME`
- [ ] Grant third-party app permissions (e.g. Codecov) [here](https://github.com/organizations/cmi-dair/settings/installations), if necessary.
- [ ] Either generate a `CODECOV_TOKEN` secret [here](https://github.com/cmi-dair/flowdump/blob/main/.github/workflows/python_tests.yaml) (if its a private repository) or remove the line `token: ${{ secrets.CODECOV_TOKEN }}`
- [ ] API docs website: After the first successful build, go to the `Settings` tab of your repository, scroll down to the `GitHub Pages` section, and select `gh-pages` as the source. This will generate a link to your API docs.

# Project name

[![Build](https://github.com/cmi-dair/template-python-repository/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/cmi-dair/template-python-repository/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/cmi-dair/template-python-repository/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/cmi-dair/template-python-repository)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![L-GPL License](https://img.shields.io/badge/license-L--GPL-blue.svg)](https://github.com/cmi-dair/template-python-repository/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://cmi-dair.github.io/template-python-repository)

What problem does this tool solve?

## Features

- A few
- Cool
- Things

## Installation

Install this package via :

```sh
pip install APP_NAME
```

Or get the newest development version via:

```sh
pip install git+https://github.com/cmi-dair/template-python-repository
```

## Quick start

Short tutorial, maybe with a

```Python
import APP_NAME

APP_NAME.short_example()
```

## Links or References

- [https://www.wikipedia.de](https://www.wikipedia.de)
