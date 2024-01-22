# Contributing

## Guidelines

- Every contribution (except for very quick fixes by the maintainers) should go
  through a pull request (PR). Ideally, there is also an open issue that the PR
  solves or that is used to discuss implementation details before the PR is
  opened.
- Code format and style are enforced by the linting tools with the configuration
  defined at the root directory of this package
- Every new contribution must have 100% test coverage. Exception might be made,
  but need to be discussed and justified.

## Step-by-step guide to set up dev workflow

1. Login github and go to https://github.com/OpenSenseAction/poligrain
1. Forking: Click on fork and create a fork with the default settings
1. Cloning: Go to your fork, click on “code”, copy info from ssh or https and
   enter if after $ git clone (e.g. $ git clone
   https://github.com/maxmargraf/poligrain.git)
1. install poetry and nox, e.g. in your base conda env with $ mamba install
   poetry nox if you are not on python >= 3.10 you have to add a new conda env
   with $ mamba create -n poetry_env python=3.10 then activate this env with $
   conda activate poetry_env
1. Run $ poetry install (does config based on pyproject.toml)
1. now you can perform following lints:
   1. $ nox -s lint # Lint only
   1. $ nox -s tests # Python tests
   1. $ nox -s docs -- --serve # Build and serve the docs
   1. $ nox -s build # Make an SDist and wheel
1. Install pre-commit (with $ mamba install pre-commit) and set a hook
   ($
   pre-commit install) (does config based on pyproject.toml) so that you can run
   $
   pre-commit run -a to check if all files pass the style checks prior to a git
   commit

Note: If you need to commit quickly, just have things document, you can use $
git commit --no-verify

## Info on usage of `nox` and `poetry`

See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

### Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `pipx run nox` to run it without installing, or `pipx install nox`.
If you don't have pipx (pip for applications), then you can install with
`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use
`brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s docs -- --serve  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

### Setting up a development environment manually

You can set up a development environment by running:

```bash
poetry install
```

### Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

### Testing

Use pytest to run the unit checks:

```bash
pytest
```

### Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=poligrain
```

### Building docs

You can build the docs using:

```bash
nox -s docs
```

You can see a preview with:

```bash
nox -s docs -- --serve
```

### Pre-commit

This project uses pre-commit for all style checking. While you can run it with
nox, this is such an important tool that it deserves to be installed on its own.
Install pre-commit and run:

```bash
pre-commit run -a
```

to check all files.
