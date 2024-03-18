# How to contribute

## Guidelines for contributions

- Every contribution (except for very quick fixes by the maintainers) should go
  through a pull request (PR). Ideally, there is also an open issue that the PR
  solves or that is used to discuss implementation details before the PR is
  opened.
- Code format and style are enforced by the linting tools with the configuration
  defined at the root directory of this package.
- Some linting rules might create false-positives and some might turn out to be
  to strict for our use. In most cases they help the code quality of our
  project, though. If you feel you need to disable a linting rule, please
  discuss that with the maintainers.
- Every new contribution must have 100% test coverage. Exception might be made,
  but need to be discussed and justified.

## Step-by-step guide to set up the required dev workflow

### Fork repo and clone to your local machine

1. Login to github and go to
   [https://github.com/OpenSenseAction/poligrain](https://github.com/OpenSenseAction/poligrain).
1. Forking: Click on "fork" and create a fork with the default settings.
1. Cloning: Go to your fork, click on “code”, copy info from ssh or https and
   enter if after `git clone` (e.g.
   `git clone https://github.com/maxmargraf/poligrain.git`) on the command line.
   This will download the current version of the code into a directory called
   `poligrain`.

### Set up Python environments

**Preface**: There are different ways to set up all dependencies and tools. We
recommend to start from a `conda` environment where you, as a user, can install
new packages without admin rights. In the `conda` environment we only need some
"tools" to handle our dev workflow, but these tools are not part of our actual
package `poligrain`. On top of the `conda` environment we need a `poetry`
environment in which all package dependencies are managed and synchronized with
the `pyproject.toml` file which is later used to build the package that can be
uploaded to pypi.org.

These are our recommended steps:

1. Make sure that your `conda` environment is on Python >=3.10. If this is the
   case you can do the next step in you base environment. If not you, have to
   add a new environment with `mamba create -n poetry_env python=3.10` and then
   activate this env with `conda activate poetry_env`.
1. In your `conda` environment, install `poetry`, `nox` and `pre-commit` via
   `mamba install poetry nox pre-commit`.
1. Run `poetry install` (this does all configurations based on pyproject.toml)
1. To make sure that everything is working correctly you can now perform the
   following linting, testing and building of doc:
   - `nox -s lint # Lint only`
   - `nox -s tests # Python tests`
   - `nox -s docs -- --serve` # Build and serve the docs so that you can check
     them locally in your browser
   - `nox -s build` # Make an SDist and wheel, which is the same package that
     would be uploaded to PyPi

### Write code and commit the changes

1. Activate the pre-commit hook with `pre-commit install` (this is based on
   config in pyproject.toml and .pre-commit-config.yaml) so that you can run
   `pre-commit run -a`` to check if all files pass the style checks prior to a
   git commit. The pre-commit hook will be run for all staged files before I
   commit is accepted. Many checks do automatic updates, e.g. of formatting. You
   just have to add these changes then to your commit and try to commit again.
1.

start Jupyterlab in `poetry` environment...

Note: If you need to commit quickly, just have things document, you can use
`git commit --no-verify`. But this will results in errors during the linting of
the CI run.

### Contributing via a pull request

sync fork,

commit push PR

## More detailed info on dev workflow and the used tools

See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

### Quick development

The fastest way to start with development is to use nox

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
