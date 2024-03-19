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
"tools" to handle our dev workflow (in our case these are `poetry`, `pre-commit`
and `nox`), but these tools are not part of our actual package `poligrain`. On
top of the `conda` environment we need a `poetry` environment in which all
package dependencies are managed and synchronized with the `pyproject.toml`
file, which is later used to build the package that can be uploaded to pypi.org.

We recommend to use `miniforge` as basis for working with `conda` because it
includes the fast dependency solver `mamba` and is based on packages from the
`conda-forge` channel, where, in contrast to the "default channel" of Anaconda
Inc, everybody can upload and provide packages.

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
   - `nox -s tests # Python tests` # You have to run test
   - `nox -s docs -- --serve` # Build and serve the docs so that you can check
     them locally in your browser (this is handy to debug problems with the
     docs)
   - `nox -s build` # Make an SDist and wheel, which is the same package that
     would be uploaded to PyPi (this is not required for you to do, but it
     should work)

### Write code and commit the changes

**Preface**: To be able to write code that conforms to our linting rules, you
need it to pass the checks that are run by `pre-commit`, which you need
installed and activated. We provide code examples as notebooks that are then
added to the docs. Hence, you need Jupyterlab to work with notebooks. The
notebooks have to run in the `poetry` environment where `poligrain` is available
as development install, so that you always import the current state of the
codebase.

These are the steps to get a suitable setup:

1. Activate the pre-commit hook with `pre-commit install` (this is based on
   config in pyproject.toml and .pre-commit-config.yaml) so that you can run
   `pre-commit run -a`` to check if all files pass the style checks prior to a git commit. The `pre-commit`
   hook will be run for all files that are staged for commit before a commit is
   accepted. Many checks do automatic updates, e.g. of formatting. You just have
   to stage these automatically applied changes to your commit and try to commit
   again. Some checks will require manual adjustment, though.
1. To run Jupyterlab inside the `poetry` environment run `poetry shell`, which
   opens a shell inside the `poetry` env, which might be indicated in your
   terminal prompt by e.g. `(poligrain-py3.10)`. Then run `jupyter-lab` and go
   to the browser where it is viewed. If you open a new notebook there, or if
   you run one of the existing ones, you should be able to do
   `import poligrain`.
1. You can now change the code under `src/poligrain` which is then directly
   available in the notebook via `import poligrain`. Since imports in Python are
   cached, you should add the following at the top of your notebook
   ```
   %load_ext autoreload
   %autoreload 2
   ```
   Note that you have to restart your kernel and run these two lines before
   doing the first import of `poligrain` to get the current non-cached version
   of the code.
1. You should regularly commit your changes. Please look at a git tutorial if
   you are not yet familiar with this concept. Each commit has to pass all
   `pre-commit` checks, which are run automatically, see info about `pre-commit`
   above. Note that if you need to commit quickly, e.g. to just have things
   document, you can use `git commit --no-verify`. But this will results in
   errors during the linting of the CI run.
1. It is recommended that you create a new branch when working on a new feature
   or a fix, because it might take longer than expected and you might want to go
   back to your working `main` branch.
1. You should also regularly push your commits to your branch so that others are
   aware of your changes. If you do `git push` while you are on your newly
   generated feature branch, git will give you an error message that is does not
   know which remote branch to push to. But git will also give you the
   recommended command which you just copy-paste and run. Git will most likely
   now already recommend that you generate a pull-request (PR), see details
   below, which is generally recommended unless you do not intent to ever merge
   your changes into the parent repository. If your code is in an early stage,
   you can just as "[WIP]" to your PR title.

### Contributing via a pull request

**Preface**: To bring your changes to the parent repository of `poligrain` at
https://github.com/OpenSenseAction/poligrain for which only the maintainers have
write permissions, you need to create a pull-request (PR) so that the
maintainers can pull in your changes. But, after some time, the parent
repository might have moved forward by adding changes that you do not have in
your fork and in your local clone of the repo. Hence, you will need to
synchronize your local code with the latest changes in the parent repository.
This is not always required, but the time will come when it is, e.g. because you
need a feature or a bug fix that was added by somebody else in the meantime.

The following steps 1 and 2 are only required if you need to sync your repo to
the parent repo.

1. To sync to the newest change in the parent epo, go to your fork on github and
   click the `sync fork` button, see the github docs
   [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).
1. Then you have get your updated `main` branch from your fork to your local
   machine. This is done via `git fetch origin` (assuming `origin` is referring
   to your fork, which is the default if you followed this step-by-step guide)
   or just `git fetch`. Then switch to your `main` branch and either do a merge
   or a rebase on `origin/main`. You can also stay on your feature branch and do
   the same merge or rebase. If you are not 100% sure that a rebase is a good
   idea, please use `git merge origin/main`. If you are sure that a rebase is a
   good idea, do `git rebase origin/main`.
1. Create a pull-request on github, which is typically shown as a recommendation
   when going to the `poligrain` repo or it is directly recommended when doing a
   `git push` of your feature branch for the first time.
1. In the PR template provide some relevant information about your PR. If you
   are unsure what to write, look at old PRs to get an idea which content fits.
1. The CI will run the `pre-commmit` checks and the unit tests, see info below.
   If your `pre-commit` checks were successful locally on your machine, they
   should also pass here. If not, you have to inspect the logs of the CI.
1. We require 100% test coverage in our PRs. Hence, if you write new code, you
   also have to write tests. Test coverage will automatically be shown by
   codecov in the PR. Please look in the section below to learn about writing
   tests.
1. If there are errors in your code, found by CI or by the maintainers which
   will have a look at your PR, you have to apply changes locally to your
   feature branch and then push again. Your PR will automatically be updated.

Please be aware that the maintainers might request several updates of your code
before it can be merged. This might seem cumbersome, but is is important to have
good code quality when merging the PR, because very often there is not time
later to think through and fix merged code.

### Writing tests

We require 100% test coverage in PRs. Below are some points to consider when
writing tests.

1. Test are located in the directory `tests` and are in individual Python files,
   one for each Python module file in `src/poligrain`, following the naming
   convention e.g. `tests/test_some_module.py` for
   `src/poligrain/some_module.py`.
1. Test should be "unit tests" in the sense that they test the smallest possible
   unit, i.e. individual functions.
1. A test should be named `test_foo_condition` e.g. for function `foo` which is
   called with a specific condition, e.g. with a specific type of keyword
   argument. An example is `test_plot_lines_with_dataarray_colored_lines()`.
1. For your test you will need the expected output. If you do not know
   beforehand what the expected output is, e.g. because your implemented method
   does some complex processing of a long time series, you should apply your
   function in a notebook and then check in detail the output. If you are sure
   that this is the expected output, use it as expected output in the test.
1. Please note that you have to cover all if and else statements in your
   function, also those that just raise an error. Look at existing test to see
   how this is done.
1. You can also check test coverage locally by running `pytest --cov=poligrain`,
   but if you do not have to iterate very often it can be enough to just push
   your changes and wait till the CI run has completed. Be aware, that resources
   are wasted if you push a lot of small changes which all do not results in
   fixing a problem. Try to fix problems locally first.

### Know problems

As with all software-related things, there can be problems even though a
step-by-step guide is followed. It is hard to foresee these problems. Here we
list problems that occurred and were documented.

- `poetry install` gives `ModuleNotFoundError: No module named 'poetry.console'`
  error. Possible fixes are in
  [this Stackoverflow post](https://stackoverflow.com/questions/67813396/modulenotfounderror-no-module-named-poetry-console).

## More info on dev workflow and the used tools

See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/
