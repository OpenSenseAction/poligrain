# Development

## Guidelines

- PR
- formatting
- test coverage
- ...

## Releasing packages

[PyPA-Build](https://pypa-build.readthedocs.io/en/latest/) is used to release
packages on PyPi. This is automatically done when a release is published via the
github WebGUI. See more details
[here](https://learn.scientific-python.org/development/guides/gha-pure/#distribution-pure-python-wheels).

Note that this requires a manual activation of the connection between this repo
and PyPi. A guide how to do that can be found
[here](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
The workflow is called `cd.yml` and the environment is called `pypi`.

Steps for releasing on pypi:

1. Add a git tag (locally or on github) with the version number like `v0.2.4`
   following semantic versioning [semver](https://semver.org/). If you do this
   locally, push the tags to github via `git push --tags`.
2. Create a release on github using the tag that was just creatd and name the
   release the same as the tag, e.g. `v0.2.4`. This will automatically trigger
   the CD run that publishes to PyPi.

Note that the version numbers in pyprojects.toml and `src/poligrain/__init__.py`
are only set to the version number from the git tag during the build process.
Hence, they stay at 0.0.0 in the version in the repo but are correct in the
sdist or wheel.

## Repository structure and tooling

### Initialization of repo structure via cookiecutter template

The repo structure is based on the cookiecutter template
https://github.com/scientific-python/cookie and was initialized using `cruft` to
seamlessly apply updates when the template changes.

During initialization with the cookiecutter template `poetry` was chosen to
manage packaging. Many best-practice tools are provides by the templates, e.g.
`ruff` for linting and docs generation via `sphinx` for ReadTheDocs. For
details, please see the info at the website of the template.

Below is a documentation of the changes that have been applied to the repo
structure, which hopefully helps us when using the template in an another repo.
(Of course we could fork and change the template but that is most likely not
worth it since our changes might be few and small.)

### Steps taken after repo structure initialization to get build processes going

1. Needed to manually activate ReadTheDoc build process for this repo. Currently
   the ReadTheDocs processes are running on the account of `cchwala`.
2. Connection to PyPi has to be established manually by allowing and configuring
   the creation of a PyPI project with a "trusted publisher" (see info in
   section [Releasing packages](#releasing-packages) above and in more detail in
   the
   [pypi docs](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)).
   This is currently done via the account of `cchwala`.

### Changes to default repo structure

None yet.

### Solutions for common problems with tooling

#### Use `poetry` environment in VSCode

In VSCode you can select which Python environment shall be used for code
introspection. By default the `poetry` env that is used by the repo, as it is
initialized by the template, is not found by VSCode. There is a solution in
[this answer on stackoverflow](https://stackoverflow.com/a/64434542/356463)
which is just one line of code (if you do this before creating the env with
`poetry` for the first time)

```
poetry config virtualenvs.in-project true
```

or

```
poetry env list  # shows the name of the current environment
poetry env remove <current environment>
poetry install  # will create a new environment using your updated configuration
```

if you do this after you have already created the `poetry` env.
