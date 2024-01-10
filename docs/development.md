# Development

## Guidelines

- PR
- formatting
- test coverage
- ...

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
   the ReadTheDocs processes are running on the account of `cchwala.

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
