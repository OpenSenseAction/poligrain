"""
This sets a symbolic link from the `bin` directory of the `poetry` venv to the
pandoc binary which was installed via pip and the package `pypandoc-binary`.
"""

import pathlib
import sys

python_version = sys.version_info

p = pathlib.Path("../.nox/docs/bin/pandoc")
if not p.exists():
    p.symlink_to(
        f"../lib/python{python_version[0]}.{python_version[1]}/site-packages/pypandoc/files/pandoc"
    )
