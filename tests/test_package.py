from __future__ import annotations

import importlib.metadata

import poligrain as m


def test_version():
    assert importlib.metadata.version("poligrain") == m.__version__
