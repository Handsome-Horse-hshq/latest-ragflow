"""Minimal checks for the phase-one project skeleton."""

import rag_ds


def test_package_imports() -> None:
    """The source package can be imported by the test environment."""
    assert rag_ds.__version__ == "0.1.0"

