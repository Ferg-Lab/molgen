"""
Unit and regression test for the molgen package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import molgen


def test_molgen_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "molgen" in sys.modules
