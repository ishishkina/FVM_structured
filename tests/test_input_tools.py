"""Parsing and validation of the key=value input file."""
import pytest

import InputTools


def _write(tmp_path, text):
    path = tmp_path / "input.txt"
    path.write_text(text)
    return str(path)


def test_parses_types_and_optional_data(tmp_path):
    init = InputTools.Initializer(_write(tmp_path, "mesh=Grids/base.msh\niterations=3\nmod=2\n"))
    assert init.mesh == "Grids/base.msh"
    assert init.iterations == 3 and isinstance(init.iterations, int)
    assert init.mod == 2
    assert init.data == ""


def test_missing_mesh_is_rejected(tmp_path):
    with pytest.raises(AttributeError):
        InputTools.Initializer(_write(tmp_path, "iterations=1\nmod=0\n"))


def test_non_integer_iterations_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        InputTools.Initializer(_write(tmp_path, "mesh=a.msh\niterations=1.5\nmod=0\n"))


def test_invalid_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        InputTools.Initializer(_write(tmp_path, "mesh=a.msh\niterations=1\nmod=0\n2bad=1\n"))
