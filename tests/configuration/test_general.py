"""Tests for pySC.configuration.general: get_error, scale_error_table, get_indices_*."""
import pytest
import numpy as np

from pySC.configuration.general import (
    get_error,
    scale_error_table,
    get_indices_with_regex,
    get_indices_and_names,
)


# ---------------------------------------------------------------------------
# get_error
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_get_error_from_table():
    """Looking up an existing key returns the float value."""
    table = {"mag_offset": "0.001", "bpm_noise": "1e-4"}
    result = get_error("mag_offset", table)
    assert result == pytest.approx(0.001)


@pytest.mark.slow
def test_get_error_none_returns_zero():
    """Passing None as error_name always returns 0, regardless of the table."""
    table = {"mag_offset": "0.001"}
    assert get_error(None, table) == 0


@pytest.mark.slow
def test_get_error_missing_key_raises():
    """An unknown key raises an Exception."""
    table = {"mag_offset": "0.001"}
    with pytest.raises(Exception, match="not found"):
        get_error("nonexistent_key", table)


# ---------------------------------------------------------------------------
# scale_error_table
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scale_error_table():
    """Scaling by 2 doubles every value in the table."""
    table = {"a": "0.5", "b": "1e-3"}
    scaled = scale_error_table(table, scale=2)
    assert float(scaled["a"]) == pytest.approx(1.0)
    assert float(scaled["b"]) == pytest.approx(2e-3)


@pytest.mark.slow
@pytest.mark.regression
def test_scale_error_table_does_not_mutate_input():
    """scale_error_table should not modify the original dict.

    Regression: the original implementation mutated error_table in-place.
    """
    table = {"a": "0.5", "b": "1e-3"}
    original_a = table["a"]
    original_b = table["b"]
    scaled = scale_error_table(table, scale=2)
    # Original table should be unchanged
    assert table["a"] == original_a
    assert table["b"] == original_b
    # Scaled table should be a different object
    assert scaled is not table


# ---------------------------------------------------------------------------
# get_indices_with_regex
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_get_indices_with_regex(sc):
    """A regex matching BPM names finds all 10 BPM indices."""
    category_conf = {"regex": "^BPM"}
    indices = get_indices_with_regex(sc, "bpms", category_conf)
    assert len(indices) == 10
    # Indices should be sorted
    assert indices == sorted(indices)


@pytest.mark.slow
def test_get_indices_with_regex_exclude(sc):
    """Regex with exclude pattern removes matches."""
    # Match all quads, then exclude those ending in A
    category_conf = {"regex": "^Q", "exclude": "A$"}
    indices = get_indices_with_regex(sc, "quads", category_conf)

    # All remaining should NOT have names ending in A
    for idx in indices:
        name = sc.lattice.design[idx].FamName
        assert not name.endswith("A"), f"{name} should have been excluded"

    # Verify we got fewer than the total quads
    all_quad_conf = {"regex": "^Q"}
    all_indices = get_indices_with_regex(sc, "quads", all_quad_conf)
    assert len(indices) < len(all_indices)


# ---------------------------------------------------------------------------
# get_indices_and_names (regex mode)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_get_indices_and_names_regex(sc):
    """get_indices_and_names with regex returns matching (indices, names) using lattice naming."""
    category_conf = {"regex": "^BPM"}
    indices, names = get_indices_and_names(sc, "bpms", category_conf)

    assert len(indices) == len(names) == 10
    # Names should come from the lattice naming scheme (FamName since the sc fixture uses naming="FamName")
    for idx, name in zip(indices, names):
        assert name == sc.lattice.get_name_from_index(idx)


# ---------------------------------------------------------------------------
# get_indices_and_names (mapping mode)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_get_indices_and_names_mapping(sc, tmp_path):
    """get_indices_and_names with a YAML mapping file returns (indices, names) from the file."""
    import yaml

    mapping = {"MyQuad1": 5, "MyQuad2": 9}
    mapping_file = str(tmp_path / "mapping.yaml")
    with open(mapping_file, "w") as f:
        yaml.dump(mapping, f)

    category_conf = {"mapping": mapping_file}
    indices, names = get_indices_and_names(sc, "test_map", category_conf)

    assert names == ["MyQuad1", "MyQuad2"]
    assert indices == [5, 9]
