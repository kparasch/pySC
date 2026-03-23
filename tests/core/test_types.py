"""Tests for pySC.core.types: NPARRAY type annotation and BaseModelWithSave."""
import json

import numpy as np
import pytest
import yaml
from pydantic import ConfigDict

from pySC.core.types import NPARRAY, BaseModelWithSave


class ArrayModel(BaseModelWithSave):
    """Minimal model using NPARRAY for testing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: NPARRAY


# ---------- NPARRAY coercion ----------

def test_nparray_from_list():
    m = ArrayModel(values=[1, 2, 3])
    assert isinstance(m.values, np.ndarray)
    np.testing.assert_array_equal(m.values, np.array([1, 2, 3]))


def test_nparray_from_ndarray():
    arr = np.array([4.0, 5.0, 6.0])
    m = ArrayModel(values=arr)
    assert isinstance(m.values, np.ndarray)
    np.testing.assert_array_equal(m.values, arr)


def test_nparray_from_nested_list():
    nested = [[1, 2], [3, 4]]
    m = ArrayModel(values=nested)
    assert isinstance(m.values, np.ndarray)
    assert m.values.shape == (2, 2)
    np.testing.assert_array_equal(m.values, np.array(nested))


# ---------- Serialization round-trip ----------

def test_nparray_roundtrip_serialization():
    original = ArrayModel(values=[10, 20, 30])
    data = original.model_dump()
    # Serialized form is a plain list
    assert isinstance(data["values"], list)
    restored = ArrayModel.model_validate(data)
    np.testing.assert_array_equal(restored.values, original.values)


def test_nparray_empty_array():
    m = ArrayModel(values=[])
    assert len(m.values) == 0
    data = m.model_dump()
    restored = ArrayModel.model_validate(data)
    assert len(restored.values) == 0


def test_nparray_nan_inf():
    m = ArrayModel(values=[float("nan"), float("inf"), float("-inf")])
    data = m.model_dump()
    restored = ArrayModel.model_validate(data)
    assert np.isnan(restored.values[0])
    assert np.isposinf(restored.values[1])
    assert np.isneginf(restored.values[2])


# ---------- BaseModelWithSave ----------

def test_save_as_json(tmp_path):
    m = ArrayModel(values=[1, 2, 3])
    path = tmp_path / "test.json"
    m.save_as(path)
    with open(path) as fp:
        data = json.load(fp)
    assert data["values"] == [1, 2, 3]


def test_save_as_yaml(tmp_path):
    m = ArrayModel(values=[1, 2, 3])
    path = tmp_path / "test.yaml"
    m.save_as(path)
    with open(path) as fp:
        data = yaml.safe_load(fp)
    assert data["values"] == [1, 2, 3]


def test_save_as_unknown_suffix_raises(tmp_path):
    m = ArrayModel(values=[1, 2, 3])
    path = tmp_path / "test.xyz"
    with pytest.raises(Exception, match="Unknown file extension"):
        m.save_as(path)
