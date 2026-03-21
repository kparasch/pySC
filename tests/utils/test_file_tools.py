"""Tests for pySC.utils.file_tools: HDF5 serialization utilities."""
import numpy as np
import pytest
import h5py

from pySC.utils.file_tools import dict_to_h5, h5_to_dict, h5_to_obj, overwrite, PYTHON_NONE


class TestDictToH5Roundtrip:

    def test_dict_to_h5_roundtrip(self, tmp_path):
        """dict_to_h5 then h5_to_dict preserves scalar and array data."""
        fname = str(tmp_path / "roundtrip.h5")
        original = {
            "voltage": 3.5e6,
            "count": 42,
            "name": "quadrupole",
            "gains": np.array([1.0, 0.99, 1.01]),
        }
        dict_to_h5(original, fname)
        restored = h5_to_dict(fname)

        assert restored["voltage"] == pytest.approx(original["voltage"])
        assert restored["count"] == original["count"]
        assert restored["name"] == original["name"]
        np.testing.assert_array_equal(restored["gains"], original["gains"])

    def test_dict_to_h5_ndarray(self, tmp_path):
        """Numpy arrays are stored with gzip compression."""
        fname = str(tmp_path / "compressed.h5")
        data = {"matrix": np.random.randn(10, 10)}
        dict_to_h5(data, fname)

        with h5py.File(fname, "r") as fid:
            ds = fid["matrix"]
            assert ds.compression == "gzip"
            assert ds.compression_opts == 9
            np.testing.assert_array_equal(ds[()], data["matrix"])

    def test_dict_to_h5_none_handling(self, tmp_path):
        """Python None is stored as sentinel string and restored as None."""
        fname = str(tmp_path / "nones.h5")
        data = {"empty_field": None, "real_field": 1.0}
        dict_to_h5(data, fname)

        # Verify the sentinel is written to the file
        with h5py.File(fname, "r") as fid:
            raw = fid["empty_field"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            assert raw == PYTHON_NONE

        # Verify roundtrip restores None
        restored = h5_to_dict(fname)
        assert restored["empty_field"] is None
        assert restored["real_field"] == 1.0

    def test_dict_to_h5_string_encoding(self, tmp_path):
        """Strings are stored as bytes in HDF5 and restored as Python str."""
        fname = str(tmp_path / "strings.h5")
        data = {"label": "sextupole_SF2A"}
        dict_to_h5(data, fname)

        # Verify stored as bytes
        with h5py.File(fname, "r") as fid:
            raw = fid["label"][()]
            assert isinstance(raw, bytes)

        # Verify restored as str
        restored = h5_to_dict(fname)
        assert isinstance(restored["label"], str)
        assert restored["label"] == "sextupole_SF2A"


class TestOverwrite:

    def test_overwrite(self, tmp_path):
        """overwrite modifies values in an existing HDF5 file."""
        fname = str(tmp_path / "overwrite.h5")
        original = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        dict_to_h5(original, fname)

        # Overwrite only 'x'
        new_x = np.array([10.0, 20.0, 30.0])
        overwrite({"x": new_x}, fname)

        restored = h5_to_dict(fname)
        np.testing.assert_array_equal(restored["x"], new_x)
        np.testing.assert_array_equal(restored["y"], original["y"])


class TestH5ToObj:

    def test_h5_to_obj(self, tmp_path):
        """h5_to_obj returns an object with attributes matching dict keys."""
        fname = str(tmp_path / "obj.h5")
        data = {
            "beta_x": np.array([10.0, 12.0]),
            "tune": 0.31,
            "name": "BPM_01",
            "empty": None,
        }
        dict_to_h5(data, fname)
        obj = h5_to_obj(fname)

        np.testing.assert_array_equal(obj.beta_x, data["beta_x"])
        assert obj.tune == pytest.approx(data["tune"])
        assert obj.name == data["name"]
        assert obj.empty is None

    def test_h5_to_obj_no_extra_attrs(self, tmp_path):
        """h5_to_obj should not add attributes beyond what is in the dict."""
        fname = str(tmp_path / "obj2.h5")
        data = {"alpha": 1.0}
        dict_to_h5(data, fname)
        obj = h5_to_obj(fname)

        # Only the dict key plus any inherited object attrs should exist
        assert hasattr(obj, "alpha")
        assert not hasattr(obj, "beta")


class TestGroupSupport:

    def test_dict_to_h5_with_group(self, tmp_path):
        """dict_to_h5 and h5_to_dict work correctly with named groups."""
        fname = str(tmp_path / "grouped.h5")
        data = {"value": np.array([1.0, 2.0])}
        dict_to_h5(data, fname, group="my_group")

        restored = h5_to_dict(fname, group="my_group")
        np.testing.assert_array_equal(restored["value"], data["value"])
