"""Tests for pySC.apps.multipoles — systematic and random multipole assignment."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from pySC.apps.multipoles import (
    _extend_magnet_lists,
    _multipoles_to_dict,
    _apply_zero_orders,
    set_systematic_multipoles,
    set_random_multipoles,
    read_multipole_table,
    MultipoleTable,
)


class FakeMagnet:
    """Lightweight stand-in for a magnet settings object."""

    def __init__(self, max_order=2):
        self.max_order = max_order
        self.offset_A = [0.0] * (max_order + 1)
        self.offset_B = [0.0] * (max_order + 1)
        self.A = [0.0] * (max_order + 1)
        self.B = [0.0] * (max_order + 1)
        self.sim_index = 0
        self.update_count = 0

    def update(self):
        self.update_count += 1


class TestExtendMagnetLists:
    """Test _extend_magnet_lists grows field lists correctly."""

    def test_extends_when_needed(self):
        mag = FakeMagnet(max_order=1)
        assert len(mag.offset_A) == 2
        _extend_magnet_lists(mag, 4)
        assert len(mag.offset_A) == 5
        assert len(mag.offset_B) == 5
        assert len(mag.A) == 5
        assert len(mag.B) == 5
        assert mag.max_order == 4

    def test_no_extend_when_within_bounds(self):
        mag = FakeMagnet(max_order=5)
        original_len = len(mag.offset_A)
        _extend_magnet_lists(mag, 3)
        assert len(mag.offset_A) == original_len
        # max_order should NOT change when no extension needed
        assert mag.max_order == 5

    def test_extension_pads_with_zeros(self):
        mag = FakeMagnet(max_order=0)
        mag.offset_A = [1.0]
        _extend_magnet_lists(mag, 2)
        assert mag.offset_A == [1.0, 0.0, 0.0]


class TestMultipolesToDict:
    """Test numpy array to dict conversion."""

    def test_ndarray_conversion(self):
        arr = np.array([[0.0, 0.1], [0.0, 1.0], [0.005, -0.003]])
        result = _multipoles_to_dict(arr)
        assert result == {
            ('B', 1): 0.1,
            ('B', 2): 1.0,
            ('A', 3): 0.005,
            ('B', 3): -0.003,
        }

    def test_skips_zeros(self):
        arr = np.array([[0.0, 0.0], [0.0, 0.5]])
        result = _multipoles_to_dict(arr)
        assert result == {('B', 2): 0.5}

    def test_dict_passthrough(self):
        d = {('A', 1): 0.5}
        assert _multipoles_to_dict(d) is d

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _multipoles_to_dict(np.array([1.0, 2.0]))


class TestApplyZeroOrders:
    """Test zero_orders filtering."""

    def test_removes_specified_orders(self):
        multipoles = {('A', 1): 0.1, ('B', 1): 0.2, ('B', 2): 0.3, ('A', 3): 0.4}
        result = _apply_zero_orders(multipoles, [1, 3])
        assert result == {('B', 2): 0.3}

    def test_none_returns_unchanged(self):
        multipoles = {('B', 1): 0.5}
        assert _apply_zero_orders(multipoles, None) is multipoles


class TestSetSystematicMultipoles:
    """Test set_systematic_multipoles applies values correctly."""

    def _make_sc(self, magnet_names):
        SC = MagicMock()
        magnets = {}
        for name in magnet_names:
            magnets[name] = FakeMagnet(max_order=2)
        SC.magnet_settings.magnets = magnets
        SC.lattice.get_magnet_component.return_value = 10.0  # nominal field
        return SC

    def test_relative_scaling(self):
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 3): 0.01},
            relative_to_nominal=True,
        )
        # 0.01 * 10.0 = 0.1
        assert SC.magnet_settings.magnets["Q1"].offset_B[2] == pytest.approx(0.1)

    def test_absolute_scaling(self):
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("A", 2): 0.05},
            relative_to_nominal=False,
        )
        assert SC.magnet_settings.magnets["Q1"].offset_A[1] == pytest.approx(0.05)

    def test_invalid_component_raises(self):
        SC = self._make_sc(["Q1"])
        with pytest.raises(ValueError, match="Component must be"):
            set_systematic_multipoles(
                SC, ["Q1"],
                multipoles={("C", 1): 0.1},
                relative_to_nominal=False,
            )

    def test_update_called_per_magnet(self):
        SC = self._make_sc(["Q1", "Q2"])
        set_systematic_multipoles(
            SC, ["Q1", "Q2"],
            multipoles={("B", 1): 0.001},
            relative_to_nominal=False,
        )
        for name in ["Q1", "Q2"]:
            assert SC.magnet_settings.magnets[name].update_count == 1

    def test_zero_main_field_uses_unity(self):
        """When nominal field is zero, scaling falls back to 1.0."""
        SC = self._make_sc(["Q1"])
        SC.lattice.get_magnet_component.return_value = 0.0
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.5},
            relative_to_nominal=True,
        )
        # main_field set to 1.0, so value = 0.5 * 1.0
        assert SC.magnet_settings.magnets["Q1"].offset_B[0] == pytest.approx(0.5)

    def test_accumulation(self):
        """Calling twice should accumulate, not overwrite (bug fix)."""
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.001},
            relative_to_nominal=False,
        )
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.002},
            relative_to_nominal=False,
        )
        assert SC.magnet_settings.magnets["Q1"].offset_B[0] == pytest.approx(0.003)

    def test_accumulation_different_components(self):
        """Accumulation works across A and B independently."""
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("A", 2): 0.01, ("B", 2): 0.02},
            relative_to_nominal=False,
        )
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("A", 2): 0.03, ("B", 2): 0.04},
            relative_to_nominal=False,
        )
        assert SC.magnet_settings.magnets["Q1"].offset_A[1] == pytest.approx(0.04)
        assert SC.magnet_settings.magnets["Q1"].offset_B[1] == pytest.approx(0.06)

    def test_main_order_param(self):
        """Explicit main_order overrides magnet.max_order for field lookup."""
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.01},
            relative_to_nominal=True,
            main_order=1,  # dipole
        )
        # Should look up index 0 (main_order - 1 = 0), not max_order=2
        SC.lattice.get_magnet_component.assert_called_with(0, 'B', 0, use_design=True)

    def test_main_component_A(self):
        """main_component='A' looks up the A component for scaling."""
        SC = self._make_sc(["Q1"])
        SC.lattice.get_magnet_component.return_value = 5.0
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.1},
            relative_to_nominal=True,
            main_component='A',
        )
        SC.lattice.get_magnet_component.assert_called_with(0, 'A', 2, use_design=True)
        # 0.1 * 5.0 = 0.5
        assert SC.magnet_settings.magnets["Q1"].offset_B[0] == pytest.approx(0.5)

    def test_numpy_array_input(self):
        """np.ndarray input produces same result as equivalent dict."""
        SC = self._make_sc(["Q1"])
        arr = np.array([[0.0, 0.1], [0.0, 1.0], [0.005, -0.003]])
        set_systematic_multipoles(SC, ["Q1"], multipoles=arr, relative_to_nominal=False)
        mag = SC.magnet_settings.magnets["Q1"]
        assert mag.offset_B[0] == pytest.approx(0.1)
        assert mag.offset_B[1] == pytest.approx(1.0)
        assert mag.offset_A[2] == pytest.approx(0.005)
        assert mag.offset_B[2] == pytest.approx(-0.003)

    def test_zero_orders(self):
        """zero_orders excludes specified orders."""
        SC = self._make_sc(["Q1"])
        set_systematic_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 0.1, ("B", 2): 0.2, ("B", 3): 0.3},
            relative_to_nominal=False,
            zero_orders=[1, 2],
        )
        mag = SC.magnet_settings.magnets["Q1"]
        assert mag.offset_B[0] == pytest.approx(0.0)
        assert mag.offset_B[1] == pytest.approx(0.0)
        assert mag.offset_B[2] == pytest.approx(0.3)


class TestSetRandomMultipoles:
    """Test set_random_multipoles adds random offsets."""

    def test_adds_random_values(self):
        SC = MagicMock()
        mag = FakeMagnet(max_order=2)
        SC.magnet_settings.magnets = {"Q1": mag}
        SC.rng.normal_trunc.return_value = 0.123

        set_random_multipoles(SC, ["Q1"], multipoles={("B", 2): 1.0})

        SC.rng.normal_trunc.assert_called_once_with(loc=0, scale=1.0, sigma_truncate=2)
        assert mag.offset_B[1] == pytest.approx(0.123)

    def test_accumulates_on_existing(self):
        SC = MagicMock()
        mag = FakeMagnet(max_order=2)
        mag.offset_A[0] = 0.5
        SC.magnet_settings.magnets = {"Q1": mag}
        SC.rng.normal_trunc.return_value = 0.1

        set_random_multipoles(SC, ["Q1"], multipoles={("A", 1): 1.0})

        assert mag.offset_A[0] == pytest.approx(0.6)

    def test_invalid_component_raises(self):
        SC = MagicMock()
        SC.magnet_settings.magnets = {"Q1": FakeMagnet()}
        SC.rng.normal_trunc.return_value = 0.0
        with pytest.raises(ValueError, match="Component must be"):
            set_random_multipoles(SC, ["Q1"], multipoles={("X", 1): 1.0})

    def test_numpy_array_input(self):
        SC = MagicMock()
        mag = FakeMagnet(max_order=2)
        SC.magnet_settings.magnets = {"Q1": mag}
        SC.rng.normal_trunc.return_value = 0.05

        arr = np.array([[0.0, 0.5], [0.3, 0.0]])
        set_random_multipoles(SC, ["Q1"], multipoles=arr)

        # Should be called twice: ('B',1)=0.5 and ('A',2)=0.3
        assert SC.rng.normal_trunc.call_count == 2

    def test_zero_orders(self):
        SC = MagicMock()
        mag = FakeMagnet(max_order=2)
        SC.magnet_settings.magnets = {"Q1": mag}
        SC.rng.normal_trunc.return_value = 0.1

        set_random_multipoles(
            SC, ["Q1"],
            multipoles={("B", 1): 1.0, ("B", 2): 1.0, ("A", 3): 1.0},
            zero_orders=[1],
        )
        # Only orders 2 and 3 should be applied
        assert SC.rng.normal_trunc.call_count == 2
        assert mag.offset_B[0] == pytest.approx(0.0)  # order 1 zeroed out


class TestReadMultipoleTable:
    """Test read_multipole_table parses files correctly."""

    def test_basic_parse(self, tmp_path):
        table_file = tmp_path / "quad_multipoles.txt"
        table_file.write_text(
            "# n  PolynomA(n)  PolynomB(n)\n"
            "0    0.0          -0.00171\n"
            "1    0.0           1.0\n"
            "2    0.0          -0.003\n"
        )
        result = read_multipole_table(table_file)
        assert isinstance(result, MultipoleTable)
        assert result.AB.shape == (3, 2)
        assert result.main_order == 2  # 1-based, row 1
        assert result.main_component == 'B'
        # Nominal zeroed
        assert result.AB[1, 1] == 0.0
        # Other values preserved
        assert result.AB[0, 1] == pytest.approx(-0.00171)
        assert result.AB[2, 1] == pytest.approx(-0.003)

    def test_skew_magnet(self, tmp_path):
        """Detects A-component nominal for a skew magnet."""
        table_file = tmp_path / "skew_quad.txt"
        table_file.write_text(
            "0  0.0   0.0\n"
            "1  1.0   0.0\n"
            "2  0.001 0.0\n"
        )
        result = read_multipole_table(table_file)
        assert result.main_order == 2
        assert result.main_component == 'A'
        assert result.AB[1, 0] == 0.0  # nominal zeroed

    def test_comments_and_blanks_ignored(self, tmp_path):
        table_file = tmp_path / "with_comments.txt"
        table_file.write_text(
            "# Header line\n"
            "\n"
            "# Another comment\n"
            "0  0.0  0.5\n"
            "1  0.0  1.0\n"
        )
        result = read_multipole_table(table_file)
        assert result.AB.shape == (2, 2)
        assert result.main_order == 2

    def test_empty_file_raises(self, tmp_path):
        table_file = tmp_path / "empty.txt"
        table_file.write_text("# only comments\n")
        with pytest.raises(ValueError, match="No valid data rows"):
            read_multipole_table(table_file)

    def test_fallback_to_largest_magnitude(self, tmp_path):
        """When no cell is close to 1.0, falls back to largest magnitude."""
        table_file = tmp_path / "no_unity.txt"
        table_file.write_text(
            "0  0.0    0.0\n"
            "1  0.0   50.0\n"
            "2  0.001  0.003\n"
        )
        result = read_multipole_table(table_file)
        assert result.main_order == 2  # row 1 has 50.0 (largest)
        assert result.main_component == 'B'
        assert result.AB[1, 1] == 0.0  # nominal zeroed

    def test_tab_delimited(self, tmp_path):
        table_file = tmp_path / "tabs.txt"
        table_file.write_text("0\t0.0\t0.5\n1\t0.0\t1.0\n")
        result = read_multipole_table(table_file)
        assert result.main_order == 2
        assert result.main_component == 'B'

    def test_integration_with_set_systematic(self, tmp_path):
        """Read a table and feed it directly to set_systematic_multipoles."""
        table_file = tmp_path / "sext.txt"
        table_file.write_text(
            "0  0.0     -0.001\n"
            "1  0.0      0.0\n"
            "2  0.0      1.0\n"
        )
        table = read_multipole_table(table_file)

        SC = MagicMock()
        mag = FakeMagnet(max_order=2)
        SC.magnet_settings.magnets = {"S1": mag}
        SC.lattice.get_magnet_component.return_value = 100.0

        set_systematic_multipoles(
            SC, ["S1"],
            multipoles=table.AB,
            relative_to_nominal=True,
            main_order=table.main_order,
            main_component=table.main_component,
        )
        # Dipole error: -0.001 * 100.0 = -0.1
        assert mag.offset_B[0] == pytest.approx(-0.1)
        # Nominal (order 3, sextupole) was zeroed in table, so no contribution
        assert mag.offset_B[2] == pytest.approx(0.0)
