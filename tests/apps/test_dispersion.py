"""Tests for pySC/apps/dispersion.py — dispersion measurement."""

import pytest
import numpy as np

from pySC.apps.dispersion import DispersionMeasurement, DispersionData
from pySC.apps.codes import DispersionCode


class TestDispersionMeasurement:
    def test_initialization(self):
        """DispersionData created with correct delta."""
        dm = DispersionMeasurement(delta=100.0, shots_per_orbit=2, bipolar=True)
        assert dm.dispersion_data is not None
        assert dm.dispersion_data.delta == 100.0
        assert dm.dispersion_data.shots_per_orbit == 2
        assert dm.dispersion_data.bipolar is True

    def test_loop_code_sequence_bipolar(self, mock_interface):
        """Generator yields correct code sequence for bipolar measurement."""
        iface = mock_interface(n_bpms=5)
        dm = DispersionMeasurement(delta=100.0, bipolar=True)
        codes = list(dm.generate(interface=iface, get_output=iface.get_orbit))

        # Bipolar: INITIALIZED, AFTER_GET(center), AFTER_SET(down), AFTER_GET(down),
        #          AFTER_SET(up), AFTER_GET(up), AFTER_RESTORE, DONE
        assert codes[0] == DispersionCode.INITIALIZED
        assert codes[-1] == DispersionCode.DONE
        assert codes[-2] == DispersionCode.AFTER_RESTORE
        assert len(codes) == 8

    def test_loop_code_sequence_unipolar(self, mock_interface):
        """Generator yields correct code sequence for unipolar measurement."""
        iface = mock_interface(n_bpms=5)
        dm = DispersionMeasurement(delta=100.0, bipolar=False)
        codes = list(dm.generate(interface=iface, get_output=iface.get_orbit))

        # Unipolar: INITIALIZED, AFTER_GET(center), AFTER_SET(up), AFTER_GET(up),
        #           AFTER_RESTORE, DONE
        assert codes[0] == DispersionCode.INITIALIZED
        assert codes[-1] == DispersionCode.DONE
        assert len(codes) == 6

    def test_loop_restores_frequency(self, mock_interface):
        """After measurement, RF frequency is restored."""
        iface = mock_interface(n_bpms=5)
        f0 = iface.get_rf_main_frequency()

        dm = DispersionMeasurement(delta=100.0, bipolar=True)
        for _ in dm.generate(interface=iface, get_output=iface.get_orbit):
            pass

        assert iface.get_rf_main_frequency() == pytest.approx(f0)

    def test_calculate_response_bipolar(self, mock_interface):
        """Bipolar: (up - down) / delta gives correct response."""
        n_bpms = 5
        dm = DispersionMeasurement(delta=100.0, bipolar=True)
        data = dm.dispersion_data

        up_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        down_x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        data.raw_orbit_x_up = up_x
        data.raw_orbit_x_down = down_x
        data.raw_orbit_y_up = np.zeros(n_bpms)
        data.raw_orbit_y_down = np.zeros(n_bpms)
        data.raw_orbit_x_err_up = np.zeros(n_bpms)
        data.raw_orbit_x_err_down = np.zeros(n_bpms)
        data.raw_orbit_y_err_up = np.zeros(n_bpms)
        data.raw_orbit_y_err_down = np.zeros(n_bpms)

        dm.calculate_response()
        expected = (up_x - down_x) / 100.0
        np.testing.assert_allclose(data.frequency_response_x, expected)

    def test_calculate_response_unipolar(self, mock_interface):
        """Unipolar: (up - center) / delta gives correct response."""
        n_bpms = 5
        dm = DispersionMeasurement(delta=100.0, bipolar=False)
        data = dm.dispersion_data

        up_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center_x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        data.raw_orbit_x_up = up_x
        data.raw_orbit_x_center = center_x
        data.raw_orbit_y_up = np.zeros(n_bpms)
        data.raw_orbit_y_center = np.zeros(n_bpms)
        data.raw_orbit_x_err_up = np.zeros(n_bpms)
        data.raw_orbit_x_err_center = np.zeros(n_bpms)
        data.raw_orbit_y_err_up = np.zeros(n_bpms)
        data.raw_orbit_y_err_center = np.zeros(n_bpms)

        dm.calculate_response()
        expected = (up_x - center_x) / 100.0
        np.testing.assert_allclose(data.frequency_response_x, expected)

    def test_dispersion_data_save(self, mock_interface, tmp_path):
        """DispersionData.save() writes HDF5 file."""
        iface = mock_interface(n_bpms=5)
        dm = DispersionMeasurement(delta=100.0, bipolar=True)
        for _ in dm.generate(interface=iface, get_output=iface.get_orbit):
            pass

        path = dm.dispersion_data.save(folder_to_save=tmp_path)
        assert path.exists()
        assert path.suffix == '.h5'
