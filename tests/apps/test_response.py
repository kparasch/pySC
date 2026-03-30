"""Tests for pySC/apps/response.py — orbit response matrix measurement."""

import pytest
import numpy as np

from pySC.apps.response import ResponseMeasurement, ResponseData
from pySC.apps.codes import ResponseCode


# ===========================================================================
# Initialization tests
# ===========================================================================

class TestResponseMeasurementInit:
    def test_response_measurement_initialization(self):
        """ResponseData matrices are zeros of correct shape."""
        n_outputs = 20
        input_names = ['CH1', 'CH2', 'CV1']
        meas = ResponseMeasurement(
            inputs_delta=[1e-4, 1e-4, 1e-4],
            input_names=input_names,
            n_outputs=n_outputs,
        )
        rd = meas.response_data
        assert rd.matrix.shape == (n_outputs, 3)
        assert rd.matrix_err.shape == (n_outputs, 3)
        assert rd.raw_up.shape == (n_outputs, 3)
        np.testing.assert_array_equal(rd.matrix, 0)

    def test_response_measurement_scalar_delta(self):
        """Float inputs_delta gets broadcast to list."""
        meas = ResponseMeasurement(
            inputs_delta=2e-4,
            input_names=['CH1', 'CH2'],
            n_outputs=10,
        )
        assert isinstance(meas.inputs_delta, list)
        assert len(meas.inputs_delta) == 2
        assert meas.inputs_delta[0] == pytest.approx(2e-4)
        assert meas.inputs_delta[1] == pytest.approx(2e-4)


# ===========================================================================
# Generator / loop tests (need MockInterface)
# ===========================================================================

class TestResponseLoop:
    def test_response_loop_code_sequence(self, mock_interface):
        """Generator yields INITIALIZED, then AFTER_SET/AFTER_GET/AFTER_RESTORE
        per input, then DONE."""
        n_bpms = 5
        iface = mock_interface(n_bpms=n_bpms)
        input_names = ['CH1', 'CH2']

        meas = ResponseMeasurement(
            inputs_delta=[1e-4, 1e-4],
            input_names=input_names,
            n_outputs=2 * n_bpms,
            bipolar=True,
        )

        codes = list(meas.generate(iface, get_output=iface.get_orbit))

        assert codes[0] == ResponseCode.INITIALIZED
        assert codes[-1] == ResponseCode.DONE

        # For bipolar, each input produces: AFTER_SET, AFTER_GET, AFTER_SET, AFTER_GET, AFTER_RESTORE
        # So between INITIALIZED and DONE we have 5 codes * n_inputs
        middle = codes[1:-1]
        assert len(middle) == 5 * len(input_names)

    def test_response_loop_restores_setpoints(self, mock_interface):
        """After measurement, all correctors are at original setpoints."""
        n_bpms = 5
        iface = mock_interface(n_bpms=n_bpms)
        input_names = ['CH1', 'CH2']
        # Set initial setpoints
        iface.set('CH1', 0.001)
        iface.set('CH2', -0.002)

        meas = ResponseMeasurement(
            inputs_delta=[1e-4, 1e-4],
            input_names=input_names,
            n_outputs=2 * n_bpms,
            bipolar=True,
        )

        original_ch1 = iface.get('CH1')
        original_ch2 = iface.get('CH2')

        list(meas.generate(iface, get_output=iface.get_orbit))

        assert iface.get('CH1') == pytest.approx(original_ch1)
        assert iface.get('CH2') == pytest.approx(original_ch2)


# ===========================================================================
# Calculate response tests
# ===========================================================================

class TestCalculateResponse:
    def test_calculate_response_bipolar(self):
        """(up - down) / delta gives correct RM column."""
        n_outputs = 6
        n_inputs = 2
        delta = [1e-4, 2e-4]

        meas = ResponseMeasurement(
            inputs_delta=delta,
            input_names=['CH1', 'CH2'],
            n_outputs=n_outputs,
            bipolar=True,
        )

        # Manually set raw data
        up = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0],
                       [7.0, 8.0],
                       [9.0, 10.0],
                       [11.0, 12.0]])
        down = np.array([[0.5, 1.0],
                         [1.5, 2.0],
                         [2.5, 3.0],
                         [3.5, 4.0],
                         [4.5, 5.0],
                         [5.5, 6.0]])

        meas.response_data.raw_up = up
        meas.response_data.raw_down = down

        meas.calculate_response()

        expected_col0 = (up[:, 0] - down[:, 0]) / delta[0]
        expected_col1 = (up[:, 1] - down[:, 1]) / delta[1]
        np.testing.assert_allclose(meas.response_data.matrix[:, 0], expected_col0)
        np.testing.assert_allclose(meas.response_data.matrix[:, 1], expected_col1)

    def test_calculate_response_unipolar(self):
        """(up - center) / delta gives correct RM column."""
        n_outputs = 4
        delta = [1e-4]

        meas = ResponseMeasurement(
            inputs_delta=delta,
            input_names=['CH1'],
            n_outputs=n_outputs,
            bipolar=False,
        )

        up = np.array([[2.0], [4.0], [6.0], [8.0]])
        center = np.array([[1.0], [2.0], [3.0], [4.0]])

        meas.response_data.raw_up = up
        meas.response_data.raw_center = center

        meas.calculate_response()

        expected = (up[:, 0] - center[:, 0]) / delta[0]
        np.testing.assert_allclose(meas.response_data.matrix[:, 0], expected)


# ===========================================================================
# Save test
# ===========================================================================

class TestResponseDataSave:
    def test_response_data_save(self, tmp_path):
        """ResponseData.save() writes HDF5 file."""
        import datetime

        rd = ResponseData(
            inputs_delta=[1e-4],
            shots_per_orbit=1,
            bipolar=True,
            input_names=['CH1'],
            output_names=['BPM1_x', 'BPM1_y'],
            timestamp=datetime.datetime(2025, 1, 1).timestamp(),
        )
        rd.matrix = np.zeros((2, 1))
        rd.matrix_err = np.zeros((2, 1))
        rd.raw_up = np.zeros((2, 1))
        rd.raw_err_up = np.zeros((2, 1))
        rd.raw_down = np.zeros((2, 1))
        rd.raw_err_down = np.zeros((2, 1))
        rd.reference = np.zeros(2)
        rd.reference_err = np.zeros(2)

        path = rd.save(folder_to_save=tmp_path)
        assert path.exists()
        assert path.suffix == '.h5'
