"""Tests for pySC.apps.codes — IntEnum status codes."""
import pytest

from pySC.apps.codes import BBACode, ResponseCode, DispersionCode


class TestBBACode:
    def test_bba_code_values(self):
        assert BBACode.HYSTERESIS == 1
        assert BBACode.HYSTERESIS_DONE == 2
        assert BBACode.HORIZONTAL == 3
        assert BBACode.HORIZONTAL_DONE == 4
        assert BBACode.VERTICAL == 5
        assert BBACode.VERTICAL_DONE == 6
        assert BBACode.DONE == 7


class TestResponseCode:
    def test_response_code_values(self):
        assert ResponseCode.INITIALIZED == 0
        assert ResponseCode.AFTER_SET == 3
        assert ResponseCode.AFTER_GET == 4
        assert ResponseCode.AFTER_RESTORE == 5
        assert ResponseCode.MEASURING == 5
        assert ResponseCode.DONE == 6

    def test_measuring_equals_after_restore(self):
        """MEASURING is intentionally aliased to AFTER_RESTORE (both == 5)."""
        assert ResponseCode.MEASURING == ResponseCode.AFTER_RESTORE
        # In IntEnum, the first name wins; MEASURING is an alias.
        assert ResponseCode(5).name == 'AFTER_RESTORE'


class TestDispersionCode:
    def test_dispersion_code_values(self):
        assert DispersionCode.INITIALIZED == 0
        assert DispersionCode.AFTER_SET == 3
        assert DispersionCode.AFTER_GET == 4
        assert DispersionCode.AFTER_RESTORE == 5
        assert DispersionCode.MEASURING == 5
        assert DispersionCode.DONE == 6

    @pytest.mark.regression
    def test_dispersion_code_done_is_distinct(self):
        """Regression: DispersionCode.DONE previously collided with AFTER_GET (both 4).
        Now DONE = 6 is distinct, matching the ResponseCode pattern.
        """
        assert DispersionCode.DONE == 6
        assert DispersionCode.DONE != DispersionCode.AFTER_GET
        assert DispersionCode(6).name == 'DONE'
        assert DispersionCode(4).name == 'AFTER_GET'

        # Both ResponseCode and DispersionCode now have DONE = 6
        assert ResponseCode.DONE == 6
        assert ResponseCode(6).name == 'DONE'
