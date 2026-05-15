"""Apps-layer fixtures: MockInterface for measurement algorithm tests."""
import numpy as np
import pytest

from pySC.apps.interface import AbstractInterface


class MockInterface(AbstractInterface):
    """Mock interface that stores setpoints and returns synthetic orbits.

    Orbit response: when a setpoint changes by delta, the orbit shifts
    by `response_scale * delta` at all BPMs (simple proportional model).
    """

    _setpoints: dict = {}
    _orbit_x: np.ndarray = np.array([])
    _orbit_y: np.ndarray = np.array([])
    _rf_frequency: float = 500e6
    _response_scale: float = 1e-3

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, n_bpms: int = 10, response_scale: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self._setpoints = {}
        self._orbit_x = np.zeros(n_bpms)
        self._orbit_y = np.zeros(n_bpms)
        self._rf_frequency = 500e6
        self._response_scale = response_scale

    # --- AbstractInterface contract (5 methods) ---

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        return self._orbit_x.copy(), self._orbit_y.copy()

    def get(self, name: str) -> float:
        return self._setpoints.get(name, 0.0)

    def set(self, name: str, value: float):
        old = self._setpoints.get(name, 0.0)
        self._setpoints[name] = value
        delta = value - old
        self._orbit_x += self._response_scale * delta
        self._orbit_y += self._response_scale * delta * 0.5

    def get_many(self, names: list) -> dict[str, float]:
        return {name: self.get(name) for name in names}

    def set_many(self, data: dict[str, float]):
        for name, value in data.items():
            self.set(name, value)

    # --- Additional RF methods (for dispersion tests, NOT in AbstractInterface) ---

    def get_rf_main_frequency(self) -> float:
        return self._rf_frequency

    def set_rf_main_frequency(self, value: float):
        self._rf_frequency = value


@pytest.fixture
def mock_interface():
    """Factory fixture: returns a MockInterface with configurable n_bpms."""
    def _make(n_bpms=10, response_scale=1e-3):
        return MockInterface(n_bpms=n_bpms, response_scale=response_scale)
    return _make
