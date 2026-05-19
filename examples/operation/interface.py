"""Control-system interface skeleton for operational pySC examples."""

from pathlib import Path

import numpy as np

from pySC.apps.interface import AbstractInterface


DATA_FOLDER = Path("data")


class Interface(AbstractInterface):
    """Implement these methods for the target control system."""

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current horizontal and vertical orbit arrays."""
        raise NotImplementedError("Implement get_orbit() for your control system.")

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the horizontal and vertical reference orbit arrays."""
        raise NotImplementedError("Implement get_ref_orbit() for your control system.")

    def get(self, name: str) -> float:
        """Return one magnet strength in physics units."""
        raise NotImplementedError("Implement get() for your control system.")

    def set(self, name: str, value: float) -> None:
        """Set one magnet strength in physics units and wait until it is settled."""
        raise NotImplementedError("Implement set() for your control system.")

    def get_many(self, names: list[str]) -> dict[str, float]:
        """Return magnet strengths as a mapping from control name to value."""
        raise NotImplementedError("Implement get_many() for your control system.")

    def set_many(self, data: dict[str, float]) -> None:
        """Set multiple magnet strengths and wait until they are settled."""
        raise NotImplementedError("Implement set_many() for your control system.")

    def get_rf_main_frequency(self) -> float:
        """Return the main RF frequency in Hz if RF correction is used."""
        raise NotImplementedError("Implement get_rf_main_frequency() if RF correction is used.")

    def set_rf_main_frequency(self, frequency: float) -> None:
        """Set the main RF frequency in Hz if RF correction is used."""
        raise NotImplementedError("Implement set_rf_main_frequency() if RF correction is used.")


class InterfaceInjection(Interface):
    """Interface variant for first-turn or multi-turn trajectory measurements."""

    n_turns: int = 1
    trigger_injection: bool = False

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Return turn-by-turn trajectory arrays flattened with order='F'."""
        raise NotImplementedError("Implement get_orbit() for turn-by-turn data.")

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the matching reference trajectory flattened with order='F'."""
        raise NotImplementedError("Implement get_ref_orbit() for turn-by-turn data.")
