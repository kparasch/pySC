from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Tuple, Callable
import datetime
import logging
import numpy as np
from pathlib import Path

from .codes import DispersionCode
from ..utils.file_tools import dict_to_h5
from .tools import get_average_orbit
from .interface import AbstractInterface
from ..core.types import NPARRAY

logger = logging.getLogger(__name__)

class DispersionData(BaseModel):
    raw_orbit_x_up: Optional[NPARRAY] = None
    raw_orbit_y_up: Optional[NPARRAY] = None
    raw_orbit_x_center: Optional[NPARRAY] = None
    raw_orbit_y_center: Optional[NPARRAY] = None
    raw_orbit_x_down: Optional[NPARRAY] = None
    raw_orbit_y_down: Optional[NPARRAY] = None

    raw_orbit_x_err_up: Optional[NPARRAY] = None
    raw_orbit_y_err_up: Optional[NPARRAY] = None
    raw_orbit_x_err_center: Optional[NPARRAY] = None
    raw_orbit_y_err_center: Optional[NPARRAY] = None
    raw_orbit_x_err_down: Optional[NPARRAY] = None
    raw_orbit_y_err_down: Optional[NPARRAY] = None

    frequency_response_x: Optional[NPARRAY] = None
    frequency_response_y: Optional[NPARRAY] = None
    frequency_response_x_err: Optional[NPARRAY] = None
    frequency_response_y_err: Optional[NPARRAY] = None

    delta: float
    momentum_compaction: Optional[float] = None
    shots_per_orbit: int = 1
    bipolar: bool = True

    timestamp: Optional[float] = None
    original_save_path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def frequency_response(self) -> Tuple[NPARRAY]:
        return self.frequency_response_x, self.frequency_response_y

    @property
    def not_normalized_frequency_response(self) -> Tuple[NPARRAY]:
        return self.frequency_response_x * self.delta, self.frequency_response_y * self.delta

    def save(self, folder_to_save: Optional[Path] = None) -> Path:
        if folder_to_save is None:
            folder_to_save = Path('data')
        time_str = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = Path(folder_to_save) / Path(f'Dispersion_{time_str}.h5')
        self.original_save_path = str(filename.resolve())
        dict_to_save = self.model_dump()
        dict_to_h5(dict_to_save, filename)
        logger.info(f'Saved data to {filename} .')
        return filename


class DispersionMeasurement(BaseModel):
    delta: float
    shots_per_orbit: int = 1
    bipolar: bool = True

    timestamp: Optional[float] = None

    dispersion_data: Optional[DispersionData] = None

    _get_output: Optional[Callable] = PrivateAttr(default=None) # to be set at generation of measurement
    _interface: Optional[AbstractInterface] = PrivateAttr(default=None) # to be set at generation of measurement

    @model_validator(mode="after")
    def initialize_measurement(self):
        if self.dispersion_data is None:
            self.dispersion_data = DispersionData(delta=self.delta, shots_per_orbit=self.shots_per_orbit, bipolar=self.bipolar)
        return self

    def calculate_response(self):
        if self.bipolar:
            self.dispersion_data.frequency_response_x = (self.dispersion_data.raw_orbit_x_up - self.dispersion_data.raw_orbit_x_down) / self.delta
            self.dispersion_data.frequency_response_y = (self.dispersion_data.raw_orbit_y_up - self.dispersion_data.raw_orbit_y_down) / self.delta
            self.dispersion_data.frequency_response_x_err = np.sqrt(self.dispersion_data.raw_orbit_x_err_up**2 + self.dispersion_data.raw_orbit_x_err_down**2) / self.delta
            self.dispersion_data.frequency_response_y_err = np.sqrt(self.dispersion_data.raw_orbit_y_err_up**2 + self.dispersion_data.raw_orbit_y_err_down**2) / self.delta
        else:
            self.dispersion_data.frequency_response_x = (self.dispersion_data.raw_orbit_x_up - self.dispersion_data.raw_orbit_x_center) / self.delta
            self.dispersion_data.frequency_response_y = (self.dispersion_data.raw_orbit_y_up - self.dispersion_data.raw_orbit_y_center) / self.delta
            self.dispersion_data.frequency_response_x_err = np.sqrt(self.dispersion_data.raw_orbit_x_err_up**2 + self.dispersion_data.raw_orbit_x_err_center**2) / self.delta
            self.dispersion_data.frequency_response_y_err = np.sqrt(self.dispersion_data.raw_orbit_y_err_up**2 + self.dispersion_data.raw_orbit_y_err_center**2) / self.delta
        return

    def response_loop(self, bipolar=False):

        frequency = self._interface.get_rf_main_frequency()
        assert frequency is not None, "Could not get RF frequency from interface."

        yield DispersionCode.INITIALIZED

        x_center, y_center, x_center_err, y_center_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
        self.dispersion_data.raw_orbit_x_center = x_center
        self.dispersion_data.raw_orbit_y_center = y_center
        self.dispersion_data.raw_orbit_x_err_center = x_center_err
        self.dispersion_data.raw_orbit_y_err_center = y_center_err
        yield DispersionCode.AFTER_GET

        if bipolar:
            logger.debug("Stepping frequency down.")
            self._interface.set_rf_main_frequency(frequency - self.delta / 2)
            yield DispersionCode.AFTER_SET

            x_down, y_down, x_down_err, y_down_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
            self.dispersion_data.raw_orbit_x_down = x_down
            self.dispersion_data.raw_orbit_y_down = y_down
            self.dispersion_data.raw_orbit_x_err_down = x_down_err
            self.dispersion_data.raw_orbit_y_err_down = y_down_err
            yield DispersionCode.AFTER_GET

            logger.debug("Stepping frequency up.")
            self._interface.set_rf_main_frequency(frequency + self.delta / 2)
        else:
            logger.debug("Stepping frequency up.")
            self._interface.set_rf_main_frequency(frequency + self.delta)
        yield DispersionCode.AFTER_SET

        x_up, y_up, x_up_err, y_up_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
        self.dispersion_data.raw_orbit_x_up = x_up
        self.dispersion_data.raw_orbit_y_up = y_up
        self.dispersion_data.raw_orbit_x_err_up = x_up_err
        self.dispersion_data.raw_orbit_y_err_up = y_up_err
        yield DispersionCode.AFTER_GET

        logger.debug("Setting original frequency.")
        self._interface.set_rf_main_frequency(frequency)
        yield DispersionCode.AFTER_RESTORE
        self.calculate_response()
        yield DispersionCode.DONE

    def generate(self, interface: AbstractInterface, get_output: Callable):
        """
        step through the measurement.
        """
        self._interface = interface
        self._get_output = get_output
        self.dispersion_data.timestamp = datetime.datetime.now().timestamp()
        self.timestamp = self.dispersion_data.timestamp
        for code in self.response_loop(bipolar=self.bipolar):
            yield code


