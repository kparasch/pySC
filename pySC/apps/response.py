from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Union, Callable
import datetime
import logging
import numpy as np
from pathlib import Path
from contextlib import nullcontext

from .codes import ResponseCode
from ..utils.file_tools import dict_to_h5
from .tools import get_average_orbit
from .interface import AbstractInterface
from ..core.types import NPARRAY

DISABLE_RICH = False

logger = logging.getLogger(__name__)

def no_rich_progress():
    progress = nullcontext()
    progress.add_task = lambda *args, **kwargs: 1
    progress.update = lambda *args, **kwargs: None
    progress.remove_task = lambda *args, **kwargs: None
    return progress

try:
    from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn
    rich_progress = Progress(
                             TextColumn("[progress.description]{task.description}"),
                             BarColumn(),
                             MofNCompleteColumn(),
                             TimeRemainingColumn(),
                            )
except ModuleNotFoundError:
    rich_progress = no_rich_progress()
    DISABLE_RICH = True

class ResponseData(BaseModel):
    matrix: Optional[NPARRAY] = None
    matrix_err: Optional[NPARRAY] = None

    inputs_delta: list[float]
    shots_per_orbit: int = 1
    bipolar: bool = True

    raw_up: Optional[NPARRAY] = None
    raw_center: Optional[NPARRAY] = None
    raw_down: Optional[NPARRAY] = None

    raw_err_up: Optional[NPARRAY] = None
    raw_err_center: Optional[NPARRAY] = None
    raw_err_down: Optional[NPARRAY] = None

    reference: Optional[NPARRAY] = None
    reference_err: Optional[NPARRAY] = None

    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None

    last_input: Optional[str] = None
    last_number: Optional[int] = None
    timestamp: Optional[float] = None
    original_save_path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def not_normalized_response_matrix(self):
        rm = np.zeros_like(self.matrix)
        for i in range(len(self.input_names)):
            rm[:, i] = self.matrix[:, i] * self.inputs_delta[i]
        return rm


    def save(self, folder_to_save: Optional[Path] = None) -> Path:
        if folder_to_save is None:
            folder_to_save = Path('data')
        time_str = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = Path(folder_to_save) / Path(f'ORM_{time_str}.h5')
        self.original_save_path = str(filename.resolve())
        dict_to_save = self.model_dump()
        dict_to_h5(dict_to_save, filename)
        logger.info(f'Saved data to {filename} .')
        return filename


class ResponseMeasurement(BaseModel):
    inputs_delta: Union[float, list[float]]
    shots_per_orbit: int = 1
    bipolar: bool = True

    input_names: list[str]
    n_outputs: int
    output_names: Optional[list[str]] = None

    last_input: Optional[str] = None
    last_number: int = -1
    timestamp: Optional[float] = None

    response_data: Optional[ResponseData] = None

    _get_output: Optional[Callable] = PrivateAttr(default=None) # to be set at generation of measurement
    _interface: Optional[AbstractInterface] = PrivateAttr(default=None) # to be set at generation of measurement
    _progress: Optional[nullcontext] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def initialize_measurement(self):
        n_inputs = len(self.input_names)

        if type(self.inputs_delta) is float:
            self.inputs_delta = [self.inputs_delta] * n_inputs

        if self.response_data is None:
            zero_matrix_2d = np.zeros((self.n_outputs, n_inputs))
            zero_matrix_1d = np.zeros(self.n_outputs)
            self.response_data = ResponseData(inputs_delta=self.inputs_delta, shots_per_orbit=self.shots_per_orbit, 
                                              bipolar=self.bipolar, input_names=self.input_names, output_names=self.output_names)

            self.response_data.raw_up = zero_matrix_2d.copy()
            self.response_data.raw_err_up = zero_matrix_2d.copy()
            self.response_data.matrix = zero_matrix_2d.copy()
            self.response_data.matrix_err = zero_matrix_2d.copy()

            if self.bipolar:
                self.response_data.raw_down = zero_matrix_2d.copy()
                self.response_data.raw_err_down = zero_matrix_2d.copy()
            else:
                self.response_data.raw_center = zero_matrix_2d.copy()
                self.response_data.raw_err_center = zero_matrix_2d.copy()

            self.response_data.reference = zero_matrix_1d.copy()
            self.response_data.reference_err = zero_matrix_1d.copy()

        return self

    def calculate_response(self):
        if self.bipolar:
            for i in range(len(self.input_names)):
                self.response_data.matrix[:, i] = (self.response_data.raw_up[:, i] - self.response_data.raw_down[:, i]) / self.inputs_delta[i]
                self.response_data.matrix_err[:, i] = np.sqrt(self.response_data.raw_err_up[:, i]**2 + self.response_data.raw_err_down[:, i]**2) / self.inputs_delta[i]
        else:
            for i in range(len(self.input_names)):
                self.response_data.matrix[:, i] = (self.response_data.raw_up[:, i] - self.response_data.raw_center[:, i]) / self.inputs_delta[i]
                self.response_data.matrix_err[:, i] = np.sqrt(self.response_data.raw_err_up[:, i]**2 + self.response_data.raw_err_center[:, i]**2) / self.inputs_delta[i]
        return


    def response_loop(self, inputs, inputs_delta, normalize=True, bipolar=False):
        n_inputs = len(inputs)

        if DISABLE_RICH:
            progress = no_rich_progress()
        else:
            progress = rich_progress

        with progress:
            task_id = progress.add_task("Measuring RM", start=True, total=n_inputs)
            progress.update(task_id, total=n_inputs)

            x_ref, y_ref, x_ref_err, y_ref_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
            ref = np.concat((x_ref.flatten(order='F'), y_ref.flatten(order='F')))
            ref_err = np.concat((x_ref_err.flatten(order='F'), y_ref_err.flatten(order='F')))
            self.response_data.reference = ref
            self.response_data.reference_err = ref_err

            if np.any(np.isnan(x_ref)) or np.any(np.isnan(y_ref)):
                raise ValueError('Initial output is NaN. Aborting. ')

            yield ResponseCode.INITIALIZED

            for i, control in enumerate(inputs):
                ref_setpoint = self._interface.get(control)
                delta = inputs_delta[i]

                if bipolar:
                    step = delta / 2
                    self._interface.set(control, ref_setpoint - step)
                    yield ResponseCode.AFTER_SET
                    x_down, y_down, x_down_err, y_down_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
                    down = np.concat((x_down.flatten(order='F'), y_down.flatten(order='F')))
                    down_err = np.concat((x_down_err.flatten(order='F'), y_down_err.flatten(order='F')))
                    self.response_data.raw_down[:, i] = down
                    self.response_data.raw_err_down[:, i] = down_err
                    yield ResponseCode.AFTER_GET
                else:
                    x_center, y_center, x_center_err, y_center_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
                    center = np.concat((x_center.flatten(order='F'), y_center.flatten(order='F')))
                    center_err = np.concat((x_center_err.flatten(order='F'), y_center_err.flatten(order='F')))
                    self.response_data.raw_center[:, i] = center
                    self.response_data.raw_err_center[:, i] = center_err
                    yield ResponseCode.AFTER_GET

                    step = delta

                self._interface.set(control, ref_setpoint + step)
                yield ResponseCode.AFTER_SET

                x_up, y_up, x_up_err, y_up_err = get_average_orbit(get_orbit=self._get_output, n_orbits=self.shots_per_orbit)
                up = np.concat((x_up.flatten(order='F'), y_up.flatten(order='F')))
                up_err = np.concat((x_up_err.flatten(order='F'), y_up_err.flatten(order='F')))
                self.response_data.raw_up[:, i] = up
                self.response_data.raw_err_up[:, i] = up_err
                yield ResponseCode.AFTER_GET

                self._interface.set(control, ref_setpoint)

                self.last_input = control
                self.response_data.last_input = control
                self.last_number = i
                self.response_data.last_number = i

                progress.update(task_id, completed=i+1, description=f'Measuring response of {control}...')
                yield ResponseCode.AFTER_RESTORE

            self.calculate_response()
            progress.update(task_id, completed=n_inputs, description='Response measured.')
            progress.remove_task(task_id)

        yield ResponseCode.DONE

    def generate(self, interface: AbstractInterface, get_output: Callable):
        """
        step through the measurement.
        """
        self._interface = interface
        self._get_output = get_output
        self.response_data.timestamp = datetime.datetime.now().timestamp()
        for code in self.response_loop(self.input_names, self.inputs_delta, bipolar=self.bipolar):
            yield code


