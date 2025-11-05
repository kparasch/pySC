from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Union, Callable
import datetime
import logging
import numpy as np
from enum import IntEnum
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn
from contextlib import nullcontext

from .codes import ResponseCode
from ..utils.file_tools import dict_to_h5
from ..tuning.tools import get_average_orbit
from .interface import AbstractInterface
from ..core.numpy_type import NPARRAY

DISABLE_RICH = False

logger = logging.getLogger(__name__)

rich_progress = Progress(
                         TextColumn("[progress.description]{task.description}"),
                         BarColumn(),
                         MofNCompleteColumn(),
                         TimeRemainingColumn(),
                        )

def no_rich_progress():
    progress = nullcontext()
    progress.add_task = lambda *args, **kwargs: 1
    progress.update = lambda *args, **kwargs: None
    progress.remove_task = lambda *args, **kwargs: None
    return progress

class ResponseData(BaseModel):
    matrix: NPARRAY

    inputs_delta: Union[float, list[float]]
    shots_per_orbit: int = 1
    bipolar: bool = True

    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None

    last_input: Optional[str] = None
    last_number: Optional[int] = None
    timestamp: Optional[float] = None
    original_save_path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        matrix = np.full((self.n_outputs, n_inputs), np.nan)

        if type(self.inputs_delta) is float:
            self.inputs_delta = [self.inputs_delta] * n_inputs

        if self.response_data is None:
            self.response_data = ResponseData(matrix=matrix, inputs_delta=self.inputs_delta, shots_per_orbit=self.shots_per_orbit, 
                                              bipolar=self.bipolar, input_names=self.input_names, output_names=self.output_names)
        return self

    def response_loop(self, inputs, inputs_delta, get_output, settings, normalize=True, bipolar=False):
        n_inputs = len(inputs)

        if DISABLE_RICH:
            progress = no_rich_progress()
        else:
            progress = rich_progress

        with progress:
            task_id = progress.add_task("Measuring RM", start=True, total=n_inputs)
            progress.update(task_id, total=n_inputs)

            reference = get_output()
            if np.any(np.isnan(reference)):
                raise ValueError('Initial output is NaN. Aborting. ')

            yield ResponseCode.INITIALIZED

            for i, control in enumerate(inputs):
                ref_setpoint = settings.get(control)
                delta = inputs_delta[i]
                if bipolar:
                    step = delta / 2
                    settings.set(control, ref_setpoint - step)
                    reference = get_output()
                else:
                    step = delta
                settings.set(control, ref_setpoint + step)
                output = get_output()

                self.response_data.matrix[:, i] = (output - reference)
                if normalize:
                    self.response_data.matrix[:, i] /= delta
                settings.set(control, ref_setpoint)

                self.last_input = control
                self.response_data.last_input = control
                self.last_number = i
                self.response_data.last_number = i

                progress.update(task_id, completed=i+1, description=f'Measuring response of {control}...')
                yield ResponseCode.MEASURING

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
        for code in self.response_loop(self.input_names, self.inputs_delta, self._get_output, self._interface, bipolar=self.bipolar):
            yield code


