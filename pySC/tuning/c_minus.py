from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging

from ..core.control import KnobControl, KnobData
from ..utils import rdt
from ..apps.response_matrix import ResponseMatrix

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class CMinus(BaseModel, extra="forbid"):
    knob_real: str = 'c_minus_real'
    knob_imag: str = 'c_minus_imag'
    controls: list[str] = []
    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def c_minus_response(self, delta: float = 1e-5):
        SC = self._parent._parent

        N = len(self.controls)
        assert N > 0

        c_minus_0 = rdt.calculate_c_minus(SC, use_design=True)

        delta_c_minus = np.zeros_like(self.controls, dtype=complex)
        for ii, control in enumerate(self.controls):
            logger.info(f'[{ii+1:d}/{N:d}] Calculating ideal c_minus response of {control}.')
            sp0 = SC.magnet_settings.get(control, use_design=True)
            SC.magnet_settings.set(control, delta + sp0, use_design=True)
            c_minus = rdt.calculate_c_minus(SC, use_design=True)
            SC.magnet_settings.set(control, sp0, use_design=True)
            delta_c_minus[ii] = (c_minus - c_minus_0) / delta

        return delta_c_minus

    def create_c_minus_knobs(self, delta: float = 1e-5) -> None:
        if not len(self.controls) > 0:
            raise Exception('c_minus.controls is empty. Please set.')

        matrix = np.zeros((2,len(self.controls)))
        delta_c_minus = self.c_minus_response(delta=delta)
        matrix[0] = delta_c_minus.real
        matrix[1] = delta_c_minus.imag

        c_minus_response_matrix = ResponseMatrix(matrix=matrix,
                                                 outputs_plane=['SQ'] * len(self.controls)
                                                )
        inverse_matrix = c_minus_response_matrix.build_pseudoinverse().matrix
        c_minus_real_knob = inverse_matrix[:, 0]
        c_minus_imag_knob = inverse_matrix[:, 1]

        knob_data = KnobData(data={
            self.knob_real: KnobControl(control_names=self.controls, weights=list(c_minus_real_knob)),
            self.knob_imag: KnobControl(control_names=self.controls, weights=list(c_minus_imag_knob))
            })

        logger.info(f"{self.knob_real}: sum(|weights|)={np.sum(np.abs(c_minus_real_knob)):.2e}")
        logger.info(f"{self.knob_imag}: sum(|weights|)={np.sum(np.abs(c_minus_imag_knob)):.2e}")
        return knob_data

    def trim(self, real: float = 0, imag: float = 0, use_design: bool = False) -> None:
        SC = self._parent._parent

        if use_design:
            assert self.knob_real in SC.design_magnet_settings.controls.keys()
            assert self.knob_imag in SC.design_magnet_settings.controls.keys()
        else:
            assert self.knob_real in SC.magnet_settings.controls.keys()
            assert self.knob_imag in SC.magnet_settings.controls.keys()

        real0 = SC.magnet_settings.get(self.knob_real, use_design=use_design)
        imag0 = SC.magnet_settings.get(self.knob_imag, use_design=use_design)

        SC.magnet_settings.set(self.knob_real, real0 + real, use_design=use_design)
        SC.magnet_settings.set(self.knob_imag, imag0 + imag, use_design=use_design)

        return

    def correct(self, target_c_minus_real: float = 0, target_c_minus_imag: float = 0,
                n_iter: int = 1, gain: float = 1, measurement_method: str = 'cheat') -> None:
        '''
        Correct c_minus to the target values.
        Parameters
        ----------
        target_c_minus_real : float
            Target real c_minus. Default is 0.
        target_c_minus_imag : float, optional
            Target imaginary c_minus. Default is 0.
        n_iter : int, optional
            Number of correction iterations. Default is 1.
        gain : float, optional
            Gain for the correction. Default is 1.
        measurement_method : str, optional
            Method to measure c_minus. Options are 'cheat'.
            Default is 'cheat'.
        '''

        if measurement_method not in ['cheat']:
            raise NotImplementedError(f'{measurement_method=} not implemented yet.')

        for _ in range(n_iter):
            if measurement_method == 'cheat':
                c_minus = self.cheat()
            else:
                raise Exception(f'Unknown measurement_method {measurement_method}')
            if c_minus is None or c_minus != c_minus:
                logger.info("C_minus measurement failed, skipping correction.")
                return
            logger.info(f"Measured c_minus = {c_minus:.4f}")
            delta_real = c_minus.real - target_c_minus_real
            delta_imag = c_minus.imag - target_c_minus_imag
            self.trim(real=-gain*delta_real, imag=-gain*delta_imag)
        return

    def cheat(self, use_design: bool = False) -> complex:
        SC = self._parent._parent
        try:
            c_minus = rdt.calculate_c_minus(SC, use_design=use_design)
        except Exception as exc:
            logger.warning(f"Exception while measuring c_minus: {exc}")
            c_minus = np.nan

        return c_minus