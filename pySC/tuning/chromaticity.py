from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging

from ..core.types import NPARRAY

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class Chromaticity(BaseModel, extra="forbid"):
    controls_1: list[str] = []
    controls_2: list[str] = []
    response_matrix: Optional[NPARRAY] = None
    inverse_response_matrix: Optional[NPARRAY] = None
    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def design_dqx(self):
        return self._parent._parent.lattice.twiss['dqx']

    @property
    def design_dqy(self):
        return self._parent._parent.lattice.twiss['dqy']

    def chromaticity_response(self, controls: list[str], delta: float = 1e-5):
        SC = self._parent._parent
        dq1_i, dq2_i = SC.lattice.get_chromaticity(use_design=True)

        ref_data = SC.design_magnet_settings.get_many(controls)
        data = {key: ref_data[key] + delta for key in ref_data.keys()}
        SC.design_magnet_settings.set_many(data)

        dq1_f, dq2_f = SC.lattice.get_chromaticity(use_design=True)

        SC.design_magnet_settings.set_many(ref_data)

        delta_dq1 = (dq1_f - dq1_i) / delta
        delta_dq2 = (dq2_f - dq2_i) / delta
        return delta_dq1, delta_dq2

    def build_response_matrix(self, delta: float = 1e-5) -> None:
        if not len(self.controls_1) > 0:
            raise Exception('chromaticity.controls_1 is empty. Please set.')
        if not len(self.controls_2) > 0:
            raise Exception('chromaticity.controls_2 is empty. Please set.')

        RM = np.zeros((2,2))
        RM[:, 0] = self.chromaticity_response(self.controls_1, delta=delta)
        RM[:, 1] = self.chromaticity_response(self.controls_2, delta=delta)
        iRM = np.linalg.inv(RM)

        self.response_matrix = RM
        self.inverse_response_matrix = iRM
        return

    def trim(self, delta_dqx: float = 0, delta_dqy: float = 0, use_design: bool = False) -> None:
        SC = self._parent._parent
        if self.inverse_response_matrix is None:
            logger.info('Did not find inverse (chromaticity) response matrix. Building now.')
            self.build_response_matrix()

        delta1, delta2 = np.dot(self.inverse_response_matrix, [delta_dqx, delta_dqy])
        ref_data1 = SC.magnet_settings.get_many(self.controls_1, use_design=use_design)
        ref_data2 = SC.magnet_settings.get_many(self.controls_2, use_design=use_design)
        data1 = {key: ref_data1[key] + delta1 for key in ref_data1.keys()}
        data2 = {key: ref_data2[key] + delta2 for key in ref_data2.keys()}

        SC.magnet_settings.set_many(data1, use_design=use_design)
        SC.magnet_settings.set_many(data2, use_design=use_design)
        return

    def correct(self, target_dqx: Optional[float] = None, target_dqy: Optional[float] = None,
                n_iter: int = 1, gain: float = 1, measurement_method: str = 'cheat'):
        '''
        Correct the chromaticity to the target values.
        Parameters
        ----------
        target_dqx : float, optional
            Target horizontal chromaticity. If None, use design chromaticity.
        target_dqy : float, optional
            Target vertical chromaticity. If None, use design chromaticity.
        n_iter : int, optional
            Number of correction iterations. Default is 1.
        gain : float, optional
            Gain for the correction. Default is 1.
        measurement_method : str, optional
            Method to measure the chromaticity. Options are 'cheat', 'cheat4d'.
            Default is 'cheat'.
        '''

        if measurement_method not in ['cheat', 'cheat4d']:
            raise NotImplementedError(f'{measurement_method=} not implemented yet.')

        if target_dqx is None:
            target_dqx = self.design_dqx

        if target_dqy is None:
            target_dqy = self.design_dqy

        for _ in range(n_iter):
            if measurement_method == 'cheat':
                dqx, dqy = self.cheat()
            elif measurement_method == 'cheat4d':
                dqx, dqy = self.cheat4d()
            else:
                raise Exception(f'Unknown measurement_method {measurement_method}')
            if dqx is None or dqy is None or dqx != dqx or dqy != dqy:
                logger.info("Chromaticity measurement failed, skipping correction.")
                return
            logger.info(f"Measured tune: dqx={dqx:.4f}, dqy={dqy:.4f}")
            delta_dqx = dqx - target_dqx
            delta_dqy = dqy - target_dqy
            logger.info(f"Delta tune: delta_dqx={delta_dqx:.4f}, delta_dqy={delta_dqy:.4f}")
            self.trim(delta_dqx=-gain*delta_dqx, delta_dqy=-gain*delta_dqy)
        return

    def cheat4d(self) -> tuple[float, float]:
        SC = self._parent._parent
        dqx, dqy = SC.lattice.get_chromaticity(method='4d')
        delta_x = dqx - self.design_dqx
        delta_y = dqy - self.design_dqy
        logger.info(f"Horizontal chromaticity: dQx = {dqx:.3f} (Δ = {delta_x:.3f})")
        logger.info(f"Vertical chromaticity: dQy = {dqy:.3f} (Δ = {delta_y:.3f})")

        return dqx, dqy

    def cheat(self) -> tuple[float, float]:
        SC = self._parent._parent
        dqx, dqy = SC.lattice.get_chromaticity(method='6d')
        delta_x = dqx - self.design_dqx
        delta_y = dqy - self.design_dqy
        logger.info(f"Horizontal tune: Qx = {dqx:.3f} (Δ = {delta_x:.3f})")
        logger.info(f"Vertical tune: Qy = {dqy:.3f} (Δ = {delta_y:.3f})")

        return dqx, dqy