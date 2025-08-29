from typing import Dict, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging

from ..core.numpy_type import NPARRAY

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class Tune(BaseModel, extra="forbid"):
    tune_quad_controls_1: list[str] = []
    tune_quad_controls_2: list[str] = []
    tune_response_matrix: Optional[NPARRAY] = None
    inverse_tune_response_matrix: Optional[NPARRAY] = None
    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def design_qx(self):
        return self._parent._parent.lattice.twiss['qx']

    @property
    def design_qy(self):
        return self._parent._parent.lattice.twiss['qy']

    def tune_response(self, quads: list[str], dk: float = 1e-5):
        SC = self._parent._parent
        twiss = SC.lattice.get_twiss(use_design=True)
        q1_i, q2_i = twiss['qx'], twiss['qy']

        ref_data = SC.design_magnet_settings.get_many(quads)
        data = {key: ref_data[key] + dk for key in ref_data.keys()}
        SC.design_magnet_settings.set_many(data)

        twiss = SC.lattice.get_twiss(use_design=True)
        q1_f, q2_f = twiss['qx'], twiss['qy']

        SC.design_magnet_settings.set_many(ref_data)

        dq1 = (q1_f - q1_i)/dk
        dq2 = (q2_f - q2_i)/dk
        return dq1, dq2

    def build_tune_response_matrix(self, dk: float = 1e-5) -> None:
        if not len(self.tune_quad_controls_1) > 0:
            raise Exception('tune_quad_controls_1 is empty. Please set.')
        if not len(self.tune_quad_controls_2) > 0:
            raise Exception('tune_quad_controls_2 is empty. Please set.')

        TRM = np.zeros((2,2))
        TRM[:, 0] = self.tune_response(self.tune_quad_controls_1, dk=dk)
        TRM[:, 1] = self.tune_response(self.tune_quad_controls_2, dk=dk)
        iTRM = np.linalg.inv(TRM)

        self.tune_response_matrix = TRM
        self.inverse_tune_response_matrix = iTRM
        return

    def trim_tune(self, dqx: float = 0, dqy: float = 0):
        SC = self._parent._parent
        if self.inverse_tune_response_matrix is None:
            logger.info('Did not find inverse tune response matrix. Building now.')
            self.build_tune_response_matrix()

        dk1, dk2 = np.dot(self.inverse_tune_response_matrix, [dqx, dqy])
        ref_data1 = SC.magnet_settings.get_many(self.tune_quad_controls_1)
        ref_data2 = SC.magnet_settings.get_many(self.tune_quad_controls_2)
        data1 = {key: ref_data1[key] + dk1 for key in ref_data1.keys()}
        data2 = {key: ref_data2[key] + dk2 for key in ref_data2.keys()}

        SC.magnet_settings.set_many(data1)
        SC.magnet_settings.set_many(data2)
        return

    def measure_with_kick(self, kick_px=10e-6, kick_py=10e-6, n_turns=100):
        SC = self._parent._parent
        dqtol = 0.01
        import nafflib 
        x, y = SC.bpm_system.capture_kick(n_turns=n_turns, kick_px=kick_px, kick_py=kick_py)
        amp_x, harm_x = nafflib.harmonics(x[0,:] - np.mean(x[0,:]), num_harmonics=2)
        amp_y, harm_y = nafflib.harmonics(y[0,:] - np.mean(y[0,:]), num_harmonics=2)
        qx = harm_x[0]
        qy = None
        for hy in harm_y:
            if abs(hy - qx) < dqtol:
                continue
            qy = hy
        return float(qx), float(qy)

    def correct(self, target_qx: Optional[float] = None, target_qy: Optional[float] = None,
                kick_px: float = 10e-6, kick_py: float = 10e-6, n_turns: int = 100, n_iter: int = 1,
                gain: float = 1, measurement_method: str = 'kick'):
        if measurement_method not in ['kick']:
            raise NotImplementedError(f'{measurement_method=} not implemented yet.')

        if target_qx is None:
            target_qx = self.design_qx

        if target_qy is None:
            target_qy = self.design_qy

        for _ in range(n_iter):
            qx, qy = self.measure_with_kick(kick_px, kick_py, n_turns=n_turns)
            if qx is None or qy is None:
                logger.info("Tune measurement failed, skipping correction.")
                return
            logger.info(f"Measured tune: qx={qx:.4f}, qy={qy:.4f}")
            dqx = qx - target_qx
            dqy = qy - target_qy
            logger.info(f"Delta tune: delta_q1={dqx:.4f}, delta_q2={dqy:.4f}")
            self.trim_tune(dqx=-gain*dqx, dqy=-gain*dqy)
        return