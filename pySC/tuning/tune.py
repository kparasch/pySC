from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging
import scipy.optimize

from ..core.control import KnobControl, KnobData
from ..core.types import NPARRAY
from ..apps.response_matrix import ResponseMatrix

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class Tune(BaseModel, extra="forbid"):
    knob_qx: str = 'qx_trim'
    knob_qy: str = 'qy_trim'
    controls_1: list[str] = []
    controls_2: list[str] = []
    tune_response_matrix: Optional[NPARRAY] = None
    inverse_tune_response_matrix: Optional[NPARRAY] = None
    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def controls(self) -> list[str]:
        return self.controls_1 + self.controls_2

    @property
    def design_qx(self):
        return np.mod(self._parent._parent.lattice.twiss['qx'], 1)

    @property
    def integer_qx(self):
        return np.floor(self._parent._parent.lattice.twiss['qx'])

    @property
    def design_qy(self):
        return np.mod(self._parent._parent.lattice.twiss['qy'], 1)

    @property
    def integer_qy(self):
        return np.floor(self._parent._parent.lattice.twiss['qy'])

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
        if not len(self.controls_1) > 0:
            raise Exception('tune_quad_controls_1 is empty. Please set.')
        if not len(self.controls_2) > 0:
            raise Exception('tune_quad_controls_2 is empty. Please set.')

        TRM = np.zeros((2,2))
        TRM[:, 0] = self.tune_response(self.controls_1, dk=dk)
        TRM[:, 1] = self.tune_response(self.controls_2, dk=dk)
        return TRM

    def create_tune_knobs(self, delta: float = 1e-5) -> None:
        if not len(self.controls) > 0:
            raise Exception('tune.controls_1/tune.controls_2 are empty. Please set.')

        matrix = self.build_tune_response_matrix(dk=delta)

        tune_response_matrix = ResponseMatrix(matrix=matrix)
        inverse_matrix = tune_response_matrix.build_pseudoinverse().matrix

        dk1_qx, dk2_qx = inverse_matrix[:,0]
        dk1_qy, dk2_qy = inverse_matrix[:,1]

        n1 = len(self.controls_1)
        n2 = len(self.controls_2)
        qx_weights = [float(dk1_qx)] * n1  + [float(dk2_qx)] * n2
        qy_weights = [float(dk1_qy)] * n1  + [float(dk2_qy)] * n2

        knob_data = KnobData(data={
            self.knob_qx: KnobControl(control_names=self.controls, weights=qx_weights),
            self.knob_qy: KnobControl(control_names=self.controls, weights=qy_weights)
            })

        logger.info(f"{self.knob_qx}: sum(|weights|)={np.sum(np.abs(qx_weights)):.2e}")
        logger.info(f"{self.knob_qy}: sum(|weights|)={np.sum(np.abs(qy_weights)):.2e}")
        return knob_data

    def trim_tune(self, dqx: float = 0, dqy: float = 0, use_design: bool = False) -> None:
        logger.warning('Deprecation: please use .trim instead of .trim_tune.')
        return self.trim(dqx=dqx, dqy=dqy, use_design=use_design)

    def trim(self, dqx: float = 0, dqy: float = 0, use_design: bool = False) -> None:
        SC = self._parent._parent

        if use_design:
            assert self.knob_qx in SC.design_magnet_settings.controls.keys()
            assert self.knob_qy in SC.design_magnet_settings.controls.keys()
        else:
            assert self.knob_qx in SC.magnet_settings.controls.keys()
            assert self.knob_qy in SC.magnet_settings.controls.keys()

        dqx0 = SC.magnet_settings.get(self.knob_qx, use_design=use_design)
        dqy0 = SC.magnet_settings.get(self.knob_qy, use_design=use_design)

        SC.magnet_settings.set(self.knob_qx, dqx0 + dqx, use_design=use_design)
        SC.magnet_settings.set(self.knob_qy, dqy0 + dqy, use_design=use_design)

        return

    def measure_with_kick(self, kick_px=10e-6, kick_py=10e-6, n_turns=100):
        SC = self._parent._parent
        dqtol = 0.01
        import nafflib 
        x, y = SC.bpm_system.capture_kick(n_turns=n_turns, kick_px=kick_px, kick_py=kick_py)
        amp_x, harm_x = nafflib.harmonics(x[0,:] - np.mean(x[0,:]), num_harmonics=2)
        amp_y, harm_y = nafflib.harmonics(y[0,:] - np.mean(y[0,:]), num_harmonics=2)
        qx = abs(harm_x[0])
        qy = None
        for hy in harm_y:
            if abs(abs(hy) - qx) < dqtol:
                continue
            qy = abs(hy)
        return float(qx), float(qy)

    def correct(self, target_qx: Optional[float] = None, target_qy: Optional[float] = None,
                kick_px: float = 10e-6, kick_py: float = 10e-6, n_turns: int = 100, n_iter: int = 1,
                gain: float = 1, measurement_method: str = 'kick'):
        '''
        Correct the tune to the target values.
        Parameters
        ----------
        target_qx : float, optional
            Target horizontal tune. If None, use design tune.
        target_qy : float, optional
            Target vertical tune. If None, use design tune.
        kick_px : float, optional
            Kick in x for tune measurement. Default is 10e-6.
            Relevant only for 'kick', 'first_turn' and 'orbit' methods.
        kick_py : float, optional
            Kick in y for tune measurement. Default is 10e-6.
            Relevant only for 'kick' method.
        n_turns : int, optional
            Number of turns to track for tune measurement. Default is 100.
            Relevant only for 'kick' method.
        n_iter : int, optional
            Number of correction iterations. Default is 1.
        gain : float, optional
            Gain for the correction. Default is 1.
        measurement_method : str, optional
            Method to measure the tune. Options are 'kick', 'first_turn', 'orbit',
             'cheat', 'cheat4d', 'cheat_with_integer'.
            Default is 'kick'.
        '''

        if measurement_method not in ['kick', 'first_turn', 'orbit', 'cheat', 'cheat4d', 'cheat_with_integer']:
            raise NotImplementedError(f'{measurement_method=} not implemented yet.')

        if target_qx is None:
            target_qx = self.design_qx
            if measurement_method == 'cheat_with_integer':
                target_qx += self.integer_qx

        if target_qy is None:
            target_qy = self.design_qy
            if measurement_method == 'cheat_with_integer':
                target_qy += self.integer_qy

        for _ in range(n_iter):
            if measurement_method == 'kick':
                qx, qy = self.measure_with_kick(kick_px, kick_py, n_turns=n_turns)
            elif measurement_method == 'first_turn':
                qx, qy = self.estimate_from_first_turn(dk0=kick_px)
            elif measurement_method == 'orbit':
                qx, qy = self.estimate_from_orbit(dk0=kick_px)
            elif measurement_method == 'cheat':
                qx, qy = self.cheat()
            elif measurement_method == 'cheat4d':
                qx, qy = self.cheat4d()
            elif measurement_method == 'cheat_with_integer':
                qx, qy = self.cheat_with_integer()
            else:
                raise Exception(f'Unknown measurement_method {measurement_method}')
            if qx is None or qy is None or qx != qx or qy != qy:
                logger.info("Tune measurement failed, skipping correction.")
                return
            logger.info(f"Measured tune: qx={qx:.4f}, qy={qy:.4f}")
            dqx = qx - target_qx
            dqy = qy - target_qy
            logger.info(f"Delta tune: delta_q1={dqx:.4f}, delta_q2={dqy:.4f}")
            self.trim(dqx=-gain*dqx, dqy=-gain*dqy)
        return

    def get_design_corrector_response_injection(self, corr: str, dk0: float = 1e-6):
        SC = self._parent._parent
        k0 = SC.magnet_settings.get(corr, use_design=True)
        SC.magnet_settings.set(corr, k0 + dk0, use_design=True)
        bunch = SC.injection.generate_bunch(use_design=True)
        x1, y1 = SC.lattice.track(bunch.copy(), indices=SC.bpm_system.indices, use_design=True)
        SC.magnet_settings.set(corr, k0 - dk0, use_design=True)
        x2, y2 = SC.lattice.track(bunch.copy(), indices=SC.bpm_system.indices, use_design=True)
        SC.magnet_settings.set(corr, k0, use_design=True)
        dx = (x1[0,:,0] - x2[0,:,0]) / (2*dk0)
        dy = (y1[0,:,0] - y2[0,:,0]) / (2*dk0)
        return dx, dy


    def estimate_from_first_turn(self, dk0: float = 1e-4):
        SC = self._parent._parent
        hcorr = SC.tuning.HCORR[0]
        vcorr = SC.tuning.VCORR[0]

        hcorr_k0 = SC.magnet_settings.get(hcorr)
        vcorr_k0 = SC.magnet_settings.get(vcorr)

        def get_average_xy(SC, n=10):
            N = len(SC.bpm_system.names)
            x_avg = np.zeros(N)
            y_avg = np.zeros(N)
            for _ in range(n):
                x, y = SC.bpm_system.capture_injection(n_turns=1)
                x_avg += x[:, 0]
                y_avg += y[:, 0]
            return x_avg / n, y_avg / n

        # measure responses
        x0, y0 = get_average_xy(SC, n=5)
        #x0, y0 = SC.bpm_system.capture_injection()

        SC.magnet_settings.set(hcorr, hcorr_k0 + dk0)
        x1, _ = get_average_xy(SC, n=5)
        #x1, _ = SC.bpm_system.capture_injection()
        SC.magnet_settings.set(hcorr, hcorr_k0)

        SC.magnet_settings.set(vcorr, vcorr_k0 + dk0)
        #_, y1 = SC.bpm_system.capture_injection()
        _, y1 = get_average_xy(SC, n=5)
        SC.magnet_settings.set(vcorr, vcorr_k0)

        dx0 = (x1 - x0) / (dk0)
        dy0 = (y1 - y0) / (dk0)
        #####

        ### do fit based on knobs

        def x_chi2(delta):
            SC.tuning.tune.trim(delta, 0, use_design=True)
            dx_ideal, _ = self.get_design_corrector_response_injection(hcorr)
            SC.tuning.tune.trim(-delta, 0, use_design=True)
            return np.sum((dx0 - dx_ideal)**2)

        def y_chi2(delta):
            SC.tuning.tune.trim(0, delta, use_design=True)
            _, dy_ideal = self.get_design_corrector_response_injection(vcorr)
            SC.tuning.tune.trim(0, -delta, use_design=True)
            return np.sum((dy0 - dy_ideal)**2)

        x_res = scipy.optimize.minimize_scalar(x_chi2, (-0.1, 0.1), method='Brent')
        delta_x = x_res.x
        est_qx = self.design_qx + delta_x
        logger.info(f"Estimated horizontal tune: Qx = {est_qx:.3f} (Δ = {delta_x:.3f})")

        y_res = scipy.optimize.minimize_scalar(y_chi2, (-0.1, 0.1), method='Brent')
        delta_y = y_res.x
        est_qy = self.design_qy + delta_y
        logger.info(f"Estimated vertical tune: Qy = {est_qy:.3f} (Δ = {delta_y:.3f})")

        return est_qx, est_qy

    def get_design_corrector_response_orbit(self, corr: str, dk0: float = 1e-6):
        SC = self._parent._parent
        k0 = SC.magnet_settings.get(corr, use_design=True)
        SC.magnet_settings.set(corr, k0 + dk0, use_design=True)
        x1, y1 = SC.lattice.get_orbit(indices=SC.bpm_system.indices, use_design=True)
        SC.magnet_settings.set(corr, k0 - dk0, use_design=True)
        x2, y2 = SC.lattice.get_orbit(indices=SC.bpm_system.indices, use_design=True)
        SC.magnet_settings.set(corr, k0, use_design=True)
        dx = (x1 - x2) / (2*dk0)
        dy = (y1 - y2) / (2*dk0)
        return dx, dy


    def estimate_from_orbit(self, dk0: float = 1e-5):
        SC = self._parent._parent
        hcorr = SC.tuning.HCORR[0]
        vcorr = SC.tuning.VCORR[0]

        hcorr_k0 = SC.magnet_settings.get(hcorr)
        vcorr_k0 = SC.magnet_settings.get(vcorr)

        def get_average_xy(SC, n=10):
            N = len(SC.bpm_system.names)
            x_avg = np.zeros(N)
            y_avg = np.zeros(N)
            for _ in range(n):
                x, y = SC.bpm_system.capture_orbit()
                x_avg += x
                y_avg += y
            return x_avg / n, y_avg / n

        # measure responses
        #x0, y0 = get_average_xy(SC, n=5)
        x0, y0 = SC.bpm_system.capture_orbit()

        SC.magnet_settings.set(hcorr, hcorr_k0 + dk0)
        x1, _ = get_average_xy(SC, n=5)
        #x1, _ = SC.bpm_system.capture_orbit()
        SC.magnet_settings.set(hcorr, hcorr_k0 - dk0)
        x0, _ = get_average_xy(SC, n=5)
        #x0, _ = SC.bpm_system.capture_orbit()
        SC.magnet_settings.set(hcorr, hcorr_k0)

        SC.magnet_settings.set(vcorr, vcorr_k0 + dk0)
        _, y1 = get_average_xy(SC, n=5)
        #_, y1 = SC.bpm_system.capture_orbit()
        SC.magnet_settings.set(vcorr, vcorr_k0 - dk0)
        _, y0 = get_average_xy(SC, n=5)
        #_, y0 = SC.bpm_system.capture_orbit()
        SC.magnet_settings.set(vcorr, vcorr_k0)

        dx0 = (x1 - x0) / (2 * dk0)
        dy0 = (y1 - y0) / (2 * dk0)
        #####

        ### do fit based on knobs

        def x_chi2(delta):
            SC.tuning.tune.trim(delta, 0, use_design=True)
            dx_ideal, _ = self.get_design_corrector_response_orbit(hcorr, dk0=dk0)
            SC.tuning.tune.trim(-delta, 0, use_design=True)
            return np.sum((dx0 - dx_ideal)**2)

        def y_chi2(delta):
            SC.tuning.tune.trim(0, delta, use_design=True)
            _, dy_ideal = self.get_design_corrector_response_orbit(vcorr, dk0=dk0)
            SC.tuning.tune.trim(0, -delta, use_design=True)
            return np.sum((dy0 - dy_ideal)**2)

        x_res = scipy.optimize.minimize_scalar(x_chi2, (-0.1, 0.1), method='Brent')
        delta_x = x_res.x
        est_qx = self.design_qx + delta_x
        logger.info(f"Estimated horizontal tune: Qx = {est_qx:.3f} (Δ = {delta_x:.3f})")

        y_res = scipy.optimize.minimize_scalar(y_chi2, (-0.1, 0.1), method='Brent')
        delta_y = y_res.x
        est_qy = self.design_qy + delta_y
        logger.info(f"Estimated vertical tune: Qy = {est_qy:.3f} (Δ = {delta_y:.3f})")

        return est_qx, est_qy

    def cheat4d(self) -> tuple[float, float]:
        SC = self._parent._parent
        qx, qy = SC.lattice.get_tune(method='4d')
        delta_x = qx - self.design_qx
        delta_y = qy - self.design_qy
        logger.info(f"Horizontal tune: Qx = {qx:.3f} (Δ = {delta_x:.3f})")
        logger.info(f"Vertical tune: Qy = {qy:.3f} (Δ = {delta_y:.3f})")

        return qx, qy

    def cheat(self) -> tuple[float, float]:
        SC = self._parent._parent
        qx, qy = SC.lattice.get_tune(method='6d')
        delta_x = qx - self.design_qx
        delta_y = qy - self.design_qy
        logger.info(f"Horizontal tune: Qx = {qx:.3f} (Δ = {delta_x:.3f})")
        logger.info(f"Vertical tune: Qy = {qy:.3f} (Δ = {delta_y:.3f})")

        return qx, qy

    def cheat_with_integer(self) -> tuple[float, float]:
        SC = self._parent._parent
        twiss = SC.lattice.get_twiss()
        qx = twiss['qx']
        qy = twiss['qy']
        delta_x = qx - self.design_qx - self.integer_qx
        delta_y = qy - self.design_qy - self.integer_qy
        logger.info(f"Horizontal tune: Qx = {qx:.3f} (Δ = {delta_x:.3f})")
        logger.info(f"Vertical tune: Qy = {qy:.3f} (Δ = {delta_y:.3f})")

        return qx, qy