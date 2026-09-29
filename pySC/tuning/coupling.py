from typing import Optional, Union, Tuple, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging
import warnings
import scipy.optimize
import json
from ..core.control import KnobControl, KnobData
from ..core.types import NPARRAY, QuadrupoleResponse, SkewQuadrupoleResponse
from ..apps.response_matrix import ResponseMatrix
from ..apps.measurements import measure_dispersion
from .pySC_interface import pySCOrbitInterface
from ..utils.sc_tools import nanmean, nanstd
from ..utils import rdt

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)
TWOPI = 2*np.pi

class Coupling_Tuning(BaseModel, extra="forbid"):
    skew_quadrupoles: list[str] = []
    skew_quad_weights: Optional[list[float]] = None
    dispersion_weight: float = 1
    _skew_quadrupole_response: Optional[SkewQuadrupoleResponse] = PrivateAttr(default=None)

    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def build_skew_quadrupole_response(self, delta: float = 1e-6, save_as: Optional[str] = None):
        if len(self.skew_quadrupoles) == 0:
            raise Exception("list of tuning.optics.skew_quadrupoles is empty.")

        logger.info(f"Building optics responses of {len(self.skew_quadrupoles)} skew quadrupoles.")
        SC = self._parent._parent._parent

        bpm_indices = SC.bpm_system.indices
        twiss0 = SC.lattice.twiss
        integrated_strengths0 = rdt.get_integrated_strengths_with_feeddown(SC, use_design=True)
        dy0 = twiss0["dy"][bpm_indices]
        c_minus_0 = rdt.calculate_c_minus(SC, use_design=True)
        f1001_0 = rdt.fjklm(SC, 1, 0, 0, 1, use_design=True, twiss=twiss0, integrated_strengths=integrated_strengths0)
        f1010_0 = rdt.fjklm(SC, 1, 0, 1, 0, use_design=True, twiss=twiss0, integrated_strengths=integrated_strengths0)

        N = len(self.skew_quadrupoles)
        M = len(SC.bpm_system.names)
        dy_response = np.zeros([N, M])
        f1001_real_response = np.zeros([N, M])
        f1001_imag_response = np.zeros([N, M])
        f1010_real_response = np.zeros([N, M])
        f1010_imag_response = np.zeros([N, M])
        c_minus_real_response = np.zeros([N, 1])
        c_minus_imag_response = np.zeros([N, 1])

        for ii, skew_quad in enumerate(self.skew_quadrupoles):
            logger.info(f"Calculating response of {skew_quad} ({ii}/{N}).")
            k1 = SC.design_magnet_settings.get(skew_quad)
            try:
                SC.design_magnet_settings.set(skew_quad, k1 + delta)
                twiss = SC.lattice.get_twiss(use_design=True)
                integrated_strengths = rdt.get_integrated_strengths_with_feeddown(SC, use_design=True)

                f1001 = rdt.fjklm(SC, 1, 0, 0, 1, use_design=True, twiss=twiss, integrated_strengths=integrated_strengths)
                f1010 = rdt.fjklm(SC, 1, 0, 1, 0, use_design=True, twiss=twiss, integrated_strengths=integrated_strengths)
                c_minus = rdt.calculate_c_minus(SC, use_design=True)
            finally:
                SC.design_magnet_settings.set(skew_quad, k1)

            dy_response[ii] = (twiss["dy"][bpm_indices] - dy0) / delta
            f1001_real_response[ii] = (f1001.real - f1001_0.real)[bpm_indices] / delta
            f1001_imag_response[ii] = (f1001.imag - f1001_0.imag)[bpm_indices] / delta
            f1010_real_response[ii] = (f1010.real - f1010_0.real)[bpm_indices] / delta
            f1010_imag_response[ii] = (f1010.imag - f1010_0.imag)[bpm_indices] / delta
            c_minus_real_response[ii] = (c_minus.real - c_minus_0.real) / delta
            c_minus_imag_response[ii] = (c_minus.imag - c_minus_0.imag) / delta

        response = SkewQuadrupoleResponse(skew_quadrupoles=self.skew_quadrupoles,
                                          dy_response=dy_response,
                                          f1001_real_response=f1001_real_response,
                                          f1001_imag_response=f1001_imag_response,
                                          f1010_real_response=f1010_real_response,
                                          f1010_imag_response=f1010_imag_response,
                                          c_minus_real_response=c_minus_real_response,
                                          c_minus_imag_response=c_minus_imag_response,
                                          )
        self._skew_quadrupole_response = response
        if save_as is not None:
            logger.info(f"Saving skew quadrupole response in {save_as}.")
            response.save_as(filename=save_as)
        return

    def load_skew_quadrupole_response(self, filename: Optional[str] = None):
        if filename is None:
            filename = self._parent._parent.RM_folder + '/skew_quadrupole_responses.json'
        logger.info(f"Loading skew quadrupole responses: {filename}.")
        self._skew_quadrupole_response = SkewQuadrupoleResponse.load(filename=filename)
        return

    def coupling_rdts_cheat(self, use_design: bool = False):
        SC = self._parent._parent._parent
        twiss = SC.lattice.get_twiss(use_design=use_design)
        integrated_strengths = rdt.get_integrated_strengths_with_feeddown(SC, use_design=use_design)

        f1001 = rdt.fjklm(SC, 1, 0, 0, 1, use_design=use_design, twiss=twiss, integrated_strengths=integrated_strengths)[SC.bpm_system.indices]
        f1010 = rdt.fjklm(SC, 1, 0, 1, 0, use_design=use_design, twiss=twiss, integrated_strengths=integrated_strengths)[SC.bpm_system.indices]

        f1001_real_err = np.ones_like(f1001.real, dtype=float)
        f1001_imag_err = np.ones_like(f1001.imag, dtype=float)
        f1010_real_err = np.ones_like(f1010.real, dtype=float)
        f1010_imag_err = np.ones_like(f1010.imag, dtype=float)

        return f1001.real, f1001.imag, f1010.real, f1010.imag, f1001_real_err, f1001_imag_err, f1010_real_err, f1010_imag_err

    def assemble_response_matrix(self, observables: list[str]):
        SC = self._parent._parent._parent
        nbpm = len(SC.bpm_system.indices)
        nquads = len(self._skew_quadrupole_response.skew_quadrupoles)
        nobs = len(observables)

        matrix = np.zeros([nobs*nbpm, nquads])
        for ii, obs in enumerate(observables):
            matrix[ii * nbpm:(ii + 1) * nbpm, :] = np.transpose(getattr(self._skew_quadrupole_response, f"{obs}_response"))


        input_weights = np.array(self.skew_quad_weights) if self.skew_quad_weights is not None else None
        RM = ResponseMatrix(matrix=matrix, input_weights=input_weights, input_names=self.skew_quadrupoles)

        if 'dy' in observables and len(observables) > 1:
            if observables[-1] != 'dy':
                raise Exception(f"Dispersion not in the last place of observables list: {observables}")
            split = (len(observables) - 1) * nbpm
            coupling_norm = np.linalg.norm(matrix[:split])
            dispersion_norm = np.linalg.norm(matrix[split:])
            if coupling_norm == 0:
                raise Exception("Coupling response matrix has zero norm!")
            if dispersion_norm == 0:
                raise Exception("Dispersion response matrix has zero norm!")
            # set dispersion weight
            RM.output_weights[split:] = self.dispersion_weight * coupling_norm / dispersion_norm

        return RM

    def correct(self, measurements: dict[str,dict], gain: float = 1, correction_method: str = "svd_values",
                correction_parameter: Union[int, float] = 100):
        SC = self._parent._parent._parent
        coupling_measurements = [
            'coupling_rdts_cheat',
        ]
        active_coupling_measurements = [name for name in coupling_measurements if name in measurements]
        if len(active_coupling_measurements) > 1:
            raise ValueError(
                f"Only one optics measurement can be used per correction: {active_coupling_measurements}"
            )

        if len(measurements.keys()) and self._skew_quadrupole_response is None:
            self.load_skew_quadrupole_response()

        observables = []
        model = {}
        if len(active_coupling_measurements) > 0:
            coupling_measurement = active_coupling_measurements[0]
            if coupling_measurement in ['coupling_rdts_cheat']:
                observables.append('f1001_real')
                observables.append('f1001_imag')
                observables.append('f1010_real')
                observables.append('f1010_imag')

        if 'dispersion' in measurements:
            observables.append('dy')

        if not len(observables):
            raise Exception("No observables were defined for the measurement.")

        bpm_indices = SC.bpm_system.indices
        nbpm = len(bpm_indices)
        RM = self.assemble_response_matrix(observables=observables)

        model = {}
        estimations = {}
        errors = {}
        if 'coupling_rdts_cheat' in measurements:
            f1001r, f1001i, f1010r, f1010i, f1001re, f1001ie, f1010re, f1010ie = self.coupling_rdts_cheat()
            estimations['f1001_real'] = f1001r
            estimations['f1001_imag'] = f1001i
            estimations['f1010_real'] = f1010r
            estimations['f1010_imag'] = f1010i
            errors['f1001_real'] = f1001re
            errors['f1001_imag'] = f1001ie
            errors['f1010_real'] = f1010re
            errors['f1010_imag'] = f1010ie
            mf1001r, mf1001i, mf1010r, mf1010i, _, _, _, _ = self.coupling_rdts_cheat(use_design=True)
            model['f1001_real'] = mf1001r
            model['f1001_imag'] = mf1001i
            model['f1010_real'] = mf1010r
            model['f1010_imag'] = mf1010i

        if 'dispersion' in measurements:
            settings = measurements['dispersion']
            dx_est, dy_est, dx_err, dy_err = self._parent.dispersion(use_design=False, **settings)
            estimations['dy'] = dy_est
            errors['dy'] = dy_err
            model['dy'] = SC.lattice.twiss['dy'][bpm_indices]

        beating = np.zeros([len(observables) * nbpm])
        for ii, obs in enumerate(observables):
            beating[ii * nbpm:(ii + 1) * nbpm] = estimations[obs] - model[obs]

        trims = - RM.solve(beating, method=correction_method, parameter=correction_parameter)

        initial_k1 = np.array([SC.magnet_settings.get(skew_quad) for skew_quad in self.skew_quadrupoles])
        final_k1 = initial_k1 + gain * trims
        data = {quad: final_k1[ii] for ii,quad in enumerate(self.skew_quadrupoles)}
        SC.magnet_settings.set_many(data)
        return
