from typing import Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr, ConfigDict
import numpy as np
import logging
import warnings
import scipy.optimize
import json
from ..core.control import KnobControl, KnobData
from ..core.types import NPARRAY, Quadrupole_response
from ..apps.response_matrix import ResponseMatrix
from ..apps.measurements import measure_dispersion
from .pySC_interface import pySCOrbitInterface
from ..utils.sc_tools import nanmean, nanstd

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class Optics_tuning(BaseModel, extra="forbid"):
    quadrupoles: list[str] = None
    quad_weights: Optional[list[float]] = None
    response: Optional[Quadrupole_response] = None

    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_quadrupole_response(self, delta: float = 1e-6, save_as: Optional[str] = None):
        if self.quadrupoles is None:
            raise Exception("tuning.optics.quads have not been specified.")
        if self.quadrupoles is not None and len(self.quadrupoles) == 0:
            raise Exception("list of tuning.optics.quads is empty.")

        logger.info(f"Building optics responses of {len(self.quadrupoles)} quadrupoles.")
        SC = self._parent._parent

        bpm_indices = SC.bpm_system.indices
        twiss0 = SC.lattice.twiss
        betx0 = twiss0["betx"][bpm_indices]
        bety0 = twiss0["bety"][bpm_indices]
        dx0 = twiss0["dx"][bpm_indices]
        mux0 = twiss0["mux"][bpm_indices]
        muy0 = twiss0["muy"][bpm_indices]
        qx0 = twiss0["qx"]
        qy0 = twiss0["qy"]

        N = len(self.quadrupoles)
        M = len(SC.bpm_system.names)
        betx_response = np.zeros([N, M])
        bety_response = np.zeros([N, M])
        dx_response = np.zeros([N, M])
        eta_response = np.zeros([N, M])
        mux_response = np.zeros([N, M])
        muy_response = np.zeros([N, M])
        qx_response = np.zeros([N, 1])
        qy_response = np.zeros([N, 1])

        for ii, quad in enumerate(self.quadrupoles):
            logger.info(f"Calculating response of {quad} ({ii}/{N}).")
            k1 = SC.design_magnet_settings.get(quad)
            SC.design_magnet_settings.set(quad, k1 + delta)
            twiss = SC.lattice.get_twiss(use_design=True)
            SC.design_magnet_settings.set(quad, k1)

            betx_response[ii] = (twiss["betx"][bpm_indices] - betx0) / delta
            bety_response[ii] = (twiss["bety"][bpm_indices] - bety0) / delta
            dx_response[ii] = (twiss["dx"][bpm_indices] - dx0) / delta
            eta_response[ii] = (twiss["dx"][bpm_indices]/np.sqrt(twiss["betx"][bpm_indices]) - dx0/np.sqrt(betx0)) / delta
            mux_response[ii] = (twiss["mux"][bpm_indices] - mux0) / delta
            muy_response[ii] = (twiss["muy"][bpm_indices] - muy0) / delta
            qx_response[ii] = (twiss["qx"] - qx0) / delta
            qy_response[ii] = (twiss["qy"] - qy0) / delta

        response = Quadrupole_response(quadrupoles=self.quadrupoles,
                                       betx_response=betx_response, bety_response=bety_response,
                                       mux_response=mux_response, muy_response=muy_response,
                                       dx_response=dx_response, eta_response=eta_response,
                                       qx_response=qx_response, qy_response=qy_response)
        self.response = response
        if save_as is not None:
            logger.info(f"Saving quadrupole response in {save_as}.")
            response.save_as(filename=save_as)
        return

    def load_quadrupole_response(self, filename: Optional[str] = None):
        if filename is None:
            filename = self._parent.RM_folder + '/quadrupole_responses.json'
        logger.info(f"Loading quadrupole responses: {filename}.")
        self.response = Quadrupole_response.load(filename=filename)
        return

    def beta_from_amplitude(self, n_kicks: int = 1, n_turns: int = 50, 
                            qx_low: float = 0.15, qx_high: float = 0.22,
                            qy_low: float = 0.23, qy_high: float = 0.30,
                            kick_px: float = 1e-6, kick_py: float = 1e-6,
                            use_design: bool = False, fft_n=10000):
        SC = self._parent._parent
        import nafflib
        N = len(SC.bpm_system.names)

        betx_model = SC.lattice.twiss["betx"][SC.bpm_system.indices]
        bety_model = SC.lattice.twiss["bety"][SC.bpm_system.indices]

        betx_est = np.zeros([N, n_kicks])
        bety_est = np.zeros([N, n_kicks])
        freqs = np.fft.fftfreq(fft_n)
        pmask = freqs > 0

        for iter in range(n_kicks):
            logger.info(f"Capturing {iter+1} out {n_kicks} kicks.")
            # twiss = SC.lattice.get_twiss()
            # SC.injection.delta = twiss['delta'][0]
            # SC.injection.tau = twiss['tau'][0]
            # SC.injection.x = twiss['x'][0]
            # SC.injection.y = twiss['y'][0]
            # SC.injection.px = twiss['px'][0]+kick_px
            # SC.injection.py = twiss['py'][0]+kick_py
            # x_tbt, y_tbt = SC.bpm_system.capture_injection(n_turns=n_turns, use_design=use_design)
            x_tbt, y_tbt = SC.bpm_system.capture_kick(n_turns=n_turns, kick_px=kick_px,
                                                      kick_py=kick_py, use_design=use_design)
            # SC.injection.px = 0
            # SC.injection.py = 0

            qx_rejections = 0
            qy_rejections = 0

            Ax_bpm = np.zeros(N)
            qx_bpm = np.zeros(N)
            for ii in range(N):
                fftx = np.fft.fft(x_tbt[ii] - np.mean(x_tbt[ii]), n=fft_n)
                ix = np.argmax(np.abs(fftx[pmask]))
                amp = fftx[pmask][ix]
                qx = freqs[pmask][ix]
                # amps, freqs = nafflib.harmonics(x_tbt[ii] - np.mean(x_tbt[ii]), window_order=0)
                # amp = amps[0]
                # qx = freqs[0]
                # if qx < 0:
                #     qx = - qx
                #     amp = np.conj(amp)
                if qx < qx_low or qx > qx_high:
                    Ax_bpm[ii] = np.nan
                    qx_bpm[ii] = np.nan
                    qx_rejections += 1
                    #print(qx_low, qx, qx_high)
                else:
                    Ax_bpm[ii] = abs(amp)
                    qx_bpm[ii] = qx

            Ay_bpm = np.zeros(N)
            qy_bpm = np.zeros(N)
            for ii in range(N):
                ffty = np.fft.fft(y_tbt[ii] - np.mean(y_tbt[ii]), n=fft_n)
                iy = np.argmax(np.abs(ffty[pmask]))
                amp = ffty[pmask][iy]
                qy = freqs[pmask][iy]
                # amps, freqs = nafflib.harmonics(y_tbt[ii] - np.mean(y_tbt[ii]), window_order=0)
                # amp = amps[0]
                # qy = freqs[0]
                # if qy < 0:
                #     qy = - qy
                #     amp = np.conj(amp)
                if qy < qy_low or qy > qy_high:
                    Ay_bpm[ii] = np.nan
                    qy_bpm[ii] = np.nan
                    qy_rejections += 1
                    #print(qy_low, qy, qy_high)
                else:
                    Ay_bpm[ii] = abs(amp)
                    qy_bpm[ii] = qy

            logger.info(f"Average tunes Qx={nanmean(qx_bpm):.4f}, Qy={nanmean(qy_bpm):.4f}")
            logger.info(f"Rejected {qx_rejections} on Qx and {qy_rejections} on Qy")
            action_x_estimate = nanmean(Ax_bpm**2/betx_model) / 2.
            betx_est[:,iter] = Ax_bpm**2 / action_x_estimate / 2.

            action_y_estimate = nanmean(Ay_bpm**2/bety_model) / 2.
            bety_est[:,iter] = Ay_bpm**2 / action_y_estimate / 2.

        betx_est_mean = nanmean(betx_est, axis=1)
        betx_est_error = nanstd(betx_est, axis=1)/np.sqrt(n_kicks)

        bety_est_mean = nanmean(bety_est, axis=1)
        bety_est_error = nanstd(bety_est, axis=1)/np.sqrt(n_kicks)

        beta_beat_x = nanstd(1 - betx_est_mean/SC.lattice.twiss['betx'][SC.bpm_system.indices])
        beta_beat_y = nanstd(1 - bety_est_mean/SC.lattice.twiss['bety'][SC.bpm_system.indices])
        logger.info("Beta from amplitude measurement:")
        logger.info(f"  Estimated horizontal beta-beating: {beta_beat_x * 100:.2f}%")
        logger.info(f"  Estimated vertical beta-beating: {beta_beat_y * 100:.2f}%")

        return betx_est_mean, bety_est_mean, betx_est_error, bety_est_error

    def dispersion(self, delta_frf: float = 20, shots_per_orbit: int = 1, use_design: bool = False):
        SC = self._parent._parent

        interface = pySCOrbitInterface(SC=SC)
        interface.use_design = use_design
        generator = measure_dispersion(interface=interface,
                                       delta=delta_frf,
                                       shots_per_orbit=shots_per_orbit,
                                       bipolar=True,
                                       skip_save=True)

        for code, measurement in generator:
            pass

        data = measurement.dispersion_data
        data.frequency_response_x

        alpha_c = SC.lattice.get_momentum_compaction(use_design=True)
        factor = - alpha_c * SC.rf_settings.main.frequency

        dx = factor * data.frequency_response_x
        dx_err = factor * data.frequency_response_x_err
        dy = factor * data.frequency_response_x
        dy_err = factor * data.frequency_response_x_err

        dx_beat = np.std(dx - SC.lattice.twiss['dx'][SC.bpm_system.indices])
        dy_beat = np.std(dy - SC.lattice.twiss['dy'][SC.bpm_system.indices])
        logger.info("Dispersion measurement:")
        logger.info(f"  Estimated horizontal dispersion-beating: {dx_beat * 1000:.2f} mm")
        logger.info(f"  Estimated vertical dispersion-beating: {dy_beat * 1000:.2f} mm")
        return dx, dy, dx_err, dy_err

    def assemble_response_matrix(self, observables: list[str]):
        SC = self._parent._parent
        nbpm = len(SC.bpm_system.indices)
        nquads = len(self.response.quadrupoles)
        nobs = len(observables)

        matrix = np.zeros([nobs*nbpm, nquads])
        for ii, obs in enumerate(observables):
            matrix[ii * nbpm:(ii + 1) * nbpm, :] = np.transpose(getattr(self.response, f"{obs}_response"))

        input_weights = np.array(self.quad_weights) if self.quad_weights is not None else None
        RM = ResponseMatrix(matrix=matrix, input_weights=input_weights, input_names=self.quadrupoles)

        #RM.output_weights[2*N:] = np.mean(np.std(matrix[:,:2*N], axis=0)) / np.mean(np.std(matrix[:,2*N:], axis=0))
        return RM

    def correct(self, measurements: dict[str,dict], gain: float = 1, correction_method: str = "svd_values",
                correction_parameter: Union[int, float] = 100):
        SC = self._parent._parent
        observables = []
        if 'beta_from_amplitude' in measurements:
            observables.append('betx')
            observables.append('bety')
        if 'dispersion' in measurements:
            observables.append('dx')

        bpm_indices = SC.bpm_system.indices
        nbpm = len(bpm_indices)
        model = {obs: SC.lattice.twiss[obs][bpm_indices] for obs in observables }

        RM = self.assemble_response_matrix(observables=observables)

        estimations = {}
        errors = {}
        if 'beta_from_amplitude' in measurements:
            settings = measurements['beta_from_amplitude']
            betx_est, bety_est, betx_err, bety_err = self.beta_from_amplitude(use_design=False, **settings)
            #    n_kicks=1, n_turns=50, qx_low=0.15, qx_high=0.22, qy_low=0.23, qy_high=0.30,
            #    kick_px=1e-6, kick_py=1e-6, fft_n=10000)
            estimations['betx'] = betx_est
            estimations['bety'] = bety_est
            errors['betx'] = betx_err
            errors['bety'] = bety_err
        if 'dispersion' in measurements:
            settings = measurements['dispersion']
            dx_est, dy_est, dx_err, dy_err = self.dispersion(use_design=False, **settings)
            estimations['dx'] = dx_est
            errors['dx'] = dx_err

        beating = np.zeros([len(observables) * nbpm])
        for ii, obs in enumerate(observables):
            beating[ii * nbpm:(ii + 1) * nbpm] = estimations[obs] - model[obs]

        trims = - RM.solve(beating, method=correction_method, parameter=correction_parameter)

        initial_k1 = np.array([SC.magnet_settings.get(quad) for quad in self.quadrupoles])
        final_k1 = initial_k1 + gain * trims
        data = {quad: final_k1[ii] for ii,quad in enumerate(self.quadrupoles)}
        SC.magnet_settings.set_many(data)
        return
