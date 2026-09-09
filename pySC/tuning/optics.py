from typing import Optional, Union, Tuple, TYPE_CHECKING
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
TWOPI = 2*np.pi

def circular_mean_std(phase: np.ndarray, axis: int = 0):
    mean_vector = nanmean(np.exp(1j * phase), axis=axis)
    circmean = np.angle(mean_vector)
    R = np.abs(mean_vector)
    circstd = np.sqrt(-2 * np.log(R))
    return circmean, circstd

def circular_diff(a, b):
    """Signed shortest angular distance from b to a, in [-π, π]."""
    return (a - b + np.pi) % TWOPI - np.pi

def align_pi_periodic_to_model(phase, model_phase):
    # phase and model_phase in radians.
    # phase is known modulo pi because arctan(tan()) folded it.
    offset = 0.5 * np.angle(nanmean(np.exp(2j * (phase - model_phase))))
    reference = model_phase + offset
    return reference + 0.5 * circular_diff(2 * phase, 2 * reference)

def dft(signal, frequency):
     return np.sum(signal*np.exp(-2.j*np.pi*np.arange(len(signal))*frequency))

class Optics_tuning(BaseModel, extra="forbid"):
    quadrupoles: list[str] = []
    quad_weights: Optional[list[float]] = None
    horizontal_kick: Optional[str] = None
    vertical_kick: Optional[str] = None
    horizontal_ac: Optional[str] = None
    vertical_ac: Optional[str] = None
    response: Optional[Quadrupole_response] = None

    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build_quadrupole_response(self, delta: float = 1e-6, save_as: Optional[str] = None):
        if len(self.quadrupoles) == 0:
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
        N = len(SC.bpm_system.names)

        betx_model = SC.lattice.twiss["betx"][SC.bpm_system.indices]
        bety_model = SC.lattice.twiss["bety"][SC.bpm_system.indices]

        betx_est = np.zeros([N, n_kicks])
        bety_est = np.zeros([N, n_kicks])
        freqs = np.fft.fftfreq(fft_n)
        pmask = freqs > 0

        for iter in range(n_kicks):
            logger.info(f"Capturing {iter+1} out {n_kicks} kicks.")

            if self.horizontal_kick is not None:
                SC.kickers.programs[self.horizontal_kick].amplitude = kick_px
                SC.kickers.activate(self.horizontal_kick)
                kick_px_for_capture = 0
            else:
                kick_px_for_capture = kick_px

            if self.vertical_kick is not None:
                SC.kickers.programs[self.vertical_kick].amplitude = kick_py
                SC.kickers.activate(self.vertical_kick)
                kick_py_for_capture = 0
            else:
                kick_py_for_capture = kick_py

            x_tbt, y_tbt = SC.bpm_system.capture_kick(n_turns=n_turns,
                                                      kick_px=kick_px_for_capture,
                                                      kick_py=kick_py_for_capture,
                                                      use_design=use_design)

            if self.horizontal_kick is not None:
                SC.kickers.deactivate(self.horizontal_kick)
            if self.vertical_kick is not None:
                SC.kickers.deactivate(self.vertical_kick)

            qx_rejections = 0
            qy_rejections = 0

            Ax_bpm = np.zeros(N)
            qx_bpm = np.zeros(N)
            for ii in range(N):
                fftx = np.fft.fft(x_tbt[ii] - np.mean(x_tbt[ii]), n=fft_n)
                ix = np.argmax(np.abs(fftx[pmask]))
                amp = fftx[pmask][ix]
                qx = freqs[pmask][ix]
                if qx < qx_low or qx > qx_high:
                    Ax_bpm[ii] = np.nan
                    qx_bpm[ii] = np.nan
                    qx_rejections += 1
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

            # mean_H10 = nanmean(Ax_bpm)
            # betx_est[:,iter] = betx_model * (Ax_bpm / mean_H10)**2
            # mean_V10 = nanmean(Ay_bpm)
            # bety_est[:,iter] = bety_model * (Ay_bpm / mean_V10)**2

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

    def free_kick(self, n_turns: int = 50, kick_px: float = 1e-6, kick_py: float = 1e-6,
                  use_design: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        SC = self._parent._parent

        if self.horizontal_kick is not None:
            SC.kickers.programs[self.horizontal_kick].amplitude = kick_px
            SC.kickers.activate(self.horizontal_kick)
            kick_px_for_capture = 0
        else:
            kick_px_for_capture = kick_px

        if self.vertical_kick is not None:
            SC.kickers.programs[self.vertical_kick].amplitude = kick_py
            SC.kickers.activate(self.vertical_kick)
            kick_py_for_capture = 0
        else:
            kick_py_for_capture = kick_py

        x_tbt, y_tbt = SC.bpm_system.capture_kick(n_turns=n_turns,
                                                  kick_px=kick_px_for_capture,
                                                  kick_py=kick_py_for_capture,
                                                  use_design=use_design)

        if self.horizontal_kick is not None:
            SC.kickers.deactivate(self.horizontal_kick)
        if self.vertical_kick is not None:
            SC.kickers.deactivate(self.vertical_kick)

        return x_tbt, y_tbt

    def _single_frequency_analysis(self, tbt: np.ndarray, q_low: float, q_high: float, fft_n: int):

        N = tbt.shape[0]
        freqs = np.fft.fftfreq(fft_n)
        mask = freqs > 0
        q_rejections = 0

        A_bpm = np.zeros(N)
        #phase_bpm = np.zeros(N)
        q_bpm = np.zeros(N)
        for ii in range(N):
            fftx = np.fft.fft(tbt[ii] - nanmean(tbt[ii]), n=fft_n)
            ix = np.argmax(np.abs(fftx[mask]))
            amp = fftx[mask][ix]
            q = freqs[mask][ix]
            if q < q_low or q > q_high:
                A_bpm[ii] = np.nan
                q_bpm[ii] = np.nan
                #phase_bpm[ii] = np.nan
                q_rejections += 1
            else:
                A_bpm[ii] = abs(amp)
                #phase_bpm[ii] = np.angle(amp)
                q_bpm[ii] = q

        phase_bpm = np.zeros(N)
        average_q = nanmean(q_bpm)
        for ii in range(N):
            amp = dft(tbt[ii] - nanmean(tbt[ii]), average_q)
            phase_bpm[ii] = np.angle(amp)

        return q_bpm, A_bpm, phase_bpm, q_rejections

    def phase_advance(self, n_kicks: int = 1, n_turns: int = 50, 
                      qx_low: float = 0.15, qx_high: float = 0.22,
                      qy_low: float = 0.23, qy_high: float = 0.30,
                      kick_px: float = 1e-6, kick_py: float = 1e-6,
                      use_design: bool = False, fft_n=10000):
        SC = self._parent._parent
        N = len(SC.bpm_system.names)

        mux_model = SC.lattice.twiss["mux"][SC.bpm_system.indices] % 1
        muy_model = SC.lattice.twiss["muy"][SC.bpm_system.indices] % 1

        qx_est = np.zeros([N, n_kicks])
        qy_est = np.zeros([N, n_kicks])
        phasex_est = np.zeros([N, n_kicks])
        phasey_est = np.zeros([N, n_kicks])

        for iter in range(n_kicks):
            logger.info(f"Capturing {iter+1} out {n_kicks} kicks.")

            x_tbt, y_tbt = self.free_kick(n_turns=n_turns, kick_px=kick_px,
                                          kick_py=kick_py, use_design=use_design)

            qx_bpm, Ax_bpm, phasex_bpm, qx_rejections = self._single_frequency_analysis(tbt=x_tbt,
                                                         q_low=qx_low, q_high=qx_high, fft_n=fft_n)
            qy_bpm, Ay_bpm, phasey_bpm, qy_rejections = self._single_frequency_analysis(tbt=y_tbt,
                                                         q_low=qy_low, q_high=qy_high, fft_n=fft_n)

            logger.info(f"Average tunes Qx={nanmean(qx_bpm):.4f}, Qy={nanmean(qy_bpm):.4f}")
            logger.info(f"Rejected {qx_rejections} on Qx and {qy_rejections} on Qy")
            phasex_est[:,iter] = phasex_bpm
            phasey_est[:,iter] = phasey_bpm
            qx_est[:,iter] = qx_bpm
            qy_est[:,iter] = qy_bpm


        qx_mean = nanmean(qx_est)
        qy_mean = nanmean(qy_est)

        # circular mean and circular st.dev.
        phasex_est_mean, phasex_est_std = circular_mean_std(phasex_est, axis=1)
        phasex_est_mean %= TWOPI
        phasey_est_mean, phasey_est_std = circular_mean_std(phasey_est, axis=1)
        phasey_est_mean %= TWOPI

        phasex_est_error = phasex_est_std/np.sqrt(n_kicks)

        phasey_est_error = phasey_est_std/np.sqrt(n_kicks)


        phase_beat_x = circular_diff(phasex_est_mean, TWOPI*mux_model)
        phase_beat_y = circular_diff(phasey_est_mean, TWOPI*muy_model)
        std_phase_beat_x = np.nanstd(phase_beat_x)
        std_phase_beat_y = np.nanstd(phase_beat_y)
        std_phase_beat_x /= TWOPI
        std_phase_beat_y /= TWOPI
        logger.info("Phase advance measurement:")
        logger.info(f"  Estimated horizontal phase-beating: {std_phase_beat_x:.2f} / 2π")
        logger.info(f"  Estimated vertical phase-beating: {std_phase_beat_y:.2f} / 2π")

        phasex_offset, _ = circular_mean_std(phase_beat_x)
        phasey_offset, _ = circular_mean_std(phase_beat_y)
        #phasex_est_mean = circular_diff(phasex_est_mean, phasex_offset)#  / (TWOPI)
        #phasey_est_mean = circular_diff(phasey_est_mean, phasey_offset)#  / (TWOPI)
        mux_est_mean = phasex_est_mean / TWOPI
        muy_est_mean = phasey_est_mean / TWOPI
        mux_est_error = phasex_est_error / TWOPI
        muy_est_error = phasey_est_error / TWOPI
        return mux_est_mean, muy_est_mean, mux_est_error, muy_est_error, qx_mean, qy_mean

    def phase_advance_cheat(self):
        SC = self._parent._parent

        twiss = SC.lattice.get_twiss()
        mux = twiss["mux"][SC.bpm_system.indices] % 1
        muy = twiss["muy"][SC.bpm_system.indices] % 1
        mux_err = np.ones_like(mux)
        muy_err = np.ones_like(muy)
        qx = twiss['qx']
        qy = twiss['qy']
        return mux, muy, mux_err, muy_err, qx, qy

    def ac_kick(self,  n_turns: int = 50, qx_ac: float = 0.176, qy_ac: float = 0.286,
                kick_px: float = 1e-6, kick_py: float = 1e-6, use_design: bool = False):
        SC = self._parent._parent
        if self.horizontal_ac is not None:
            SC.kickers.programs[self.horizontal_ac].amplitude = kick_px
            SC.kickers.programs[self.horizontal_ac].tune = qx_ac
            SC.kickers.activate(self.horizontal_ac)
        else:
            raise Exception("Horizontal ac program was not found. Please specify 'tuning.optics.horizontal_ac'.")

        if self.vertical_ac is not None:
            SC.kickers.programs[self.vertical_ac].amplitude = kick_py
            SC.kickers.programs[self.vertical_ac].tune = qy_ac
            SC.kickers.activate(self.vertical_ac)
        else:
            raise Exception("Vertical ac program was not found. Please specify 'tuning.optics.vertical_ac'.")

        x_tbt, y_tbt = SC.bpm_system.capture_kick(n_turns=n_turns,
                                                  kick_px=0, kick_py=0,
                                                  use_design=use_design)

        SC.kickers.deactivate(self.horizontal_ac)
        SC.kickers.deactivate(self.vertical_ac)

        return x_tbt, y_tbt

    def beta_from_amplitude_ac(self, n_kicks: int = 1, n_turns: int = 50, 
                              qx_ac: float = 0.176, qy_ac: float = 0.286,
                              kick_px: float = 1e-6, kick_py: float = 1e-6,
                              use_design: bool = False):
        """
        estimation of free betatron functions from forced betatron functions through:
        β_{d,z} (s) = (1 + λ_z^2 - 2 λ_z cos(2 φ_z(s) - 2π Q_z)) / (1 - λ_z^2) β_z ,  (Eq.)
        with λ_z = sin( π (Q_{d,z} - Q_z) ) / sin( π (Q_{d,z} + Q_z) )  (Eq)
        Equations (3.17) and (3.13), respectively, from Ref. [1].

        1) F. S. Carlier, "A Nonlinear Future: Measurements and corrections of nonlinear 
           beam dynamics using forced transverse oscillations", 2020, ISBN: 9789464022148, 
           HDL: 11245.1/b489b968-1e7c-4181-8242-01de4f67b515 .
        """

        SC = self._parent._parent
        N = len(SC.bpm_system.names)

        betx_model = SC.lattice.twiss["betx"][SC.bpm_system.indices]
        bety_model = SC.lattice.twiss["bety"][SC.bpm_system.indices]
        mux_model = SC.lattice.twiss["mux"][SC.bpm_system.indices]
        muy_model = SC.lattice.twiss["muy"][SC.bpm_system.indices]

        betx_est = np.zeros([N, n_kicks])
        bety_est = np.zeros([N, n_kicks])

        #forced beta function parameters:
        qx = self._parent.tune.design_qx
        lambda_x = np.sin(np.pi*(qx_ac - qx))/np.sin(np.pi*(qx_ac + qx))
        beta_x_forced_correction = ( 1 - lambda_x**2 ) / (1 + lambda_x**2 - 2 * lambda_x * np.cos(2 * np.pi * (2*mux_model - qx)))

        qy = self._parent.tune.design_qy
        lambda_y = np.sin(np.pi*(qy_ac - qy))/np.sin(np.pi*(qy_ac + qy))
        beta_y_forced_correction = ( 1 - lambda_y**2 ) / (1 + lambda_y**2 - 2 * lambda_y * np.cos(2 * np.pi * (2*muy_model - qy)))

        print(f"{lambda_x=}")
        print(f"{lambda_y=}")
        print(np.max(beta_x_forced_correction), np.std(beta_x_forced_correction))
        print(np.max(beta_y_forced_correction), np.std(beta_y_forced_correction))

        for iter in range(n_kicks):
            logger.info(f"Capturing {iter+1} out {n_kicks} kicks.")

            x_tbt, y_tbt = self.ac_kick(n_turns=n_turns, kick_px=0, kick_py=0, use_design=use_design)

            Ax_bpm = np.zeros(N)
            for ii in range(N):
                amp = dft(x_tbt[ii], qx_ac)
                Ax_bpm[ii] = abs(amp)

            Ay_bpm = np.zeros(N)
            for ii in range(N):
                amp = dft(y_tbt[ii], qy_ac)
                Ay_bpm[ii] = abs(amp)

            action_x_estimate = nanmean(Ax_bpm**2/betx_model) / 2.
            betx_est[:,iter] = Ax_bpm**2 / action_x_estimate / 2. * beta_x_forced_correction

            action_y_estimate = nanmean(Ay_bpm**2/bety_model) / 2.
            bety_est[:,iter] = Ay_bpm**2 / action_y_estimate / 2. * beta_y_forced_correction

        betx_est_mean = nanmean(betx_est, axis=1)
        betx_est_error = nanstd(betx_est, axis=1)/np.sqrt(n_kicks)

        bety_est_mean = nanmean(bety_est, axis=1)
        bety_est_error = nanstd(bety_est, axis=1)/np.sqrt(n_kicks)

        beta_beat_x = nanstd(1 - betx_est_mean/SC.lattice.twiss['betx'][SC.bpm_system.indices])
        beta_beat_y = nanstd(1 - bety_est_mean/SC.lattice.twiss['bety'][SC.bpm_system.indices])
        logger.info("Beta from amplitude (AC) measurement:")
        logger.info(f"  Estimated horizontal beta-beating: {beta_beat_x * 100:.2f}%")
        logger.info(f"  Estimated vertical beta-beating: {beta_beat_y * 100:.2f}%")

        return betx_est_mean, bety_est_mean, betx_est_error, bety_est_error

    def phase_advance_ac(self, n_kicks: int = 1, n_turns: int = 50, 
                         qx_ac: float = 0.176, qy_ac: float = 0.286,
                         kick_px: float = 1e-6, kick_py: float = 1e-6,
                         use_design: bool = False):
        SC = self._parent._parent
        N = len(SC.bpm_system.names)

        mux_model = SC.lattice.twiss["mux"][SC.bpm_system.indices] % 1
        muy_model = SC.lattice.twiss["muy"][SC.bpm_system.indices] % 1

        qx = self._parent.tune.design_qx
        qy = self._parent.tune.design_qy

        factor_x = np.tan(np.pi * qx_ac) / np.tan(np.pi * qx) 
        factor_y = np.tan(np.pi * qy_ac) / np.tan(np.pi * qy) 


        phasex_est = np.zeros([N, n_kicks])
        phasey_est = np.zeros([N, n_kicks])

        for iter in range(n_kicks):
            logger.info(f"Capturing {iter+1} out {n_kicks} kicks.")

            x_tbt, y_tbt = self.ac_kick(n_turns=n_turns, qx_ac=qx_ac, qy_ac=qy_ac, 
                                        kick_px=kick_px, kick_py=kick_py, use_design=use_design)

            phasex_bpm = np.zeros(N)
            for ii in range(N):
                amp = dft(x_tbt[ii] - nanmean(x_tbt[ii]), qx_ac)
                phasex_bpm[ii] = np.angle(amp)

            phasey_bpm = np.zeros(N)
            for ii in range(N):
                amp = dft(y_tbt[ii] - nanmean(x_tbt[ii]), qy_ac)
                phasey_bpm[ii] = np.angle(amp)

            phasex_bpm = np.arctan( factor_x * np.tan(phasex_bpm - np.pi*qx_ac) )
            phasey_bpm = np.arctan( factor_y * np.tan(phasey_bpm - np.pi*qy_ac) )

            phasex_bpm = align_pi_periodic_to_model(phasex_bpm, TWOPI * mux_model)
            phasey_bpm = align_pi_periodic_to_model(phasey_bpm, TWOPI * muy_model)

            phasex_est[:,iter] = phasex_bpm
            phasey_est[:,iter] = phasey_bpm


        # circular mean and circular st.dev.
        phasex_est_mean, phasex_est_std = circular_mean_std(phasex_est, axis=1)
        phasex_est_mean %= TWOPI
        phasey_est_mean, phasey_est_std = circular_mean_std(phasey_est, axis=1)
        phasey_est_mean %= TWOPI

        phasex_est_error = phasex_est_std/np.sqrt(n_kicks)

        phasey_est_error = phasey_est_std/np.sqrt(n_kicks)


        phase_beat_x = circular_diff(phasex_est_mean, TWOPI*mux_model)
        phase_beat_y = circular_diff(phasey_est_mean, TWOPI*muy_model)
        std_phase_beat_x = nanstd(phase_beat_x)
        std_phase_beat_y = nanstd(phase_beat_y)
        std_phase_beat_x /= TWOPI
        std_phase_beat_y /= TWOPI
        logger.info("Phase advance measurement:")
        logger.info(f"  Estimated horizontal phase-beating: {std_phase_beat_x:.2f} / 2π")
        logger.info(f"  Estimated vertical phase-beating: {std_phase_beat_y:.2f} / 2π")

        #phasex_offset, _ = circular_mean_std(phase_beat_x)
        #phasey_offset, _ = circular_mean_std(phase_beat_y)
        #phasex_est_mean = circular_diff(phasex_est_mean, phasex_offset) / (2*np.pi)
        #phasey_est_mean = circular_diff(phasey_est_mean, phasey_offset) / (2*np.pi)
        mux_est_mean = phasex_est_mean / TWOPI
        muy_est_mean = phasey_est_mean / TWOPI
        mux_est_error = phasex_est_error / TWOPI
        muy_est_error = phasey_est_error / TWOPI
        return mux_est_mean, muy_est_mean, mux_est_error, muy_est_error, qx, qy

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

        alpha_c = SC.lattice.get_momentum_compaction(use_design=True)
        factor = - alpha_c * SC.rf_settings.main.frequency

        dx = factor * data.frequency_response_x
        dx_err = factor * data.frequency_response_x_err
        dy = factor * data.frequency_response_y
        dy_err = factor * data.frequency_response_y_err

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
            if obs in ['mux', 'muy']:
                tune_response_name = "qx_response" if obs == "mux" else "qy_response"
                phase_response = getattr(self.response, f"{obs}_response")
                tune_response = getattr(self.response, tune_response_name)

                delta_mu_response = np.diff(
                    phase_response,
                    append=phase_response[:, :1] + tune_response,
                    axis=1
                )
                matrix[ii * nbpm:(ii + 1) * nbpm, :] = delta_mu_response.T
            else:
                matrix[ii * nbpm:(ii + 1) * nbpm, :] = np.transpose(getattr(self.response, f"{obs}_response"))

        input_weights = np.array(self.quad_weights) if self.quad_weights is not None else None
        RM = ResponseMatrix(matrix=matrix, input_weights=input_weights, input_names=self.quadrupoles)

        #RM.output_weights[2*N:] = np.mean(np.std(matrix[:,:2*N], axis=0)) / np.mean(np.std(matrix[:,2*N:], axis=0))
        return RM

    def correct(self, measurements: dict[str,dict], gain: float = 1, correction_method: str = "svd_values",
                correction_parameter: Union[int, float] = 100):
        SC = self._parent._parent
        beta_phase_measurements = [
            'beta_from_amplitude',
            'beta_from_amplitude_ac',
            'phase_advance',
            'phase_advance_ac',
            'phase_advance_cheat',
        ]
        active_beta_phase_measurements = [name for name in beta_phase_measurements if name in measurements]
        if len(active_beta_phase_measurements) > 1:
            raise ValueError(
                f"Only one optics measurement can be used per correction: {active_beta_phase_measurements}"
            )
        observables = []
        beta_phase_measurement = active_beta_phase_measurements[0]
        if beta_phase_measurement in ['beta_from_amplitude', 'beta_from_amplitude_ac', 'beta_cheat']:
            observables.append('betx')
            observables.append('bety')
        if beta_phase_measurement in ['phase_advance', 'phase_advance_ac', 'phase_advance_cheat']:
            observables.append('mux')
            observables.append('muy')
        if 'dispersion' in measurements:
            observables.append('dx')

        bpm_indices = SC.bpm_system.indices
        nbpm = len(bpm_indices)
        model = {obs: SC.lattice.twiss[obs][bpm_indices] for obs in observables }
        RM = self.assemble_response_matrix(observables=observables)

        estimations = {}
        errors = {}
        helpers = {}
        model_helpers = {}
        if 'beta_from_amplitude' in measurements:
            settings = measurements['beta_from_amplitude']
            betx_est, bety_est, betx_err, bety_err = self.beta_from_amplitude(use_design=False, **settings)
            #    n_kicks=1, n_turns=50, qx_low=0.15, qx_high=0.22, qy_low=0.23, qy_high=0.30,
            #    kick_px=1e-6, kick_py=1e-6, fft_n=10000)
            estimations['betx'] = betx_est
            estimations['bety'] = bety_est
            errors['betx'] = betx_err
            errors['bety'] = bety_err
        if 'beta_from_amplitude_ac' in measurements:
            settings = measurements['beta_from_amplitude_ac']
            betx_est, bety_est, betx_err, bety_err = self.beta_from_amplitude_ac(use_design=False, **settings)
            estimations['betx'] = betx_est
            estimations['bety'] = bety_est
            errors['betx'] = betx_err
            errors['bety'] = bety_err
        if 'dispersion' in measurements:
            settings = measurements['dispersion']
            dx_est, dy_est, dx_err, dy_err = self.dispersion(use_design=False, **settings)
            estimations['dx'] = dx_est
            errors['dx'] = dx_err
        if 'phase_advance' in measurements:
            settings = measurements['phase_advance']
            mux_est, muy_est, mux_err, muy_err, qx_est, qy_est = self.phase_advance(use_design=False, **settings)
            estimations['mux'] = mux_est
            estimations['muy'] = muy_est
            errors['mux'] = mux_err
            errors['muy'] = muy_err
            helpers['mux'] = qx_est
            helpers['muy'] = qy_est
            model_helpers['mux'] = SC.lattice.twiss['qx']
            model_helpers['muy'] = SC.lattice.twiss['qy']
        if 'phase_advance_ac' in measurements:
            settings = measurements['phase_advance_ac']
            mux_est, muy_est, mux_err, muy_err, qx_est, qy_est = self.phase_advance_ac(use_design=False, **settings)
            estimations['mux'] = mux_est
            estimations['muy'] = muy_est
            errors['mux'] = mux_err
            errors['muy'] = muy_err
            helpers['mux'] = qx_est
            helpers['muy'] = qy_est
            model_helpers['mux'] = SC.lattice.twiss['qx']
            model_helpers['muy'] = SC.lattice.twiss['qy']
        if 'phase_advance_cheat' in measurements:
            mux_est, muy_est, mux_err, muy_err, qx_est, qy_est = self.phase_advance_cheat()
            estimations['mux'] = mux_est
            estimations['muy'] = muy_est
            errors['mux'] = mux_err
            errors['muy'] = muy_err
            helpers['mux'] = qx_est
            helpers['muy'] = qy_est
            model_helpers['mux'] = SC.lattice.twiss['qx']
            model_helpers['muy'] = SC.lattice.twiss['qy']

        beating = np.zeros([len(observables) * nbpm])
        for ii, obs in enumerate(observables):
            if obs in ['mux', 'muy']:
                tune = TWOPI * helpers[obs]
                phase = TWOPI * estimations[obs]
                #delta_phase = np.diff(phase, append=phase[0] + tune) % TWOPI

                delta_phase = np.concatenate([
                                              circular_diff(phase[1:], phase[:-1]),
                                              [circular_diff(phase[0] + tune, phase[-1])],
                                             ])

                phase_model = TWOPI * model[obs]
                tune_model = TWOPI * model_helpers[obs]

                #delta_phase_model = np.diff(phase_model, append=phase_model[0] + tune_model) % TWOPI
                delta_phase_model = np.concatenate([
                                                    circular_diff(phase_model[1:], phase_model[:-1]),
                                                    [circular_diff(phase_model[0] + tune_model, phase_model[-1])],
                                                   ])

                beating[ii * nbpm:(ii + 1) * nbpm] = circular_diff(delta_phase, delta_phase_model) / TWOPI
            else:
                beating[ii * nbpm:(ii + 1) * nbpm] = estimations[obs] - model[obs]

        trims = - RM.solve(beating, method=correction_method, parameter=correction_parameter)

        initial_k1 = np.array([SC.magnet_settings.get(quad) for quad in self.quadrupoles])
        final_k1 = initial_k1 + gain * trims
        data = {quad: final_k1[ii] for ii,quad in enumerate(self.quadrupoles)}
        SC.magnet_settings.set_many(data)
        return
