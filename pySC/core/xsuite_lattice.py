from .lattice import Lattice
import xtrack as xt
from pydantic import PrivateAttr, model_validator
from typing import Optional, Literal
import re
import numpy as np
from numpy import array as nparray
import xobjects as xo
import logging

logger = logging.getLogger(__name__)

logger.info('Setting "xdeps.tasks" logger to WARNING level')
logging.getLogger('xdeps.tasks').setLevel(logging.WARNING)

COORD_MAP = {'x': 'x', 'px': 'px',
             'y': 'y', 'py': 'py', 
             'tau': 'zeta', 'delta': 'delta'}

class XSuiteLattice(Lattice):
    """
    Represents a lattice defined with XSuite.
    """

    # fake field so that pydantic can distinguish
    # the different machine types
    xsuite_simulator: None = None
    _ring: xt.Line = PrivateAttr(default=None)
    _design: xt.Line = PrivateAttr(default=None)
    _context: xo.ContextCpu = PrivateAttr(default=None)
    _index_to_name: dict[int, str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def load_lattice(self):
        self._context = xo.ContextCpu()
        self._ring = xt.Line.from_json(self.lattice_file)
        self._design = xt.Line.from_json(self.lattice_file)

        if not self.no_6d:
            self._ring.configure_radiation(model='mean')
            self._design.configure_radiation(model='mean')

        self._ring.build_tracker(_context=self._context)
        self._design.build_tracker(_context=self._context)

        self._twiss = self.get_twiss(use_design=True)

        for line in [self._ring, self._design]:
            line.env['pySC'] = 1
            for el in line.elements:
                if hasattr(el, 'k0_from_h'):
                    el.k0_from_h = False
                if el.__class__ is xt.Cavity:
                    el.absolute_time = 0

        self._index_to_name = {ii: name for ii, name in enumerate(line.element_names)}

        return self

    @property
    def omp_num_threads(self):
        return self._omp_num_threads

    @omp_num_threads.setter
    def omp_num_threads(self, value: int):
        self._omp_num_threads = value
        self._context = xo.ContextCpu(omp_num_threads=value)
        self._ring.discard_tracker()
        self._design.discard_tracker()
        self._ring.build_tracker(_context=self._context)
        self._design.build_tracker(_context=self._context)

    def track(self, bunch: nparray, indices: Optional[list[int]] = None, n_turns: int = 1, use_design: bool = False, coordinates: Optional[list] = None) -> nparray:
        n_particles = bunch.shape[0]

        if indices is None:
            indices = [-1]

        n_indices = len(indices)

        if use_design:
            line = self._design
        else:
            line = self._ring

        particles = line.build_particles(x = bunch[:, 0], px = bunch[:, 1],
                             y = bunch[:, 2], py = bunch[:, 3],
                             zeta = bunch[:, 4], delta = bunch[:, 5])

        ## transform coordinates to indices
        if coordinates is None:
            coordinates = ['x', 'y']
        coords = [COORD_MAP[c] for c in coordinates]

        n_coords = len(coordinates)
        out = np.full((n_coords, n_particles, n_indices, n_turns), np.nan)

        for turn in range(n_turns):
            if sum(particles.state > 0): #track only if there are alive particles
                line.track(particles, turn_by_turn_monitor='ONE_TURN_EBE')
                record = line.record_last_track
                lost = record.state[:, indices] != 1
                for ii in range(n_coords):
                    one_coord = getattr(record, coords[ii])[:, indices]
                    out[ii, :, :, turn] = one_coord
                    out[ii, :, :, turn][lost] = np.nan
            else:
                break
        return out

    def get_orbit(self, indices: list[int] = None, use_design: bool = False) -> dict:
        """
        Returns the closed orbit for the specified indices.
        If no indices are provided, returns the closed orbit for all elements.
        """
        if use_design:
            line = self._design
        else:
            line = self._ring

        try:
            part_co = line.find_closed_orbit()
            line.track(part_co, turn_by_turn_monitor='ONE_TURN_EBE')
            x = line.record_last_track.x[0]
            y = line.record_last_track.y[0]
        except xt.twiss.ClosedOrbitSearchError as exc:
            logger.critical(f'Caught exception "ClosedOrbitSearchError": {exc}')
            x = np.full(len(self.twiss['s']), np.nan)
            y = np.full(len(self.twiss['s']), np.nan)

        if indices is not None:
            x = x[indices]
            y = y[indices]
        return x, y

    def get_twiss(self, indices: Optional[list[int]] = None, use_design: bool = False) -> dict:
        """
        Returns the twiss parameters for the specified indices.
        If no indices are provided, returns all twiss parameters.
        """
        if indices is None:
            indices = range(len(self._design))
        line = self._design if use_design else self._ring
        if self.no_6d:
            tw = line.twiss(method='4d', at_elements=indices)
        else:
            tw = line.twiss(at_elements=indices)

        twiss = {'qx': tw.qx,
                 'qy': tw.qy,
                 'qs': tw.qs,
                 'dqx': tw.dqx,
                 'dqy': tw.dqy,
                 'name': tw.name,
                 's': tw.s,
                 'x' : tw.x,
                 'px': tw.px,
                 'y' : tw.y,
                 'py': tw.py,
                 'delta': tw.delta,
                 'tau': tw.zeta, # TODO: fix tau/zeta
                 'betx': tw.betx,
                 'bety': tw.bety,
                 'alfx': tw.alfx,
                 'alfy': tw.alfy,
                 'mux': tw.mux,
                 'muy': tw.muy,
                 'dx' : tw.dx,
                 'dpx': tw.dpx,
                 'dy' : tw.dy,
                 'dpy': tw.dpy,
                }
        return twiss 

    def get_tune(self, method: str = '6d', use_design: bool = False) -> tuple[float, float]:
        assert method in ['4d', '6d']
        line = self._design if use_design else self._ring

        if self.no_6d:
            logger.warning("Lattice has 6d disabled, using 4d method instead.")
            method = '4d'

        try:
            twiss = line.twiss()
            qx = twiss.qx
            qy = twiss.qy
        except Exception as e:
            logger.error(f"Error computing tune, {type(e)}: {e}")
            qx = np.nan
            qy = np.nan

        return qx, qy

    def find_with_regex(self, regex: str) -> list[int]:
        """
        Find elements in the ring that match the given regular expression.
        Returns a list of indices of the matching elements.
        """
        #indices = [ind for ind, name in enumerate(self.twiss['name']) if re.search(regex, name)]
        indices = [ind for ind, name in enumerate(self._ring.element_names) if re.search(regex, name)]
        return indices

    def is_dipole(self, index: int) -> bool:
        """
        Check if the element at the given index is a dipole with bending angle.
        """
        name = self._index_to_name[index]
        elem = self._design.element_dict[name]
        if hasattr(elem, 'h') and elem.h != 0:
            return True
        else:
            return False

    def get_bending_angle(self, index: int) -> float:
        name = self._index_to_name[index]
        elem = self._design.element_dict[name]
        return getattr(elem, 'angle', 0)

    def get_length(self, index: int) -> float:
        name = self._index_to_name[index]
        elem = self._design.element_dict[name]
        if hasattr(elem, 'length') and elem.length:
            return elem.length
        else: # when length is zero
            return 1

    def get_magnet_component(self, index: int, component_type: Literal['A', 'B'],
                             order: int, use_design=True) -> float:
        assert component_type in ['A', 'B']
        name = self._index_to_name[index]
        if use_design:
            elem = self._design.element_dict[name]
        else:
            elem = self._ring.element_dict[name]

        if elem.__class__ is xt.Bend:
            if component_type == 'B' and order == 0:
                extra = elem.k0
            elif component_type == 'B' and order == 1:
                extra = elem.k1
            else:
                extra = 0.0
        elif elem.__class__ is xt.Quadrupole:
            if component_type == 'B' and order == 1:
                extra = elem.k1
            elif component_type == 'A' and order == 1:
                extra = elem.k1s
            else:
                extra = 0.0
        elif elem.__class__ is xt.Sextupole:
            if component_type == 'B' and order == 2:
                extra = elem.k2
            elif component_type == 'A' and order == 2:
                extra = elem.k2s
            else:
                extra = 0.0
        elif elem.__class__ is xt.Octupole:
            if component_type == 'B' and order == 3:
                extra = elem.k3
            elif component_type == 'A' and order == 3:
                extra = elem.k3s
            else:
                extra = 0.0
        elif elem.__class__ is xt.Multipole:
            extra = 0.0
        else:
            raise NotImplementedError(f'Element type {elem.__class__} not implemented in get_magnet_component.')

        length = self.get_length(index)
        if component_type == 'B':
            value = elem.knl[order] / length
        elif component_type == 'A':
            value = elem.ksl[order] / length
        else:
            raise ValueError(f'Unknown component type {component_type}')

        return value + extra

    def set_magnet_component(self, index: int, value: float,
                             component_type: Literal['A', 'B'],
                             order: int, use_design=True) -> None:

        assert component_type in ['A', 'B']
        if use_design:
            line = self._design
        else:
            line = self._ring
        env = line.env
        element_name = self._index_to_name[index]
        element_class = line.element_dict[element_name].__class__

        value_before = self.get_magnet_component(index, component_type, order, use_design=use_design)
        difference = value - value_before

        expression_name = f"pySC_magnet_{index}_{component_type}{order}"

        if expression_name not in env.vars:
            length = self.get_length(index)
            line.vars[expression_name] = 0.0
            assigned = False
            if element_class is xt.Bend:
                if component_type == 'B' and order == 0:
                    env.ref[element_name].k0 += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
                elif component_type == 'B' and order == 1:
                    env.ref[element_name].k1 += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
            elif element_class is xt.Quadrupole:
                if component_type == 'B' and order == 1:
                    env.ref[element_name].k1 += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
                elif component_type == 'A' and order == 1:
                    env.ref[element_name].k1s += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
            elif element_class is xt.Sextupole:
                if component_type == 'B' and order == 2:
                    env.ref[element_name].k2 += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
                elif component_type == 'A' and order == 2:
                    env.ref[element_name].k2s += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
            elif element_class is xt.Octupole:
                if component_type == 'B' and order == 3:
                    env.ref[element_name].k3 += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
                elif component_type == 'A' and order == 3:
                    env.ref[element_name].k3s += env.ref['pySC'] * env.ref[expression_name]
                    assigned = True
            if not assigned:
                if component_type == 'B':
                    env.ref[element_name].knl[order] += env.ref['pySC'] * env.ref[expression_name] * length
                else:
                    env.ref[element_name].ksl[order] += env.ref['pySC'] * env.ref[expression_name] * length

        env[expression_name] += difference

        return

    def get_cavity_voltage_phase_frequency(self, index: int, use_design=True):
        ## TODO: use defferred expressions for cavity parameters
        name = self._index_to_name[index]
        if use_design:
            line = self._design
        else:
            line = self._ring
        elem = line.element_dict[name]

        assert elem.__class__ is xt.Cavity, f"Class of element {name} is not xt.Cavity, it is instead: {elem.__class__}"

        voltage = elem.voltage
        phase = elem.lag
        frequency = elem.frequency

        return voltage, phase, frequency

    def update_cavity(self, index: int, voltage: float, phase: float, frequency: float, use_design=True):
        ## TODO: use defferred expressions for cavity parameters
        name = self._index_to_name[index]
        if use_design:
            line = self._design
        else:
            line = self._ring
        env = line.env
        element_name = self._index_to_name[index]
        element_class = line.element_dict[element_name].__class__

        assert element_class is xt.Cavity, f"Class of element {name} is not xt.Cavity, it is instead: {element_class}"

        voltage_before, phase_before, frequency_before = self.get_cavity_voltage_phase_frequency(index, use_design=use_design)
        voltage_difference = voltage - voltage_before
        phase_difference = phase - phase_before
        frequency_difference = frequency - frequency_before

        expression_name_voltage = f"pySC_rf_{index}_voltage"
        if expression_name_voltage not in env.vars:
            line.vars[expression_name_voltage] = 0.0
            env.ref[element_name].voltage += env.ref['pySC'] * env.ref[expression_name_voltage]

        expression_name_phase = f"pySC_rf_{index}_phase"
        if expression_name_phase not in env.vars:
            line.vars[expression_name_phase] = 0.0
            env.ref[element_name].lag += env.ref['pySC'] * env.ref[expression_name_phase]

        expression_name_frequency = f"pySC_rf_{index}_frequency"
        if expression_name_frequency not in env.vars:
            line.vars[expression_name_frequency] = 0.0
            env.ref[element_name].frequency += env.ref['pySC'] * env.ref[expression_name_frequency]

        env[expression_name_voltage] += voltage_difference
        env[expression_name_phase] += phase_difference
        env[expression_name_frequency] += frequency_difference
        return

    def update_misalignment(self, index: int, dx: Optional[float] = None, dy: Optional[float] = None,
                            dz: Optional[float] = None, roll: Optional[float] = None, yaw: Optional[float] = None,
                            pitch: Optional[float] = None, use_design=False) -> None:
        line = self._design if use_design else self._ring
        element_name = line.element_names[index]
        env = line.env

        if dx is not None:
            expression_name = f"pySC_dx_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].shift_x += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = dx

        if dy is not None:
            expression_name = f"pySC_dy_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].shift_y += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = dy

        if dz is not None:
            expression_name = f"pySC_dz_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].shift_s += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = dz

        if roll is not None:
            expression_name = f"pySC_roll_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].rot_s_rad += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = roll

        if pitch is not None:
            expression_name = f"pySC_pitch_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].rot_x_rad += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = pitch

        if yaw is not None:
            expression_name = f"pySC_yaw_{index}"
            if expression_name not in env.vars:
                env[expression_name] = 0.
                env.ref[element_name].rot_y_rad += env.ref['pySC'] * env.ref[expression_name]
            env[expression_name] = yaw

        return