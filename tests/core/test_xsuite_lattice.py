"""Tests for pySC.core.xsuite_lattice: XSuiteLattice helpers."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c as clight

from pySC.core.transformations import at_rotation, xsuite_angles_from_rotation
from pySC.core.xsuite_lattice import XSuiteLattice


class _FakeLine:
    """Minimal Xsuite line double exposing the twiss API used by XSuiteLattice."""

    def __init__(self, twiss):
        self._twiss = twiss

    def __len__(self):
        return len(self._twiss.s)

    def twiss(self, **kwargs):
        self.last_twiss_kwargs = kwargs
        return self._twiss


class _FakeExpression:
    def __init__(self, name):
        self.name = name

    def __mul__(self, other):
        return ("mul", self.name, other)

    def __rmul__(self, other):
        return ("mul", other, self.name)


class _FakeField:
    def __init__(self):
        self.added = []

    def __iadd__(self, value):
        self.added.append(value)
        return self


class _FakeRef:
    def __init__(self, element):
        self._refs = {'pySC': _FakeExpression('pySC'), 'e0': element}

    def __getitem__(self, key):
        return self._refs.get(key, _FakeExpression(key))


class _FakeEnv:
    def __init__(self, element):
        self.vars = {}
        self.ref = _FakeRef(element)

    def __setitem__(self, key, value):
        self.vars[key] = value


def test_xsuite_get_twiss_exposes_chromatic_keys():
    """XSuiteLattice.get_twiss maps chromatic Twiss fields into pySC keys."""
    values = np.array([0.0, 1.0, 2.0])
    twiss_result = SimpleNamespace(
        qx=0.31,
        qy=0.32,
        qs=0.01,
        dqx=1.0,
        dqy=2.0,
        name=np.array(["e0", "e1", "e2"]),
        s=values,
        x=values + 0.1,
        px=values + 0.2,
        y=values + 0.3,
        py=values + 0.4,
        delta=values + 0.5,
        zeta=values + 0.6,
        betx=values + 10.0,
        bety=values + 20.0,
        alfx=values + 0.7,
        alfy=values + 0.8,
        mux=values + 0.9,
        muy=values + 1.0,
        dx=values + 1.1,
        dpx=values + 1.2,
        dy=values + 1.3,
        dpy=values + 1.4,
        wx_chrom=values + 1.5,
        bx_chrom=values + 1.6,
        ax_chrom=values + 1.7,
        wy_chrom=values + 1.8,
        by_chrom=values + 1.9,
        ay_chrom=values + 2.0,
        dmux=values + 2.1,
        dmuy=values + 2.2,
        ddx=values + 2.3,
    )
    line = _FakeLine(twiss_result)
    lattice = XSuiteLattice.model_construct(lattice_file="dummy.json", no_6d=False)
    lattice._design = line
    lattice._ring = line
    lattice.num_turns_search_t_rev = 5

    twiss = lattice.get_twiss(use_design=True)

    assert list(line.last_twiss_kwargs['at_elements']) == [0, 1, 2, 3]
    for key in [
        'wx_chrom', 'bx_chrom', 'ax_chrom',
        'wy_chrom', 'by_chrom', 'ay_chrom',
        'dmux', 'dmuy', 'ddx',
    ]:
        np.testing.assert_array_equal(twiss[key], getattr(twiss_result, key))


def test_xsuite_get_brho_uses_particle_ref_p0c():
    """get_Brho() returns particle_ref.p0c / c."""
    p0c = np.array([6.0e9])
    line = SimpleNamespace(particle_ref=SimpleNamespace(p0c=p0c))
    lattice = XSuiteLattice.model_construct(lattice_file="dummy.json", no_6d=False)
    lattice._design = line
    lattice._ring = line

    assert lattice.get_Brho(use_design=True) == pytest.approx(p0c[0] / clight)


def test_xsuite_update_misalignment_uses_xsuite_rotation_convention():
    """XSuite roll is a no-frame field rotation; x-rotation has XSuite sign."""
    element = SimpleNamespace(
        shift_x=_FakeField(),
        shift_y=_FakeField(),
        shift_s=_FakeField(),
        rot_s_rad=_FakeField(),
        rot_s_rad_no_frame=_FakeField(),
        rot_x_rad=_FakeField(),
        rot_y_rad=_FakeField(),
        rot_shift_anchor=0.0,
        length=2.0,
    )
    env = _FakeEnv(element)
    line = SimpleNamespace(element_names=['e0'], element_dict={'e0': element}, env=env)
    lattice = XSuiteLattice.model_construct(lattice_file="dummy.json", no_6d=False)
    lattice._ring = line

    rot = at_rotation(pitch=0.11, yaw=-0.07, roll=0.05)
    expected_rot_s, expected_rot_x, expected_rot_y = xsuite_angles_from_rotation(rot)

    lattice.update_misalignment(0, dx=1e-3, dy=2e-3, ds=3e-3, rot=rot)

    assert element.shift_x.added
    assert element.shift_y.added
    assert element.shift_s.added
    assert element.rot_s_rad.added == []
    assert element.rot_s_rad_no_frame.added
    assert element.rot_x_rad.added
    assert element.rot_y_rad.added
    assert element.rot_shift_anchor == pytest.approx(1.0)
    assert env.vars['pySC_roll_no_frame_0'] == pytest.approx(expected_rot_s)
    assert env.vars['pySC_pitch_0'] == pytest.approx(expected_rot_x)
    assert env.vars['pySC_yaw_0'] == pytest.approx(expected_rot_y)
