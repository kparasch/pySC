"""Tests for pySC.core.xsuite_lattice: XSuiteLattice helpers."""
from types import SimpleNamespace

import numpy as np

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

    for key in [
        'wx_chrom', 'bx_chrom', 'ax_chrom',
        'wy_chrom', 'by_chrom', 'ay_chrom',
        'dmux', 'dmuy', 'ddx',
    ]:
        np.testing.assert_array_equal(twiss[key], getattr(twiss_result, key))
