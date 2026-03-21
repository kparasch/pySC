"""Tuning-layer fixtures: configured SC with magnets + BPMs for tuning tests."""
import pytest
import at


@pytest.fixture
def sc_tuning(sc):
    """SC fixture extended with tuning configuration (HCORR, VCORR, bba_magnets).

    Sextupoles in the HMBA lattice have B1 (H corrector) and A1 (V corrector)
    windings, so control names are '<sext_name>/B1' and '<sext_name>/A1'.
    Quadrupole control names are '<quad_name>/B2'.
    """
    ring = sc.lattice.design
    sext_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]
    quad_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Quadrupole)]

    sc.tuning.HCORR = [f"{name}/B1" for name in sext_names]
    sc.tuning.VCORR = [f"{name}/A1" for name in sext_names]
    sc.tuning.bba_magnets = [f"{name}/B2" for name in quad_names]

    # Register L0 supports for all magnets and BPMs (needed for bba_to_quad_true_offset)
    quad_indices = [i for i, e in enumerate(ring) if isinstance(e, at.Quadrupole)]
    sext_indices = [i for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]
    for idx in quad_indices + sext_indices + list(sc.bpm_system.indices):
        if idx not in sc.support_system.data['L0']:
            sc.support_system.add_element(idx)

    return sc
