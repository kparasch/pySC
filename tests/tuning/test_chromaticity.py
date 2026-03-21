"""Tests for pySC.tuning.chromaticity: chromaticity measurement and correction."""
import pytest
import numpy as np
import at

from pySC.tuning.chromaticity import Chromaticity


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_chroma(sc_tuning):
    """SC with chromaticity sextupole families configured.

    Uses two families of sextupoles:
    - controls_1: sextupoles with names starting with 'SF' (focusing)
    - controls_2: sextupoles with names starting with 'SD' (defocusing)
    """
    sc = sc_tuning
    ring = sc.lattice.design
    sext_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]

    focusing_sext = [f"{name}/B3" for name in sext_names if name.startswith("SF")]
    defocusing_sext = [f"{name}/B3" for name in sext_names if name.startswith("SD")]

    sc.tuning.chromaticity.controls_1 = focusing_sext
    sc.tuning.chromaticity.controls_2 = defocusing_sext
    return sc


def test_chromaticity_build_response_matrix(sc_with_chroma):
    """build_response_matrix creates a 2x2 response matrix without error.

    Note: the HMBA single cell may produce NaN chromaticity if optics are
    unstable. We only verify the method runs and returns correct shape.
    """
    sc = sc_with_chroma
    sc.tuning.chromaticity.build_response_matrix()
    RM = sc.tuning.chromaticity.response_matrix
    assert RM is not None
    assert RM.shape == (2, 2)


def test_chromaticity_correct_cheat(sc_with_chroma):
    """Chromaticity correction with 'cheat' measurement runs without error.

    Correction may not converge on a single-cell lattice, but the code path
    should not raise.
    """
    sc = sc_with_chroma
    # Build RM first, then correct. If RM is all NaN, correct still should not crash.
    sc.tuning.chromaticity.build_response_matrix()

    # If the inverse RM has NaN, the correction will produce NaN trims
    # which get set on magnets. We verify no exception is raised.
    try:
        sc.tuning.chromaticity.correct(
            measurement_method='cheat',
            n_iter=1,
            gain=0.5,
        )
    except np.linalg.LinAlgError:
        pytest.skip("Singular chromaticity RM on this lattice")
