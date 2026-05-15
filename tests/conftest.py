"""Root test fixtures: HMBA lattice, SC factory, shared helpers."""
import os
import pytest
import numpy as np
import at

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.rfsettings import RFCavity, RFSystem

MACHINE_DATA = os.path.join(os.path.dirname(__file__), "machine_data")
HMBA_MAT = os.path.join(MACHINE_DATA, "hmba.mat")


@pytest.fixture(scope="session")
def hmba_ring():
    """AT HMBA lattice loaded once per session for reference/inspection."""
    return at.load_lattice(HMBA_MAT)


@pytest.fixture(scope="session")
def hmba_lattice_file():
    """Path to the HMBA .mat file."""
    return HMBA_MAT


@pytest.fixture
def sc(hmba_lattice_file):
    """Fully configured SimulatedCommissioning from HMBA lattice (fresh per test).

    Lattice inventory (HMBA, 1 cell of a 6 GeV storage ring):
      - 10 BPMs    (BPM_01 .. BPM_10)
      -  1 RF cav  (RFC, index 0)
      - 16 quads   (QF1A, QD2A, QD3A, QF4A, QF4B, QD5B, QF6B, QF8B,
                     QF8D, QF6D, QD5D, QF4D, QF4E, QD3E, QD2E, QF1E)
      -  6 sext    (SD1A, SF2A, SD1B, SD1D, SF2E, SD1E)
      - 32 dipoles
      -  5 multipoles (SH1A, OF1B, SH2B, OF1D, SH3E)

    Configuration applied programmatically (no YAML):
      - Quadrupoles: individually powered on B2 (normal quad component)
      - Sextupoles:  individually powered on B3 (normal sext) + B1 (H corrector) + A1 (V corrector)
      - BPMs:        all 10 registered
      - RF:          single 'main' system with 1 cavity
      - Supports:    L0 elements registered for all magnets and BPMs
    """
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    SC = SimulatedCommissioning(lattice=lattice, seed=42)

    # --- Identify element indices by type ---
    ring = SC.lattice.design
    quad_indices = [i for i, e in enumerate(ring) if isinstance(e, at.Quadrupole)]
    sext_indices = [i for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]
    bpm_indices = [i for i, e in enumerate(ring) if isinstance(e, at.Monitor)]
    rf_indices = [i for i, e in enumerate(ring) if isinstance(e, at.RFCavity)]

    # --- Configure magnets ---
    for idx in quad_indices:
        name = ring[idx].FamName
        length = SC.lattice.get_length(idx)
        SC.magnet_settings.add_individually_powered_magnet(
            sim_index=idx, controlled_components=["B2"],
            magnet_name=name, magnet_length=length)
        SC.design_magnet_settings.add_individually_powered_magnet(
            sim_index=idx, controlled_components=["B2"],
            magnet_name=name, magnet_length=length, to_design=True)

    # Sextupoles with embedded H/V corrector windings (B1 = H, A1 = V)
    for idx in sext_indices:
        name = ring[idx].FamName
        length = SC.lattice.get_length(idx)
        SC.magnet_settings.add_individually_powered_magnet(
            sim_index=idx, controlled_components=["B3", "B1", "A1"],
            magnet_name=name, magnet_length=length)
        SC.design_magnet_settings.add_individually_powered_magnet(
            sim_index=idx, controlled_components=["B3", "B1", "A1"],
            magnet_name=name, magnet_length=length, to_design=True)

    SC.magnet_settings.sendall()
    SC.design_magnet_settings.sendall()

    # --- Configure BPMs ---
    n_bpms = len(bpm_indices)
    SC.bpm_system.indices = bpm_indices
    SC.bpm_system.names = [ring[i].FamName for i in bpm_indices]
    SC.bpm_system.calibration_errors_x = np.zeros(n_bpms)
    SC.bpm_system.calibration_errors_y = np.zeros(n_bpms)
    SC.bpm_system.offsets_x = np.zeros(n_bpms)
    SC.bpm_system.offsets_y = np.zeros(n_bpms)
    SC.bpm_system.rolls = np.zeros(n_bpms)
    SC.bpm_system.noise_co_x = np.zeros(n_bpms)
    SC.bpm_system.noise_co_y = np.zeros(n_bpms)
    SC.bpm_system.noise_tbt_x = np.zeros(n_bpms)
    SC.bpm_system.noise_tbt_y = np.zeros(n_bpms)
    SC.bpm_system.bba_offsets_x = np.zeros(n_bpms)
    SC.bpm_system.bba_offsets_y = np.zeros(n_bpms)
    SC.bpm_system.reference_x = np.zeros(n_bpms)
    SC.bpm_system.reference_y = np.zeros(n_bpms)
    SC.bpm_system.gain_corrections_x = np.ones(n_bpms)
    SC.bpm_system.gain_corrections_y = np.ones(n_bpms)
    SC.bpm_system.update_rot_matrices()

    # --- Configure RF ---
    assert len(rf_indices) == 1, f"Expected 1 RF cavity, got {len(rf_indices)}"
    rf_idx = rf_indices[0]
    voltage, phase, frequency = SC.lattice.get_cavity_voltage_phase_frequency(rf_idx)
    cav_name = ring[rf_idx].FamName

    cavity = RFCavity(sim_index=rf_idx)
    design_cavity = RFCavity(sim_index=rf_idx, to_design=True)

    system = RFSystem(cavities=[cav_name], voltage=voltage, phase=phase, frequency=frequency)
    design_system = RFSystem(cavities=[cav_name], voltage=voltage, phase=phase, frequency=frequency)

    SC.rf_settings.systems["main"] = system
    SC.rf_settings.cavities[cav_name] = cavity

    SC.design_rf_settings.systems["main"] = design_system
    SC.design_rf_settings.cavities[cav_name] = design_cavity

    # Re-propagate parents after manual configuration
    SC.propagate_parents()

    # Trigger RF update so cavity elements match settings
    for rf_settings in [SC.rf_settings, SC.design_rf_settings]:
        for system_name in rf_settings.systems:
            rf_settings.systems[system_name].trigger_update()

    return SC
