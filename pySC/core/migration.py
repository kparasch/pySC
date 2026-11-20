import logging
from typing import TYPE_CHECKING

from .lattice import ATLattice

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

_TOTAL_DIPOLE_STRENGTH_VERSION = "1.6.0"

logger = logging.getLogger(__name__)

def migrate_dipole_settings(SC: "SimulatedCommissioning") -> None:
    """Convert saved AT polynomial strengths to total dipole strengths once."""
    if not isinstance(SC.lattice, ATLattice):
        return
    logger.warning(
        f"Migrating saved AT dipole settings from pySC {SC.version} to the convention "
        f"introduced in {_TOTAL_DIPOLE_STRENGTH_VERSION}. "
        "Save the simulation to retain the migrated settings.",
    )
    for settings in (SC.design_magnet_settings, SC.magnet_settings):
        for magnet in settings.magnets.values():
            elements = SC.lattice.design if magnet.to_design else SC.lattice.ring
            element = elements[magnet.sim_index]
            curvature = (getattr(element, "BendingAngle", 0.0)
                         / SC.lattice.get_length(magnet.sim_index))
            if curvature == 0:
                continue
            dipole_links = [link for link in magnet._links
                            if link.component == "B" and link.order == 1]
            if dipole_links:
                # Add curvature once to the sum, even when several controls
                # (including knobs) contribute to the same dipole.
                link = dipole_links[0]
                scale = magnet.length if link.is_integrated else 1.0
                link.error.offset += curvature * scale
            else:
                # An uncontrolled dipole has no link to convert.
                magnet.offset_B[0] += curvature

    SC.version = _TOTAL_DIPOLE_STRENGTH_VERSION
