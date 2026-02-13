"""
pySC package
~~~~~~~~~~~~~~~~

pySC

"""

__version__ = "1.0.0"

from .core.simulated_commissioning import SimulatedCommissioning
from .configuration.generation import generate_SC
from .apps.response_matrix import ResponseMatrix
from .apps.measurements import orbit_correction
from .apps.measurements import measure_bba
from .apps.measurements import measure_ORM
from .apps.measurements import measure_dispersion
from .tuning.pySC_interface import pySCInjectionInterface, pySCOrbitInterface
import logging
import sys

logging.basicConfig(
    #format='%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:\t%(message)s',
    format="{asctime} | {levelname} | {message}",
    datefmt="%d %b %Y, %H:%M:%S",
    level=logging.INFO,
    style='{',
    stream=sys.stdout
)

def disable_pySC_rich():
    from .apps import response
    response.DISABLE_RICH = True

# This is needed to avoid circular imports.
# Firstly the type of SC is hinted to avoid importing SimulatedCommissioning:
#   class pySCOrbitInterface(AbstractInterface):
#      SC: "SimulatedCommissioning" = Field(repr=False)
#
# Then, the model_rebuild is triggered here to complete the pydantic model,
# and allow validation.
# for this to be triggered, one needs to import pySC or to import from pySC
# (i.e. from pySC import ...)
# to validate a pySCInjectionInterface/pySCOrbitInterface object, one should
# already have a SimulatedCommissioning object. To acquire the SimulatedCommissioning,
# the model_rebuild is "almost certainly"? triggered.
pySCInjectionInterface.model_rebuild()
pySCOrbitInterface.model_rebuild()
