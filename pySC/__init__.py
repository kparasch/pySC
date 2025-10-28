"""
pySC package
~~~~~~~~~~~~~~~~

pySC

"""

__version__ = "0.3.0"

from .core.new_simulated_commissioning import SimulatedCommissioning
from .configuration.generation import generate_SC

import logging
import sys

logging.basicConfig(
    #format='%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:\t%(message)s',
    format="{asctime} | {levelname} | {message}",
    datefmt="%d %b% %Y, %H:%M:%S",
    level=logging.INFO,
    style='{',
    stream=sys.stdout
)

def disable_pySC_rich():
    from .tuning import response_measurements
    response_measurements.DISABLE_RICH = True