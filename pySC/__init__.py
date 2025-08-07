"""
pySC package
~~~~~~~~~~~~~~~~

pySC

"""

__version__ = "0.3.0"

from .core.new_simulated_commissioning import SimulatedCommissioning
import logging

logging.basicConfig(
    #format='%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:\t%(message)s',
    format="{asctime} | {levelname} | {message}",
    datefmt="%d %b% %Y, %H:%M:%S",
    level=logging.DEBUG,
    style='{'
)
