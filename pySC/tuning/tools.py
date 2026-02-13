import logging
from typing import Callable
from ..apps.tools import get_average_orbit as app_get_average_orbit
logger = logging.getLogger(__name__)

def get_average_orbit(get_orbit: Callable, n_orbits: int = 10):
    logger.warning('Please stop using "get_average_orbit" from pySC.tuning.tools. '
                   'Import it from pySC.apps.tools instead!')
    return app_get_average_orbit(get_orbit=get_orbit, n_orbits=n_orbits)