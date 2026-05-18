# pySC

[![Documentation Status](https://readthedocs.org/projects/accelerator-commissioning/badge/?version=latest)](https://accelerator-commissioning.readthedocs.io/en/latest/?badge=latest)

Python Simulated Commissioning toolkit for synchrotrons.

Documentation: https://accelerator-commissioning.readthedocs.io/

## Installing

```bash
pip install accelerator-commissioning
```

## Importing specific modules

Intended way of importing a pySC functionality:

```
from pySC import SimulatedCommissioning
from pySC import generate_SC

from pySC import ResponseMatrix

from pySC import orbit_correction
from pySC import measure_bba
from pySC import measure_ORM
from pySC import measure_dispersion

from pySC import pySCInjectionInterface
from pySC import pySCOrbitInterface

# the following disables rich progress bars (doesn't work well with )
from pySC import disable_pySC_rich
disable_pySC_rich()
```


## Acknowledgements

This toolkit was inspired by [SC](https://github.com/ThorstenHellert/SC) which is written in Matlab.
