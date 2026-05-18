Usage
=====

Import the main public API from the top-level ``pySC`` package:

.. code-block:: python

   from pySC import SimulatedCommissioning
   from pySC import generate_SC
   from pySC import ResponseMatrix

   from pySC import orbit_correction
   from pySC import measure_bba
   from pySC import measure_ORM
   from pySC import measure_dispersion

   from pySC import pySCInjectionInterface
   from pySC import pySCOrbitInterface

Rich progress bars can be disabled when they do not work well in the current
execution environment:

.. code-block:: python

   from pySC import disable_pySC_rich

   disable_pySC_rich()
