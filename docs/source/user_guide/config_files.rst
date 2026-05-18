Config files
============

pySC examples are configured with YAML files passed to ``generate_SC``. A good
configuration starts by defining the lattice and the ``error_table``: the
lattice defines the ideal machine, while the ``error_table`` defines the error
model used to turn that ideal machine into a simulated commissioning case.
After that, add the hardware families that pySC should expose as controls.

The main sections are:

``lattice``
   Selects the simulator and lattice file. For Accelerator Toolbox lattices,
   set ``simulator: at`` and, when needed, the MATLAB variable name with
   ``use``.

``error_table``
   Defines the named error values used by magnets, BPMs, and RF systems. This
   is the central commissioning part of the configuration: hardware sections
   should refer to entries in this table instead of hard-coding error values
   inline. Use zero values only when you intentionally want an ideal model.

``magnets``
   Groups lattice elements by regex or mapping and declares which components are
   controlled, such as ``B2`` for quadrupoles, ``B3`` for sextupoles, ``B1`` for
   horizontal correctors, and ``A1`` for vertical correctors.

``bpms``
   Selects BPM elements and defines calibration and noise settings.

``rf``
   Selects RF cavities and groups them into RF systems.

``tuning``
   Tells pySC which declared magnet controls should be used as horizontal
   correctors, vertical correctors, multipoles, and BBA magnets.

The HMBA example configuration is a compact template:

.. literalinclude:: ../../../examples/hmba_config.yaml
   :language: yaml
   :caption: examples/hmba_config.yaml
