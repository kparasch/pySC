Config files
============

pySC examples are configured with YAML files passed to ``generate_SC``. A good
configuration starts with the ``error_table`` and the lattice: the
``error_table`` defines the error model used to turn an ideal machine into a
simulated commissioning case, while the lattice defines the ideal machine
itself. After that, add the hardware families that pySC should expose as
controls.

Build the file in this order.

1. Define the error table
-------------------------

``error_table``
   Defines the named error values used by magnets, BPMs, and RF systems. This
   is the central commissioning part of the configuration: hardware sections
   should refer to entries in this table instead of hard-coding error values
   inline. Use zero values only when you intentionally want an ideal model.

For example:

.. code-block:: yaml

   error_table:
     magnet_calibration: 0
     bpm_calibration: 0
     orbit_noise: 0
     tbt_noise: 0

These names are later reused by entries such as ``B2: magnet_calibration`` or
``orbit_noise: orbit_noise``. Keeping the values in one table makes it easy to
scale or replace the error model without rewriting the hardware sections.

2. Select the lattice
---------------------

``lattice``
   Selects the simulator and lattice file. For Accelerator Toolbox lattices,
   set ``simulator: at`` and, when needed, the MATLAB variable name with
   ``use``.

For example:

.. code-block:: yaml

   lattice:
     lattice_file: ../tests/machine_data/hmba32.mat
     simulator: at
     use: RING

If the same element family name appears more than once in the lattice, leave
``naming`` unset so pySC uses index-based names. If the lattice has unique
device names in an element attribute, set ``naming`` to that attribute.

3. Declare magnet controls
--------------------------

``magnets``
   Groups lattice elements by regex or mapping and declares which components are
   controlled, such as ``B2`` for quadrupoles, ``B3`` for sextupoles, ``B1`` for
   horizontal correctors, and ``A1`` for vertical correctors.

Each magnet group needs a selector and a list of controlled components. The
selector is usually a regular expression matched against lattice element names.
Each component points to an entry in ``error_table``.

.. code-block:: yaml

   magnets:
     quadrupoles:
       regex: ^Q
       components:
         - B2: magnet_calibration
     sextupoles:
       regex: ^S[DF]
       components:
         - B3: magnet_calibration
         - B1: magnet_calibration
         - A1: magnet_calibration

Use separate groups when the same physical element type has different roles in
commissioning. For example, sextupoles can provide sextupole strength controls
and embedded corrector controls.

4. Declare BPMs and RF systems
------------------------------

``bpms``
   Selects BPM elements and defines calibration and noise settings.

``rf``
   Selects RF cavities and groups them into RF systems.

For BPMs, select the monitor elements and reference the relevant error-table
entries:

.. code-block:: yaml

   bpms:
     bpms:
       regex: ^BPM
       calibration_error: bpm_calibration
       orbit_noise: orbit_noise
       tbt_noise: tbt_noise

For RF, select the cavity elements. The group name, such as ``main``, becomes
the RF system name in pySC:

.. code-block:: yaml

   rf:
     main:
       regex: ^RFC$

5. Define tuning families
-------------------------

``tuning``
   Tells pySC which declared magnet controls should be used as horizontal
   correctors, vertical correctors, multipoles, and BBA magnets.

The tuning section does not create new controls. It selects from the controls
already declared under ``magnets``.

.. code-block:: yaml

   tuning:
     HCORR:
       - sextupoles: B1
       - correctors: B1
     VCORR:
       - sextupoles: A1
       - correctors: A1
     multipoles:
       - sextupoles: B3
     bba_magnets:
       - quadrupoles: B2

The keys on the left, such as ``sextupoles`` and ``quadrupoles``, must match
groups declared under ``magnets``. The values, such as ``B1`` and ``B2``, must
match components declared in those groups.

The HMBA example configuration is a compact template:

.. literalinclude:: ../../../examples/hmba_config.yaml
   :language: yaml
   :caption: examples/hmba_config.yaml
