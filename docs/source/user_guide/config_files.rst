Config files
============

pySC examples are configured with YAML files passed to ``generate_SC``. A good
configuration starts with the ``error_table`` and the lattice: the
``error_table`` defines the error model used to turn an ideal machine into a
simulated commissioning case, while the lattice defines the ideal machine
itself. After that, add the hardware families that pySC should expose as
controls.

Build the file in this order.

1. Define shared tables
-----------------------

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

The ``error_table`` does not have predefined option names. It can contain any
combination of ``key: value`` pairs. The keys are chosen by the configuration
author, and the values are numeric error amplitudes. Hardware sections then
reference those keys.

``parameters``
   Defines named non-random parameters used by the configuration. Unlike
   ``error_table``, this table is for deterministic values such as control
   limits.

For example:

.. code-block:: yaml

   parameters:
     quad_limit: 100

The ``parameters`` table is currently used by magnet ``limits`` entries:

.. code-block:: yaml

   magnets:
     quadrupoles:
       regex: ^Q
       components:
         - B2: magnet_calibration
       limits:
         - B2: quad_limit

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
device names in an element attribute, set ``naming`` to that attribute; for
example, use ``naming: Device`` when AT elements carry unique device names in a
``Device`` field.

A hypothetical Xsuite lattice uses the same section but switches the simulator
and points to an Xsuite line JSON file:

.. code-block:: yaml

   lattice:
     lattice_file: my_machine_line.json
     simulator: xsuite

Supported lattice options are:

``lattice_file``
   Path to the lattice input file.

``simulator``
   Lattice backend. Supported values are ``at`` and ``xsuite``.

``use``
   AT-only option selecting the variable to load from a MATLAB lattice file.
   Defaults to ``RING`` when omitted.

``naming``
   AT-only option selecting the element attribute used as the pySC element name.
   Omit it when names are not unique.

``no_6d``
   Boolean flag controlling whether pySC disables 6D tracking when loading the
   lattice. The default is ``false``.

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

Supported magnet-family options are:

``regex`` or ``mapping``
   Selects lattice elements for the family. ``regex`` matches lattice element
   names; ``mapping`` points to an explicit name-to-index mapping file.

``components``
   List of controlled multipole components. Each component maps to an
   ``error_table`` entry or ``null``.

``invert``
   List of declared components whose sign should be inverted.

``limits``
   Optional control limits. Each listed component references a value from the
   top-level ``parameters`` table.

``dx``, ``dy``, ``dz``, ``roll``, ``yaw``, ``pitch``
   Optional element misalignment entries. Each value must reference an
   ``error_table`` entry.

Components without an ``L`` suffix are interpreted as non-integrated strengths.
Append ``L`` to use integrated strengths instead:

.. code-block:: yaml

   magnets:
     quadrupoles:
       regex: ^Q
       components:
         - B2L: magnet_calibration

Use ``invert`` when a control-system convention has the opposite sign from the
simulation convention. Every inverted component must also be listed under
``components``:

.. code-block:: yaml

   magnets:
     sextupoles:
       regex: ^S[DF]
       components:
         - B1: magnet_calibration
         - A1: magnet_calibration
       invert:
         - B1

4. Declare BPMs
---------------

``bpms``
   Selects BPM elements and defines calibration and noise settings.

For BPMs, select the monitor elements and reference the relevant error-table
entries. The double ``bpms: bpms:`` nesting is intentional: the first key is
the top-level BPM configuration section, and the second key is the name of one
BPM family. This is useful when a machine has multiple BPM families with
different error models.

.. code-block:: yaml

   bpms:
     bpms:
       regex: ^BPM
       calibration_error: bpm_calibration
       orbit_noise: orbit_noise
       tbt_noise: tbt_noise

Supported BPM-family options are:

``regex`` or ``mapping``
   Selects the BPM elements. ``regex`` matches lattice element names; ``mapping``
   points to an explicit name-to-index mapping file.

``calibration_error``
   Error-table entry used for BPM calibration errors in both planes.

``orbit_noise``
   Error-table entry used for closed-orbit BPM noise.

``tbt_noise``
   Error-table entry used for turn-by-turn BPM noise.

``dx``, ``dy``, ``roll``
   Optional BPM misalignment entries. Each value must reference an
   ``error_table`` entry. For thin BPM elements these are the physically useful
   alignment terms: transverse offsets and roll. Longitudinal shifts and pitch
   or yaw are support-system fields in pySC, but they are usually not meaningful
   for BPMs modeled as thin monitors.

For example, two BPM families can be configured as:

.. code-block:: yaml

   bpms:
     arc_bpms:
       regex: ^BPM
       calibration_error: bpm_calibration
       orbit_noise: orbit_noise
       tbt_noise: tbt_noise
     id_bpms:
       regex: ^IDBPM
       calibration_error: bpm_calibration
       orbit_noise: orbit_noise
       tbt_noise: tbt_noise

5. Declare support systems
--------------------------

``supports``
   Defines alignment errors for support structures such as girders. Supports
   are configured as a list because a machine can have several support levels or
   families.

Each support entry defines a level, a pair of endpoint selectors, and the
alignment errors to apply. The error values reference entries in
``error_table``.

.. code-block:: yaml

   supports:
     - level: 1
       start_endpoints:
         regex: GS
       end_endpoints:
         regex: GE
       name: Girder
       dx: girder_offsets
       dy: girder_offsets
       roll: girder_rolls
       alignment: absolute

Supported support options are:

``level``
   Support hierarchy level. For example, ``level: 1`` creates an ``L1`` support
   family.

``start_endpoints`` and ``end_endpoints``
   Select the start and end elements for each support. They use the same
   selector style as magnets and BPMs, usually ``regex`` or ``mapping``.

``name``
   Optional family name used instead of the default level name.

``dx`` and ``dy``
   Horizontal and vertical support endpoint offsets. Each value must reference
   an ``error_table`` entry.

``roll``
   Support roll error. The value must reference an ``error_table`` entry.

``alignment``
   Either ``absolute`` or ``relative``. With ``absolute`` alignment, start and
   end offsets are drawn independently from the configured distribution. With
   ``relative`` alignment, pySC scales the endpoint offset sigma by
   :math:`1/\sqrt{2}` so the relative offset between endpoints has the requested
   sigma.

6. Declare RF systems
---------------------

``rf``
   Selects RF cavities, groups them into RF systems, and references RF error
   entries from ``error_table``.

For RF, select the cavity elements. The group name, such as ``main``, becomes
the RF system name in pySC. Phase and frequency errors are optional fields in
the schema, but should be listed explicitly in commissioning configurations so
the RF error model is visible in the YAML file:

.. code-block:: yaml

   rf:
     main:
       regex: ^RFC$
       frequency: rf_frequency
       phase: rf_phase

The referenced ``rf_frequency`` and ``rf_phase`` entries are defined in
``error_table``. Set them to zero for an ideal RF model.

Supported RF-system options are:

``regex`` or ``mapping``
   Selects RF cavities. ``regex`` matches lattice element names; ``mapping``
   points to an explicit name-to-index mapping file.

``voltage``
   Error-table entry for cavity voltage errors.

``phase``
   Error-table entry for cavity phase errors.

``frequency``
   Error-table entry for cavity frequency errors.

7. Define injection settings
----------------------------

``injection``
   Defines the injected beam coordinates, beam parameters, and injection jitter
   used by tracking-based workflows.

When ``from_design`` is true, pySC initializes the injection coordinates and
Twiss parameters from the design lattice at the start of the ring. Individual
values can still be overridden explicitly.

Injection errors are configured differently from the other commissioning
errors. Magnet, BPM, RF, and support errors reference values from
``error_table``; injection errors must be written directly in the ``injection``
section with fields such as ``x_error_syst`` or ``x_error_stat``.

.. code-block:: yaml

   injection:
     from_design: true
     emit_x: 1e-9
     emit_y: 1e-11
     bunch_length: 3e-3
     energy_spread: 1e-3
     n_particles: 1000
     x_error_syst: 1e-4
     px_error_stat: 2e-5

Supported injection options are:

``from_design``
   Boolean flag. When true, initialize coordinates and Twiss parameters from
   the design lattice. Defaults to true.

``x``, ``px``, ``y``, ``py``, ``tau``, ``delta``
   Explicit injected beam coordinates.

``betx``, ``alfx``, ``bety``, ``alfy``
   Explicit injected beam Twiss parameters.

``emit_x`` and ``emit_y``
   Horizontal and vertical geometric emittances.

``bunch_length`` and ``energy_spread``
   Longitudinal beam size parameters.

``n_particles``
   Number of particles used when generating injected bunches.

``x_error_syst``, ``px_error_syst``, ``y_error_syst``, ``py_error_syst``,
   ``tau_error_syst``, ``delta_error_syst``
   Systematic injection errors.

``x_error_stat``, ``px_error_stat``, ``y_error_stat``, ``py_error_stat``,
   ``tau_error_stat``, ``delta_error_stat``
   Shot-to-shot injection jitter.

8. Define tuning families
-------------------------

``tuning``
   Tells pySC which declared magnet controls should be used by correction and
   tuning algorithms.

The tuning section does not create new controls. It selects from controls
already declared under ``magnets``. The keys on the left, such as
``sextupoles`` and ``quadrupoles``, must match groups declared under
``magnets``. The values, such as ``B1L`` and ``B2``, must match components
declared in those groups.

.. code-block:: yaml

   tuning:
     HCORR:
       - sextupoles: B1
       - correctors: B1
     VCORR:
       - sextupoles: A1
       - correctors: A1

     model_RM_folder: ./model_RM

     multipoles:
       - sextupoles: B3
       - octupoles: B4

     bba_magnets:
       - quadrupoles: B2

     tune:
       controls_1:
         regex: ^QD2
         component: B2
       controls_2:
         regex: ^QF1
         component: B2

     c_minus:
       controls:
         - sextupoles: A2

``HCORR`` and ``VCORR``
   Horizontal and vertical corrector families. These are lists because a machine
   may use several magnet families as correctors. The selected controls are
   sorted by lattice position.

``model_RM_folder``
   Folder used when loading or saving model response matrices.

``multipoles``
   Magnet controls used as nonlinear multipoles, for example sextupole ``B3``
   and octupole ``B4`` components.

``bba_magnets``
   Magnet controls used for beam-based alignment workflows. These are usually
   quadrupole ``B2`` controls, and can be split across multiple quadrupole
   families.

``tune``
   Defines two quadrupole control families for tune correction. Each family uses
   a ``regex`` or ``mapping`` selector plus a ``component``. The resulting
   controls must already exist from the ``magnets`` section.

``c_minus``
   Defines skew-quadrupole-like controls used for coupling correction. As with
   the other tuning families, the listed magnet groups and components must have
   been declared under ``magnets``.

The HMBA example configuration is a compact template:

.. literalinclude:: ../../../examples/hmba_config.yaml
   :language: yaml
   :caption: examples/hmba_config.yaml
