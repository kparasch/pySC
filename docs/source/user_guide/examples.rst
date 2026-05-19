Examples
========

Core setup and response matrices
--------------------------------

The first example loads the HMBA lattice fixture, builds a pySC
``SimulatedCommissioning`` object from ``hmba_config.yaml``, and computes the
ideal orbit and trajectory response matrices. The generated JSON files are saved
in ``examples/model_RM/`` and can be reused by later correction and BBA examples.
Run the examples from the ``examples/`` directory.

Download the script: :download:`01_hmba_response_matrices.py <../../../examples/01_hmba_response_matrices.py>`

.. literalinclude:: ../../../examples/01_hmba_response_matrices.py
   :language: python
   :caption: examples/01_hmba_response_matrices.py

Correct injection over one turn
-------------------------------

This example loads the HMBA configuration, applies a small random setpoint to
one horizontal and one vertical corrector, and corrects the injection over one
turn. The call to ``correct_injection(n_turns=1)`` loads ``model_RM/trajectory1.json``
from the configured ``model_RM_folder``.

Download the script: :download:`02_correct_injection_1turn.py <../../../examples/02_correct_injection_1turn.py>`

.. literalinclude:: ../../../examples/02_correct_injection_1turn.py
   :language: python
   :caption: examples/02_correct_injection_1turn.py

Orbit BBA for the first 10 BPMs
-------------------------------

This example loads the HMBA configuration, generates the orbit BBA configuration
from ``model_RM/orbit.json``, and runs orbit beam-based alignment only for the
first 10 BPMs.

Download the script: :download:`03_orbit_bba_first_10_bpms.py <../../../examples/03_orbit_bba_first_10_bpms.py>`

.. literalinclude:: ../../../examples/03_orbit_bba_first_10_bpms.py
   :language: python
   :caption: examples/03_orbit_bba_first_10_bpms.py
