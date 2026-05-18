Examples
========

Core setup and response matrices
--------------------------------

The first example loads the HMBA lattice fixture, builds a pySC
``SimulatedCommissioning`` object from ``hmba_config.yaml``, and computes the
ideal orbit and trajectory response matrices. The generated JSON files are saved
in ``examples/output/`` and can be reused by later correction and BBA examples.

Download the script: :download:`01_hmba_response_matrices.py <../../../examples/01_hmba_response_matrices.py>`

.. literalinclude:: ../../../examples/01_hmba_response_matrices.py
   :language: python
   :caption: examples/01_hmba_response_matrices.py
