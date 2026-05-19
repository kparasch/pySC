Correct injection over one turn
===============================

This example loads the HMBA configuration, applies a small random setpoint to
one horizontal and one vertical corrector, and corrects the injection over one
turn. The call to ``correct_injection(n_turns=1)`` loads
``model_RM/trajectory1.json`` from the configured ``model_RM_folder``.

Download the script: :download:`02_correct_injection_1turn.py <../../../../examples/02_correct_injection_1turn.py>`

.. literalinclude:: ../../../../examples/02_correct_injection_1turn.py
   :language: python
   :caption: examples/02_correct_injection_1turn.py
