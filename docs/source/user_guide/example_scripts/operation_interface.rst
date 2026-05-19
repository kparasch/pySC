Operational interface
=====================

Operational measurements use a small interface object to connect pySC to a
machine or control system. The interface owns all hardware-specific details:
reading orbits, reading setpoints, writing setpoints, waiting for devices to
settle, and triggering turn-by-turn acquisition when needed.

The base methods required by pySC operational tools are:

``get_orbit()``
   Return the current horizontal and vertical orbit arrays.

``get(name)``
   Return one magnet strength in physics units.

``set(name, value)``
   Set one magnet strength in physics units and wait until it is settled.

``get_many(names)``
   Return a dictionary mapping magnet names to strengths.

``set_many(data)``
   Set many magnet strengths from a dictionary and wait until they are settled.

Some workflows need extra methods. Orbit correction against an operational
reference uses ``get_ref_orbit()``. RF correction uses
``get_rf_main_frequency()`` and ``set_rf_main_frequency()``.

For first-turn correction, ``InterfaceInjection.get_orbit()`` should return the
turn-by-turn trajectory flattened with ``order="F"``. Its ``get_ref_orbit()``
method should return a reference trajectory with the same shape and flattening.

Download the skeleton: :download:`interface.py <../../../../examples/operation/interface.py>`

.. literalinclude:: ../../../../examples/operation/interface.py
   :language: python
   :caption: examples/operation/interface.py
