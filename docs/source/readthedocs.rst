Read the Docs setup
===================

The canonical documentation project should use the Read the Docs slug
``accelerator-commissioning``:

.. code-block:: text

   https://accelerator-commissioning.readthedocs.io/

The repository contains ``.readthedocs.yaml`` so Read the Docs can install the
package with the ``doc`` optional dependency group and build this Sphinx
documentation from ``docs/source/conf.py``.

Optional ``pysc`` redirect project
----------------------------------

If the ``pysc`` project slug is available on Read the Docs, create a second
Read the Docs project that points to the same GitHub repository and configure it
as a redirect-only compatibility URL.

In the ``pysc`` project dashboard, add this redirect:

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - Type
     - ``Exact Redirect``
   * - From URL
     - ``/*``
   * - To URL
     - ``https://accelerator-commissioning.readthedocs.io/:splat``
   * - Force redirect
     - Enabled

This preserves nested paths such as ``/en/latest/`` while sending users to the
canonical documentation domain.
