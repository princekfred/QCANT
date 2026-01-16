Developer Guide
===============

This page describes basic development workflows for QCANT.

Local development
-----------------
Create an editable install:

.. code-block:: bash

	pip install -e .

Running tests
-------------
QCANT uses ``pytest``.

.. code-block:: bash

	python -m pytest

Building documentation
----------------------
The documentation is built with Sphinx.

From the ``docs`` directory:

.. code-block:: bash

	make html

The generated site will be under ``docs/_build/html``.
