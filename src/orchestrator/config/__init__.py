"""Packaged default configuration files.

This subpackage exists so that ``models.yaml`` and ``orchestrator.yaml`` ship
inside the wheel and can be located with :mod:`importlib.resources` rather than
by walking ``__file__`` out of the installed package.
"""
