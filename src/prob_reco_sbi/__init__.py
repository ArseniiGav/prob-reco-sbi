"""Probabilistic event reconstruction with simulation-based inference: code for a course.

Subpackages
-----------
flash
    "A Flash in the Dark": photon-transport simulation of a toy 2D scintillation detector and its data format.
flows1d
    Normalizing flows in one dimension, from scratch: transforms, training by maximum likelihood, exact diagnostics.
npe
    Neural posterior estimation: conditional spline flows of the event parameters, their training and the exact
    posteriors of the toy where it has a likelihood formula.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
