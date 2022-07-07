# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.c_bal_matrix
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#
#       File contributor(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       Distributed under the INRIA License.
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------


"""
Module defining cBalMatrix object ``ctypes`` structure.
"""

from ctypes import Structure, c_int, c_double, POINTER

__all__ = ['cBalMatrix']


class cBalMatrix(Structure):
    """Matrix ``ctypes`` structure as defined in ``bal-matrix.h``.

    Definition of ``cBalMatrix`` structure derived from the ``Structure`` class
    defined in the ``ctypes`` module.

    Notes
    -----
    See ``build-scons/include/blockmatching/bal-matrix.h``:

    .. code-block:: cpp

        typedef struct {
          int l,c;
          double *m;
        } _MATRIX;

    Attributes
    ----------
    l : c_int
        number of rows
    c : c_int
        number of column
    m : POINTER(c_double)
        array defining the matrix

    """
    _fields_ = [
        ("l", c_int),
        ("c", c_int),
        ("m", POINTER(c_double))
    ]
