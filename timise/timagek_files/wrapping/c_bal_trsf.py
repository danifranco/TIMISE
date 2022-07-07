# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.c_bal_trsf
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
Module defining cBalTrsf object ``ctypes`` structure.
"""

from ctypes import Structure, c_int, c_float

from .c_bal_image import cBalImage
from .c_bal_matrix import cBalMatrix
# from timagetk.wrapping import cBalImage
# from timagetk.wrapping import cBalMatrix

__all__ = ['cBalTrsf']


class cBalTrsf(Structure):
    """Transformation ``ctypes`` structure as defined in ``bal-transformation.h``.

    Definition of ``cBalTrsf`` structure derived from the ``Structure`` class
    defined in the ``ctypes`` module.

    Notes
    -----
    See ``build-scons/include/blockmatching/bal-transformation.h``:

    .. code-block:: cpp

        typedef struct {

          enumTypeTransfo type;
          cBalMatrix mat;
          enumUnitTransfo transformation_unit;
          bal_image vx;
          bal_image vy;
          bal_image vz;

          /* for transformation averaging
           */
          float weight;
          float error;

        } bal_transformation;

    Attributes
    ----------
    type : c_int
        type of transformation (see ``TRSF_TYPE_DICT`` in `bal_trsf.py`)
    mat : cBalMatrix
        linear transformation matrix (see ``cBalMatrix`` in `c_bal_matrix.py`)
    transformation_unit : c_int
        unit of transformation matrix (see ``TRSF_UNIT_DICT`` in `bal_trsf.py`)
    vx : cBalImage
        non-linear transformation matrix in x direction (see ``cBalImage`` in `c_bal_image.py`)
    vy : cBalImage
        non-linear transformation matrix in y direction (see ``cBalImage`` in `c_bal_image.py`)
    vz : cBalImage
        non-linear transformation matrix in z direction (see ``cBalImage`` in `c_bal_image.py`)
    weight : c_float
        weight for transformation averaging
    error : c_float
        error in transformation averaging

    """
    _fields_ = [
        ("type", c_int),
        ("mat", cBalMatrix),
        ("transformation_unit", c_int),
        ("vx", cBalImage),  # WARNING: do not confuse with x-voxelsize!
        ("vy", cBalImage),  # WARNING: do not confuse with y-voxelsize!
        ("vz", cBalImage),  # WARNING: do not confuse with z-voxelsize!
        ("weight", c_float),
        ("error", c_float),
    ]
