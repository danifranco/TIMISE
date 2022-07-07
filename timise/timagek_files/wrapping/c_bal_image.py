# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.c_bal_image
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
Module defining cBalImage object ``ctypes`` structure.
"""

from ctypes import Structure
from ctypes import c_size_t, c_uint, c_void_p, c_char_p, c_float

__all__ = ['cBalImage']
typeVoxelSize = c_float


class cBalImage(Structure):
    """Image ``ctypes`` structure as defined in ``bal-image.h``.

    Definition of ``cBalImage`` structure derived from the ``Structure`` class
    defined in the ``ctypes`` module.

    Notes
    -----
        See ``build-scons/include/blockmatching/bal-image.h``:

    .. code-block:: cpp

        typedef struct {
          size_t ncols;         /* Number of columns (X dimension) */
          size_t nrows;         /* Number of rows (Y dimension) */
          size_t nplanes;       /* Number of planes (Z dimension) */
          size_t vdim;          /* Vector size */
          bufferType c_uint;
          void *data;        /* Generic pointer on image data buffer.
                                This pointer has to be casted in proper data type
                                depending on the type field */
          void ***array;     /* Generic 3D array pointing on each image element.
                                This pointer has to be casted in proper data type
                                depending on the type field */

          typeVoxelSize vx;          /* real voxel size in X dimension */
          typeVoxelSize vy;          /* real voxel size in Y dimension */
          typeVoxelSize vz;          /* real voxel size in Z dimension */

          char *name;
        } bal_image;

    Attributes
    ----------
    ncols : c_size_t
        Number of columns (X dimension)
    nrows : c_size_t
        Number of rows (Y dimension)
    nplanes : c_size_t
        Number of planes (Z dimension)
    vdim : c_size_t
        Vector size
    type : c_uint
        bufferType
    data : c_void_p
        Generic pointer on image data buffer
    array : c_void_p
        Generic 3D array pointing on each image element
    vx : typeVoxelSize
        Real voxel size in X dimension
    vy : typeVoxelSize
        Real voxel size in Y dimension
    vz : typeVoxelSize
        Real voxel size in Z dimension
    name : c_char_p
        name of the image

    """
    _fields_ = [
        ("ncols", c_size_t),
        ("nrows", c_size_t),
        ("nplanes", c_size_t),
        ("vdim", c_size_t),
        ("type", c_uint),
        ("data", c_void_p),
        ("array", c_void_p),
        ("vx", typeVoxelSize),
        ("vy", typeVoxelSize),
        ("vz", typeVoxelSize),
        ("name", c_char_p)
    ]
