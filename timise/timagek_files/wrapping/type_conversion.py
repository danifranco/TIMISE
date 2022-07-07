# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.type_conversion
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

import numpy as np
from ctypes import c_ubyte
from ctypes import c_byte
from ctypes import c_short
from ctypes import c_ushort
from ctypes import c_uint
from ctypes import c_int
from ctypes import c_ulong
from ctypes import c_long
from ctypes import c_float
from ctypes import c_double

from .vt_typedefs import ImageType

__all__ = [
    'np_type_to_vt_type',
    'vt_type_to_np_type',
    'vt_type_to_c_type',
    'morpheme_type_arg'
]

VT_TO_ARG = {
    ImageType.SCHAR: "-o 1 -s",
    ImageType.UCHAR: "-o 1",
    ImageType.SSHORT: "-o 2 -s",
    ImageType.USHORT: "-o 2",
    ImageType.SINT: "-o 4 -s",
    ImageType.ULINT: "-o 4",
    ImageType.FLOAT: "-r",
}


def np_type_to_vt_type(dtype):
    """Convert ``numpy.dtype`` into known VT types.

    Notes
    -----
    Known VT types are defined as attributes in class ``ImageType``.

    Parameters
    ----------
    dtype : numpy.dtype
        ``numpy.dtype`` to convert

    Returns
    -------
    _type : ctypes
        known VT type

    Raises
    ------
    TypeError
        if given ``dtype`` is not mapped to a known ``ImageType`` attributes.

    """
    if dtype == np.uint8:
        _type = ImageType.UCHAR
    elif dtype == np.int8:
        _type = ImageType.SCHAR
    elif dtype == np.int16:
        _type = ImageType.SSHORT
    elif dtype == np.uint16:
        _type = ImageType.USHORT
    elif dtype == np.uint32:
        _type = ImageType.UINT
    elif dtype == np.int32:
        _type = ImageType.SINT
    elif dtype == np.uint64:
        _type = ImageType.ULINT
    elif dtype == np.int64:
        _type = ImageType.SLINT
    elif dtype == np.float32:
        _type = ImageType.FLOAT
    elif dtype == np.float64:
        _type = ImageType.DOUBLE
    #     elif dtype == np.float128: #         _type = ImageType.LONG_DO
    else:
        _type = ImageType.TYPE_UNKNOWN
    return _type


def vt_type_to_np_type(vt_type):
    """Convert ``vt_type`` into ``numpy.dtype``.

    Notes
    -----
    Known VT types are defined in class ``ImageType``.

    Parameters
    ----------
    vt_type : ImageType
        vt type to convert

    Returns
    -------
    numpy.dtype
        known dtype to use with numpy arrays

    Raises
    ------
    TypeError
        if given ``vt_type`` is not mapped to a known ``numpy.dtype``.

    """
    if vt_type == ImageType.UCHAR:
        return np.uint8
    elif vt_type == ImageType.SCHAR:
        return np.int8
    elif vt_type == ImageType.SSHORT:
        return np.int16
    elif vt_type == ImageType.USHORT:
        return np.uint16
    elif vt_type == ImageType.UINT:
        return np.uint32
    elif vt_type == ImageType.SINT:
        return np.int32
    elif vt_type == ImageType.ULINT:
        return np.uint64
    elif vt_type == ImageType.SLINT:
        return np.int64
    elif vt_type == ImageType.FLOAT:
        return np.float32
    elif vt_type == ImageType.DOUBLE:
        return np.float64
    # elif vt_type == VOXELTYPE.:     #    self._data.dtype == np.float128
    else:
        raise TypeError("Unknown type '{}'!".format(vt_type))


def vt_type_to_c_type(vt_type):
    """Convert ``vt_type`` into ``ctypes``.

    Notes
    -----
    Known ``ctypes`` are: c_ubyte, c_byte, c_short, c_ushort, c_uint, c_int, c_ulong, c_long, c_float and c_double.

    Parameters
    ----------
    vt_type : ImageType
        vt type to convert

    Returns
    -------
    ctypes
        known ctypes

    Raises
    ------
    TypeError
        if given ``vt_type`` is not mapped to a known ``ctypes``.

    """
    if vt_type == ImageType.UCHAR:
        return c_ubyte
    elif vt_type == ImageType.SCHAR:
        return c_byte
    elif vt_type == ImageType.SSHORT:
        return c_short
    elif vt_type == ImageType.USHORT:
        return c_ushort
    elif vt_type == ImageType.UINT:
        return c_uint
    elif vt_type == ImageType.SINT:
        return c_int
    elif vt_type == ImageType.ULINT:
        # return c_int
        return c_ulong
    elif vt_type == ImageType.SLINT:
        # return c_int
        return c_long
    elif vt_type == ImageType.FLOAT:
        return c_float
    elif vt_type == ImageType.DOUBLE:
        return c_double
    # elif vt_type == VOXELTYPE.:     #    self._data.dtype == np.float128
    else:
        raise TypeError("Unknown type '{}'!".format(vt_type))


def morpheme_type_arg(image):
    """Convert type of input ``image`` into arguments for wrapped C libraries.

    Parameters
    ----------
    image: numpy.array or SpatialImage
        image or array to use to convert type

    Returns
    -------
    str
        argument to pass to wrapped C libraries

    Raises
    ------
    TypeError
        if type is not defined in type conversion global variable ``VT_TO_ARG``.

    """
    vt_type = np_type_to_vt_type(image.dtype)
    if vt_type in VT_TO_ARG:
        return VT_TO_ARG[vt_type]
    else:
        raise TypeError("Unknown type '{}'!".format(vt_type))
