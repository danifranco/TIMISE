# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.bal_common
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
Module defining common functionalities to 'bal_*' stuffs.
"""

from ctypes import pointer

__all__ = [
    'bal_mystr_c_ptr',
    'bal_mystr_c_struct'
]


def bal_mystr_c_ptr(c_or_bal_mystr, cBalMyStr, BalMyStr):
    """Return a pointer to instance of cBalMyStr or BalMyStr.

    Parameters
    ----------
    c_or_bal_mystr : cBalImage, cBalMatrix, BalImage or BalMatrix
        C or Python instance to return as a pointer
    cBalMyStr : cBalImage or cBalMatrix
        ``Structure`` matrix or image class
    BalMyStr : BalImage or BalMatrix
        Python matrix or image class

    Returns
    -------
    pointer
        a pointer to instance of cBalMyStr or BalMyStr

    Raises
    ------
    NotImplementedError
        if input ``c_or_bal_mystr`` is not a known class (cBalImage, cBalMatrix,
        BalImage or BalMatrix) or ``None``

    """
    if c_or_bal_mystr is None:
        return
    elif isinstance(c_or_bal_mystr, cBalMyStr):
        return pointer(c_or_bal_mystr)
    elif isinstance(c_or_bal_mystr, BalMyStr):
        return c_or_bal_mystr.c_ptr
    else:
        return NotImplementedError


def bal_mystr_c_struct(c_or_bal_mystr, cBalMyStr, BalMyStr):
    """Return an instance of cBalMyStr or BalMyStr.

    Parameters
    ----------
    c_or_bal_mystr : cBalImage, cBalMatrix, BalImage or BalMatrix
        C or Python instance to return
    cBalMyStr : cBalImage or cBalMatrix
        ``Structure`` matrix or image class
    BalMyStr : BalImage or BalMatrix
        Python matrix or image class

    Returns
    -------
    pointer
        an instance of cBalMyStr or BalMyStr

    Raises
    ------
    NotImplementedError
        if input ``c_or_bal_mystr`` is not a known class (cBalImage, cBalMatrix,
        BalImage or BalMatrix) or ``None``

    """
    if c_or_bal_mystr is None:
        return
    elif isinstance(c_or_bal_mystr, cBalMyStr):
        return c_or_bal_mystr
    elif isinstance(c_or_bal_mystr, BalMyStr):
        return c_or_bal_mystr.c_struct
    else:
        return NotImplementedError
