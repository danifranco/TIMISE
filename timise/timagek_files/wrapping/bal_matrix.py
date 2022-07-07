# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.bal_matrix
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

import logging
import numpy as np
from ctypes import pointer

from ..util import check_type
from .clib import libblockmatching, c_stdout
from .bal_common import bal_mystr_c_ptr, bal_mystr_c_struct
from .c_bal_matrix import cBalMatrix

__all__ = [
    'bal_matrix_c_ptr',
    'bal_matrix_c_struct',
    'free_bal_matrix',
    'np_array_to_bal_matrix_fields',
    'init_c_bal_matrix',
    'allocate_c_bal_matrix',
    'new_c_bal_matrix',
    'bal_matrix_to_np_array',
    'np_array_to_c_bal_matrix',
    'BalMatrix'
]


def bal_matrix_c_ptr(c_or_bal_matrix):
    """Return a pointer to instance of cBalMatrix.

    Parameters
    ----------
    c_or_bal_matrix : cBalMatrix
        instance from which to get the pointer

    Returns
    -------
    c_ptr : pointer
        pointer to cBalMatrix instance

    Example
    -------
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_c_ptr
    >>> c_ptr = bal_matrix_c_ptr(c_or_bal_matrix)
    """
    return bal_mystr_c_ptr(c_or_bal_matrix, cBalMatrix, BalMatrix)


def bal_matrix_c_struct(c_or_bal_matrix):
    """Return an instance of cBalMatrix.

    Parameters
    ----------
    c_or_bal_matrix : cBalMatrix
        instance from which to get the pointer

    Returns
    -------
    bal_matrix : cBalMatrix
        cBalMatrix instance

    Example
    -------
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_c_struct
    >>> c_struct = bal_matrix_c_struct(c_or_bal_matrix)
    """
    return bal_mystr_c_struct(c_or_bal_matrix, cBalMatrix, BalMatrix)


def free_bal_matrix(c_or_bal_matrix):
    """Memory deallocation.

    Parameters
    ----------
    c_or_bal_matrix : cBalMatrix
        instance to deallocate from memory

    Example
    -------
    >>> from timagetk.wrapping.bal_matrix import free_bal_matrix
    >>> free_bal_matrix(c_or_bal_matrix)
    """
    c_ptr = bal_matrix_c_ptr(c_or_bal_matrix)
    if c_ptr:
        libblockmatching._free_mat(c_ptr)  # free structure
        del c_or_bal_matrix
    else:
        logging.warning("Could not retrieve pointer!")


def np_array_to_bal_matrix_fields(np_array):
    """Get ``cBalMatrix`` fields 'l' & 'c' from numpy array.

    'l' & 'c' fields correspond to number of lines and columns, respectively.

    Parameters
    ----------
    np_array : numpy.array
        array to use to get cBalMatrix fields

    Returns
    -------
    dict
        {"c": number of columns, "l": number of lines}

    Example
    -------.
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_matrix import np_array_to_bal_matrix_fields
    >>> arr = np.random.random_sample((4, 2))
    >>> np_array_to_bal_matrix_fields(arr)
    {'c': 2, 'l': 4}

    Raises
    ------
    TypeError
        If input ``np_array`` is not a ``numpy.array`` instance.

    """
    check_type(np_array, 'np_array', np.ndarray)

    # - Get array shape:
    l, c = np_array.shape
    kwargs = {"c": c, "l": l}
    return kwargs


def init_c_bal_matrix(c_bal_matrix, **kwargs):
    """Initialization of a ``cBalMatrix``.

    Here initialization mean setting ``cBalMatrix`` attributes 'l' & 'c'.

    Parameters
    ----------
    c_bal_matrix : cBalMatrix
        object to initialize

    Other Parameters
    ----------------
    l : int
        number of lines
    c : int
        number of columns

    Example
    -------
    >>> from timagetk.wrapping import cBalMatrix
    >>> from timagetk.wrapping.bal_matrix import init_c_bal_matrix
    >>> c_bal_matrix = cBalMatrix()
    >>> init_c_bal_matrix(c_bal_matrix, l=4, c=4)
    >>> c_bal_matrix.l
    4
    >>> c_bal_matrix.c
    4

    Raises
    ------
    TypeError
        If input ``c_bal_matrix`` is not a ``cBalMatrix`` instance.

    """
    check_type(c_bal_matrix, 'c_bal_matrix', cBalMatrix)

    # - Update cBalMatrix attributes:
    # c_bal_matrix.l, c_bal_matrix.c = kwargs.get('l'), kwargs.get('c')
    c_bal_matrix.l, c_bal_matrix.c = kwargs.get('l', 0), kwargs.get('c', 0)


def allocate_c_bal_matrix(c_bal_matrix, np_array):
    """Memory allocation of a ``cBalMatrix`` from a numpy array.

    Parameters
    ----------
    c_bal_matrix : cBalMatrix
        instance to allocate, this object will contain the numpy array
    np_array : numpy array
        object to allocate in a cBalMatrix instance

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping import cBalMatrix
    >>> from timagetk.wrapping.bal_matrix import init_c_bal_matrix
    >>> from timagetk.wrapping.bal_matrix import allocate_c_bal_matrix
    >>> c_bal_matrix = cBalMatrix()
    >>> arr = np.random.random_sample((4, 2))
    >>> allocate_c_bal_matrix(c_bal_matrix, arr)
    >>> c_bal_matrix.l
    4
    >>> c_bal_matrix.c
    2
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_to_np_array
    >>> bal_matrix_to_np_array(c_bal_matrix)
    array([[ 0.98157971,  0.86950611],
           [ 0.03191847,  0.82637903],
           [ 0.48722989,  0.80329761],
           [ 0.51643085,  0.99272212]])
    >>> # Test equality of given ``numpy.array`` to allocated ``cBalMatrix``
    >>> np.array_equal(arr, bal_matrix_to_np_array(c_bal_matrix))
    True

    Raises
    ------
    TypeError
        If input ``c_bal_matrix`` is not a ``cBalMatrix`` instance.

    """
    check_type(c_bal_matrix, 'c_bal_matrix', cBalMatrix)
    try:
        np_array = np.array(np_array)
    except:
        check_type(np_array, 'np_array', np.ndarray)

    # - Initialize the ``cBalMatrix`` if necessary:
    if c_bal_matrix.l == 0 or c_bal_matrix.c == 0:
        kwd = np_array_to_bal_matrix_fields(np_array)
        init_c_bal_matrix(c_bal_matrix, **kwd)

    # - Get cBalMatrix attributes 'l' & 'c':
    l, c = c_bal_matrix.l, c_bal_matrix.c
    libblockmatching._alloc_mat(pointer(c_bal_matrix), l, c)

    # - Fill matrix
    for i in range(l):
        for j in range(c):
            c_bal_matrix.m[j + i * c] = np_array[i, j]


def new_c_bal_matrix(**kwargs):
    """Create a ``cBalMatrix`` instance using kwargs.

    Other Parameters
    ----------------
    l : int
        number of lines
    c : int
        number of columns

    Returns
    -------
    c_bal_matrix : cBalMatrix
        a new cBalMatrix instance

    Example
    -------
    >>> from timagetk.wrapping.bal_matrix import new_c_bal_matrix
    >>> c_bal_matrix = new_c_bal_matrix()
    >>> c_bal_matrix.l
    0
    >>> c_bal_matrix = new_c_bal_matrix(l=2, c=4)
    >>> c_bal_matrix.l
    2
    >>> c_bal_matrix.c
    4
    """
    c_bal_matrix = cBalMatrix()
    init_c_bal_matrix(c_bal_matrix, **kwargs)
    return c_bal_matrix


def np_array_to_c_bal_matrix(np_array, **kwargs):
    """Convert ``numpy.array`` to ``cBalMatrix``.

    Parameters
    ----------
    np_array : numpy.array
        numpy array to convert as a ``cBalMatrix``

    Other Parameters
    ----------------
    l : int
        number of lines
    c : int
        number of columns

    Returns
    -------
    c_bal_matrix : cBalMatrix
        converted ``numpy.array`` instance

    Notes
    -----
    Keyword arguments defined in 'other parameters' will be updated by those
    found by inspection of given ``np_array`` with
    ``np_array_to_bal_matrix_fields``.

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_matrix import np_array_to_c_bal_matrix
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_to_np_array
    >>> arr = np.random.random_sample((4, 2))
    >>> arr
    array([[ 0.35506291,  0.99067097],
           [ 0.85754132,  0.79715911],
           [ 0.57996506,  0.49254318],
           [ 0.21778762,  0.57746731]])
    >>> c_bal_matrix = np_array_to_c_bal_matrix(arr)
    >>> c_bal_matrix.l
    2
    >>> c_bal_matrix.c
    4
    >>> bal_matrix_to_np_array(c_bal_matrix)
    array([[ 0.35506291,  0.99067097],
           [ 0.85754132,  0.79715911],
           [ 0.57996506,  0.49254318],
           [ 0.21778762,  0.57746731]])

    Raises
    ------
    TypeError
        If input ``np_array`` is not a ``numpy.array`` instance.

    """
    # - Get the 'l' & 'c' fields to create a cBalMatrix:
    np_array_kwargs = np_array_to_bal_matrix_fields(np_array)
    # - Update the keyword arguments with previously obtained fields:
    kwargs.update(np_array_kwargs)
    # - Use it to create a new cBalMatrix and allocate the array:
    c_bal_matrix = new_c_bal_matrix(**kwargs)
    allocate_c_bal_matrix(c_bal_matrix, np_array)
    return c_bal_matrix


def bal_matrix_to_np_array(c_or_bal_matrix, dtype=None):
    """Convert cBalMatrix to numpy array.

    Parameters
    ----------
    c_or_bal_matrix : cBalMatrix
        C object to convert into ``numpy.array``
    dtype : str|np.dtype
        type of the numpy array to return, `np.float16` by default

    Returns
    -------
    np_array : numpy.array
        converted array

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_to_np_array
    >>> from timagetk.wrapping.bal_matrix import np_array_to_c_bal_matrix
    >>> arr = np.random.random_sample((4, 2))
    >>> arr
    array([[ 0.80724603,  0.33081692],
           [ 0.20047332,  0.76125283],
           [ 0.91918492,  0.99985815],
           [ 0.90476838,  0.85771965]])
    >>> c_bal_matrix = np_array_to_c_bal_matrix(arr)
    >>> bal_matrix_to_np_array(c_bal_matrix)
    array([[ 0.80724603,  0.33081692],
           [ 0.20047332,  0.76125283],
           [ 0.91918492,  0.99985815],
           [ 0.90476838,  0.85771965]])
    """
    if dtype is None:
        dtype = np.float16

    m = bal_matrix_c_struct(c_or_bal_matrix)
    l = m.l
    c = m.c
    np_array = np.zeros((l, c), dtype=dtype)
    # Fill matrix
    for i in range(l):
        for j in range(c):
            np_array[i, j] = m.m[j + i * c]
    return np_array


class BalMatrix(object):
    """Class representing matrix objects.

    Parameters
    ----------
    np_array : numpy.array, optional
        numpy array to use when creating ``BalMatrix`` instance
    c_bal_matrix : cBalMatrix, optional
        C-instance to use when creating ``BalMatrix`` instance

    Other Parameters
    ----------------
    l : int
        number of lines
    c : int
        number of columns

    Notes
    -----
        Other parameters are used by:

        * ``np_array_to_c_bal_matrix`` if ``np_array`` is given;
        * ``new_c_bal_matrix`` if no parameters are given;

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping import BalMatrix
    >>> # Manually create identity quaternion:
    >>> arr = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    >>> # Initialize from ``numpy.array`` instance:
    >>> bmat = BalMatrix(np_array=arr)
    >>> bmat.to_np_array()
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> # Initialize from ``cBalMatrix`` instance:
    >>> from timagetk.wrapping.bal_matrix import np_array_to_c_bal_matrix
    >>> cbmat = np_array_to_c_bal_matrix(arr)
    >>> bmat = BalMatrix(c_bal_matrix=cbmat)
    >>> bmat.to_np_array()
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    """

    def __init__(self, np_array=None, c_bal_matrix=None, **kwargs):
        """BalMatrix object constructor. """
        if np_array is not None:
            logging.debug("Initializing BalMatrix from numpy.array...")
            self._np_array = np_array
            self._c_bal_matrix = np_array_to_c_bal_matrix(self._np_array,
                                                        **kwargs)
        elif c_bal_matrix is not None:
            logging.debug("Initializing BalMatrix from cBalMatrix...")
            self._c_bal_matrix = c_bal_matrix
            self._np_array = self.to_np_array()
        else:
            logging.debug("Initializing empty BalMatrix object from keyword arguments...")
            self._c_bal_matrix = new_c_bal_matrix(**kwargs)
            self._np_array = self.to_np_array()

    def __del__(self):
        """Class destructor. """
        self.free()

    def __eq__(self, other):
        """Test equality between self and ``other``. """
        if isinstance(other, self.__class__):
            other = other.to_np_array()
        elif isinstance(other, np.ndarray):
            pass
        else:
            return False
        return np.array_equal(self.to_np_array(), other)

    def __ne__(self, other):
        """Test non-equality between self and ``other``. """
        return not self.__eq__(other)

    @property
    def c_ptr(self):
        """Get the pointer to the C object ``cBalMatrix``."""
        return pointer(self._c_bal_matrix)

    @property
    def c_struct(self):
        """Get the ``cBalMatrix`` hidden attribute.

        Returns
        -------
        cBalMatrix
            the hidden attribute ``_c_bal_matrix``
        """
        return self._c_bal_matrix

    def free(self):
        """Free memory allocated to object. """
        if self._c_bal_matrix:
            free_bal_matrix(self._c_bal_matrix)
        self._c_bal_matrix = None

    def c_display(self, name=''):
        """Print information about the object as found in the allocated memory.

        Parameters
        ----------
        name : str
            name of the transformation, used for printing

        Returns
        -------
        str
            information about the BalMatrix object
        """
        libblockmatching._print_mat(c_stdout, self.c_ptr, name)

    def to_np_array(self, dtype=None):
        """Convert to a numpy array.

        Parameters
        ----------
        dtype : str|np.dtype
            type of the numpy array to return, `np.float16` by default

        Returns
        -------
        numpy.array
            converted instance
        """
        return bal_matrix_to_np_array(self._c_bal_matrix, dtype=dtype)


"""
/* substract two matrix */
cBalMatrix sub_mat(cBalMatrix m1, cBalMatrix m2);

/* transpose matrix */
cBalMatrix transpose(cBalMatrix m);

/* det matrix */
double det(cBalMatrix mat);

/* inverse matrix */
cBalMatrix inv_mat(cBalMatrix m);
"""


def c_bal_matrix_transpose(c_bal_matrix):
    libblockmatching.transpose.argtypes = [cBalMatrix]
    libblockmatching.transpose.restype = cBalMatrix
    c_bal_matrix_out = libblockmatching.transpose(c_bal_matrix)
    return c_bal_matrix_out


def np_array_transpose(np_array):
    c_bal_matrix = np_array_to_c_bal_matrix(np_array)
    c_bal_matrix_out = c_bal_matrix_transpose(c_bal_matrix)
    return bal_matrix_to_np_array(c_bal_matrix_out)
