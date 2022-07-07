# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.bal_trsf
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
from .c_bal_trsf import cBalTrsf
from .bal_image import BalImage, spatial_image_to_c_bal_image, allocate_c_bal_image
from .bal_matrix import BalMatrix, np_array_to_c_bal_matrix, allocate_c_bal_matrix 

__all__ = [
    'bal_trsf_c_ptr',
    'bal_trsf_c_struct',
    'free_bal_trsf',
    'BalTrsf', 'TRSF_TYPE_DICT', 'TRSF_UNIT_DICT'
]

KWARGS_DEBUG = "Got keyword argument '{}' with value: {}"

# - These two global variable are defined in bal-stddef.h
TRSF_TYPE_DICT = {
    0: "UNDEF_TRANSFORMATION",
    1: "TRANSLATION_2D",
    2: "TRANSLATION_3D",
    3: "TRANSLATION_SCALING_2D",
    4: "TRANSLATION_SCALING_3D",
    5: "RIGID_2D",
    6: "RIGID_3D",
    7: "SIMILITUDE_2D",
    8: "SIMILITUDE_3D",
    9: "AFFINE_2D",
    10: "AFFINE_3D",
    11: "VECTORFIELD_2D",
    12: "VECTORFIELD_3D",
    13: "SPLINE"
}
# update the global var with inverted dictionary:
TRSF_TYPE_DICT.update({v: k for k, v in TRSF_TYPE_DICT.items()})

TRSF_UNIT_DICT = {
    0: "UNDEF_UNIT",
    1: "VOXEL_UNIT",
    2: "REAL_UNIT"
}
# update the global var with inverted dictionary:
TRSF_UNIT_DICT.update({v: k for k, v in TRSF_UNIT_DICT.items()})

LINEAR_METHODS = [TRSF_TYPE_DICT[i] for i in range(1, 10)]
NON_LINEAR_METHODS = [TRSF_TYPE_DICT[i] for i in range(11, 13)]


def bal_trsf_c_ptr(c_or_bal_trsf):
    """Return a pointer to instance of cBalTrsf.

    Parameters
    ----------
    c_or_bal_trsf : cBalTrsf
        instance to get the pointer from

    Returns
    -------
    pointer
        a pointer to cBalTrsf instance

    Example
    -------
    >>> from timagetk.wrapping.bal_trsf import bal_trsf_c_ptr
    >>> c_ptr = bal_trsf_c_ptr(c_or_bal_trsf)
    """
    if isinstance(c_or_bal_trsf, cBalTrsf):
        return pointer(c_or_bal_trsf)
    else:
        return


def bal_trsf_c_struct(c_or_bal_trsf):
    """Return an instance of cBalTrsf.

    If ``c_or_bal_trsf`` is a ``cBalTrsf``, return it, else returns ``None``.

    Parameters
    ----------
    c_or_bal_trsf : cBalTrsf
        instance to test as cBalTrsf

    Returns
    -------
    cBalTrsf
        cBalTrsf instance

    Example
    -------
    >>> from timagetk.wrapping.bal_trsf import bal_trsf_c_struct
    >>> c_struct = bal_trsf_c_struct(c_or_bal_trsf)
    """
    if isinstance(c_or_bal_trsf, cBalTrsf):
        return c_or_bal_trsf
    else:
        return


def free_bal_trsf(c_or_bal_trsf):
    """Free trsf static attributes, not dynamics vx, vy, vz ant mat

    Parameters
    ----------
    c_or_bal_trsf : cBalTrsf or BalTrsf
        instance to deallocate from memory

    Example
    -------
    >>> from timagetk.wrapping.bal_trsf import free_bal_trsf
    >>> free_bal_trsf(c_or_bal_trsf)
    """
    c_bal_trsf = bal_trsf_c_struct(c_or_bal_trsf)
    if c_bal_trsf is None:
        return
    c_bal_trsf.mat = None
    c_bal_trsf.vx = None
    c_bal_trsf.vy = None
    c_bal_trsf.vz = None

    ptr = pointer(c_bal_trsf)
    libblockmatching.BAL_FreeTransformation(ptr)


def init_c_bal_trsf(c_bal_trsf, **kwargs):
    """Initialization of a ``cBalTrsf``.

    Here initialization mean setting ``cBalTrsf`` attributes described in the
    other parameters section.

    Parameters
    ----------
    c_bal_trsf : cBalTrsf
        object to initialize

    Other Parameters
    ----------------
    mat : numpy.array
        Quaternion describing the linear transformation.
        ``None`` if vectorfield transformation.
    vx, vy, vz : numpy.array
        Displacement matrix in x, y and z for non-linear transformation.
        ``None`` if linear transformation.
    trsf_unit : int
        unit of the transformation, available units are defined in global
        variable ``TRSF_UNIT_DICT``
    trsf_type : int
        type of the transformation, available types are defined in global
        variable ``TRSF_TYPE_DICT``

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping import cBalTrsf
    >>> from timagetk.wrapping.bal_trsf import init_c_bal_trsf
    >>> # Initialize an empty ``cBalTrsf`` instance:
    >>> c_bal_trsf = cBalTrsf()
    >>> init_c_bal_trsf(c_bal_trsf)
    >>> # Initialize an affine ``cBalTrsf`` instance in real units with random matrix:
    >>> rand_mat = np.random.random_sample((4, 4))
    >>> c_bal_trsf = cBalTrsf()
    >>> init_c_bal_trsf(c_bal_trsf, trsf_type="AFFINE_3D", trsf_unit="REAL_UNIT", mat=rand_mat)

    Raises
    ------
    TypeError
        If input ``c_bal_matrix`` is not a ``cBalMatrix`` instance.

    """
    check_type(c_bal_trsf, 'c_bal_trsf', cBalTrsf)

    # - Update cBalTrsf attributes:
    # -- update ``trsf_type``:
    trsf_type = kwargs.get('trsf_type', 0)  # default to "UNDEF"
    if isinstance(trsf_type, str):
        logging.debug(KWARGS_DEBUG.format('trsf_type', trsf_type))
        trsf_type = TRSF_TYPE_DICT[trsf_type]

    check_type(trsf_type, 'trsf_type', int)
    c_bal_trsf.type = trsf_type

    # -- update ``trsf_unit``:
    trsf_unit = kwargs.get('trsf_unit', 0)  # default to "UNDEF"
    if isinstance(trsf_unit, str):
        logging.debug(KWARGS_DEBUG.format('trsf_unit', trsf_unit))
        trsf_unit = TRSF_UNIT_DICT[trsf_unit]

    check_type(trsf_unit, 'trsf_unit', int)
    c_bal_trsf.transformation_unit = trsf_unit

    # - Linear transformation matrix:
    mat = kwargs.get('mat', None)
    if mat is not None:
        c_bal_trsf.mat = np_array_to_c_bal_matrix(mat)

    # - Non-linear transformation image:
    vx = kwargs.get('vx', None)
    vy = kwargs.get('vy', None)
    vz = kwargs.get('vz', None)
    if vx is not None and vy is not None and vz is not None:
        c_bal_trsf.vx = spatial_image_to_c_bal_image(vx)
        c_bal_trsf.vy = spatial_image_to_c_bal_image(vy)
        c_bal_trsf.vz = spatial_image_to_c_bal_image(vz)


def allocate_c_bal_trsf(c_bal_trsf, obj):
    """Memory allocation of a cBalTrsf from a numpy array.

    Notes
    -----
        Require to initialize ``c_bal_trsf`` beforehand with ``init_c_bal_trsf``.

        If a dictionary, should have three keys: 'vx', 'vz', & 'vz'.

    Parameters
    ----------
    c_bal_trsf : cBalTrsf
        Instance to allocate, this object will contain the numpy array
    obj : numpy.array or dict of SpatialImage
        Object to allocate in a cBalTrsf instance.
        If a ``numpy.array`` is provided, create a linear transformation.
        If a dict of ``SpatialImage`` is provided, create a non-linear transformation.

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping import cBalTrsf
    >>> from timagetk.wrapping.bal_trsf import init_c_bal_trsf
    >>> from timagetk.wrapping.bal_trsf import allocate_c_bal_trsf
    >>> c_bal_trsf = cBalTrsf()
    >>> arr = np.random.random_sample((4, 4))
    >>> init_c_bal_trsf(c_bal_trsf, trsf_type="AFFINE_3D", trsf_unit="REAL_UNIT", mat=arr)
    >>> allocate_c_bal_trsf(c_bal_trsf, arr)
    >>> c_bal_trsf.mat.l
    4
    >>> c_bal_trsf.mat.c
    4
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_to_np_array
    >>> bal_matrix_to_np_array(c_bal_trsf.mat)
    array([[ 0.69939593,  0.61132809,  0.00789376,  0.28343718],
           [ 0.07221215,  0.91002345,  0.90069881,  0.67705448],
           [ 0.17028411,  0.07202466,  0.30423757,  0.14721526],
           [ 0.34711198,  0.89955543,  0.13914549,  0.53726173]])
    >>> # Test equality of given ``numpy.array`` to allocated ``cBalMatrix``
    >>> np.array_equal(arr, bal_matrix_to_np_array(c_bal_trsf.mat))
    True

    Raises
    ------
    TypeError
        If input ``c_bal_trsf`` is not a ``cBalTrsf`` instance.

    """
    from ..spatial_image import SpatialImage
    check_type(c_bal_trsf, 'c_bal_trsf', cBalTrsf)

    if isinstance(obj, np.ndarray):
        # Allocate the linear transformation matrix as a cBalMatrix:
        allocate_c_bal_matrix(c_bal_trsf.mat, obj)
    elif isinstance(obj, dict) and all(
            [isinstance(im, SpatialImage) for im in obj.values()]):
        # Allocate the non-linear transformation matrix as three cBalImage:
        for name, img in obj.items():
            if name == "vx":
                allocate_c_bal_image(c_bal_trsf.vx, img)
            if name == "vy":
                allocate_c_bal_image(c_bal_trsf.vy, img)
            if name == "vz":
                allocate_c_bal_image(c_bal_trsf.vz, img)
    else:
        raise TypeError("Unknown input ``obj`` type ({})!".format(type(obj)))


def new_c_bal_trsf(**kwargs):
    """Create a ``cBalTrsf`` instance using kwargs.

    Other Parameters
    ----------------
    mat : numpy.array
        Quaternion describing the linear transformation.
        ``None`` if vectorfield transformation.
    vx, vy, vz : numpy.array
        Displacement matrix in x, y and z for non-linear transformation.
        ``None`` if linear transformation.
    trsf_unit : int
        unit of the transformation, available units are defined in TRSF_UNIT_DICT
    trsf_type : int
        type of the transformation, available types are defined in
        TRSF_TYPE_DICT

    Returns
    -------
    c_bal_trsf : cBalTrsf
        a new cBalTrsf instance

    Example
    -------
    >>> from timagetk.wrapping.bal_trsf import new_c_bal_trsf
    >>> # Initialise empty ``cBalTrsf`` instance:
    >>> c_bal_trsf = new_c_bal_trsf()
    >>> c_bal_trsf.mat.l
    0
    >>> # Initialize an affine ``cBalTrsf`` instance in real units with random matrix:
    >>> arr = np.random.random_sample((4, 4))
    >>> c_bal_trsf = new_c_bal_trsf(trsf_type="AFFINE_3D", trsf_unit="REAL_UNIT", mat=arr)
    >>> # Test equality of given ``numpy.array`` to allocated ``cBalMatrix``
    >>> from timagetk.wrapping.bal_matrix import bal_matrix_to_np_array
    >>> np.array_equal(arr, bal_matrix_to_np_array(c_bal_trsf.mat))
    True

    """
    c_bal_trsf = cBalTrsf()
    init_c_bal_trsf(c_bal_trsf, **kwargs)
    return c_bal_trsf


def np_array_to_rigid_trsf(np_array, trsf_unit="REAL_UNIT"):
    """Create a rigid ``BalTrsf`` from a numpy array.

    The numpy array must be a quaternion describing a rigid transformation

    Parameters
    ----------
    np_array : numpy.array
        quaternion describing the rigid transformation
    trsf_unit : int or str, optional
        defines if the quaternion in expressed in real ("REAL_UNIT"), by default,
        or voxel units ("VOXEL_UNIT")

    Returns
    -------
    BalTrsf
        rigid transformation

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_trsf import np_array_to_rigid_trsf
    >>> arr = np.random.random_sample((4, 4))
    >>> trsf = np_array_to_rigid_trsf(arr)
    >>> print(trsf.get_unit())
    'REAL_UNIT'
    >>> print(trsf.get_type())
    'RIGID_3D'
    >>> print(trsf.is_linear())
    True
    >>> trsf.c_display('test')
    type of 'test' is RIGID_3D
    transformation unit is in real units
       0.438030221014404    0.357154839192758    0.227787771639948    0.031846050398104
       0.239656746138632    0.498627779703492    0.768720227202931    0.449667299588282
       0.587327485769069    0.145162665186915    0.424790275453152    0.005316127646503
       0.293384023207115    0.532391710948782    0.853227217373506    0.457796981063500

    """
    c_bal_trsf = new_c_bal_trsf(trsf_unit=trsf_unit, trsf_type="RIGID_3D",
                                mat=np_array)
    allocate_c_bal_trsf(c_bal_trsf, np_array)
    return BalTrsf(c_bal_trsf=c_bal_trsf)


def np_array_to_affine_trsf(np_array, trsf_unit="REAL_UNIT"):
    """Create an affine ``BalTrsf`` from a numpy array.

    The numpy array must be a quaternion describing an affine transformation

    Parameters
    ----------
    np_array : numpy.array
        quaternion describing the affine transformation
    trsf_unit : int or str, optional
        defines if the quaternion in expressed in real unit ("REAL_UNIT"), by default,
        or voxel unit ("VOXEL_UNIT")

    Returns
    -------
    BalTrsf
        affine transformation

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_trsf import np_array_to_affine_trsf
    >>> arr = np.random.random_sample((4, 4))
    >>> trsf = np_array_to_affine_trsf(arr)
    >>> print(trsf.get_unit())
    'REAL_UNIT'
    >>> print(trsf.get_type())
    'AFFINE_3D'
    >>> print(trsf.is_linear())
    True
    >>> trsf.c_display('test')
    type of 'test' is AFFINE_3D
    transformation unit is in real units
       0.833440359505476    0.117967030337482    0.639281681539620    0.989473005721904
       0.503056486652696    0.643971409647213    0.829838922556072    0.116883038428201
       0.086115555445802    0.404529497498842    0.617357824753340    0.528172757938566
       0.441608426544292    0.884445137670544    0.838913838120309    0.692011691783862

    """
    c_bal_trsf = new_c_bal_trsf(trsf_unit=trsf_unit, trsf_type="AFFINE_3D",
                                mat=np_array)
    allocate_c_bal_trsf(c_bal_trsf, np_array)
    return BalTrsf(c_bal_trsf=c_bal_trsf)


def spatial_images_to_deformable_trsf(vx, vy, vz, trsf_unit="REAL_UNIT"):
    """Create an affine ``BalTrsf`` from a numpy array.

    The numpy array must be a quaternion describing an affine transformation

    Parameters
    ----------
    vx : SpatialImage
        x-component of the non-linear deformation matrix
    vy : SpatialImage
        y-component of the non-linear deformation matrix
    vz : SpatialImage
        z-component of the non-linear deformation matrix
    trsf_unit : int or str, optional
        defines if the quaternion in expressed in real unit ("REAL_UNIT"), by default,
        or voxel unit ("VOXEL_UNIT")

    Returns
    -------
    BalTrsf
        vectorfield transformation instance

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping.bal_trsf import np_array_to_affine_trsf
    >>> arr = np.random.random_sample((4, 4))
    >>> trsf = np_array_to_affine_trsf(arr)
    >>> print(trsf.get_unit())
    'REAL_UNIT'
    >>> print(trsf.get_type())
    'AFFINE_3D'
    >>> print(trsf.is_linear())
    True
    >>> trsf.c_display('test')
    type of 'test' is AFFINE_3D
    transformation unit is in real units
       0.833440359505476    0.117967030337482    0.639281681539620    0.989473005721904
       0.503056486652696    0.643971409647213    0.829838922556072    0.116883038428201
       0.086115555445802    0.404529497498842    0.617357824753340    0.528172757938566
       0.441608426544292    0.884445137670544    0.838913838120309    0.692011691783862

    """
    c_bal_trsf = new_c_bal_trsf(vx=vx, vy=vy, vz=vz, trsf_unit=trsf_unit,
                                trsf_type="VECTORFIELD_3D")
    allocate_c_bal_trsf(c_bal_trsf, {'vx': vx, 'vy': vy, 'vz': vz})
    return BalTrsf(c_bal_trsf=c_bal_trsf)


class BalTrsf(object):
    """Class representing transformation objects.

    .. warning :: This class do not yet follow *bal_naming_convention* and behaviour, it is just a container without memory management.

    Notes
    -----
    Yo can only pass pointers in constructors. So to create transformations,
    initialise matrix or images outside the constructor and pass pointers
    (mat.c_ptr, image.c_ptr) to constructor.
    Parameters ``trsf_type`` & ``trsf_unit`` are used only if ``c_bal_trsf`` is
    not specified (None).

    Parameters
    ----------
    trsf_type : int or str, optional
        if given, type of transformation to initialise, see TRSF_TYPE_DICT
    trsf_unit : int or str, optional
        if given, unit of transformation to initialise, see TRSF_UNIT_DICT
    c_bal_trsf : cBalTrsf or BalTrsf, optional
        if given, existing ``ctypes`` structure or BalTrsf object

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.wrapping import BalTrsf
    >>> from timagetk.wrapping.bal_matrix import np_array_to_c_bal_matrix
    >>> # Initialize 3D rigid ``BalTrsf`` instance, in real units:
    >>> trsf = BalTrsf(trsf_type='RIGID_3D', trsf_unit='REAL_UNIT')
    >>> print(trsf.get_type())
    >>> 'RIGID_3D'
    >>> print(trsf.get_unit())
    >>> 'REAL_UNIT'

    >>> # Manually create identity quaternion:
    >>> quaternion = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    >>> cbal_mat = np_array_to_c_bal_matrix(quaternion)
    >>> trsf = BalTrsf(c_bal_trsf=cbal_mat)

    Attributes
    ----------
    mat : numpy.array
        Quaternion describing the linear transformation.
        ``None`` if vectorfield transformation.
    vx, vy, vz : numpy.array
        Displacement matrix in x, y and z for non-linear transformation.
        ``None`` if linear transformation.
    trsf_unit : int
        unit of the transformation, available units are defined in TRSF_UNIT_DICT
    trsf_type : int
        type of the transformation, available types are defined in
        TRSF_TYPE_DICT

    """

    def __init__(self, trsf_type=None, trsf_unit=None, c_bal_trsf=None):
        """BalTrsf object constructor. """
        # - Initialise ``ctypes`` structure in hidden attribute ``_c_bal_trsf``:
        self._c_bal_trsf = cBalTrsf()

        # - Initialise an empty trsf or use given ``c_bal_trsf``:
        if c_bal_trsf is None:
            logging.debug("Initializing empty BalTrsf object...")
            libblockmatching.BAL_InitTransformation(self.c_ptr)
            # WARNING: BAL_InitTransformation set 'type' ``ctypes`` field to undefined (UNDEF_TRANSFORMATION)!
            # WARNING: BAL_InitTransformation set 'unit' ``ctypes`` field to undefined (UNDEF_UNIT)!
            if trsf_unit is not None:
                if isinstance(trsf_unit, str):
                    trsf_unit = TRSF_UNIT_DICT[trsf_unit]
                check_type(trsf_unit, 'trsf_unit', int)
                unit_msg = "Using transformation unit '{}' ({})."
                logging.debug(
                    unit_msg.format(TRSF_UNIT_DICT[trsf_unit], trsf_unit))
                self._c_bal_trsf.transformation_unit = trsf_unit
            else:
                self._c_bal_trsf.transformation_unit = 0
                logging.warning("No transformation unit specified!")
            if trsf_type is not None:
                if isinstance(trsf_type, str):
                    trsf_type = TRSF_TYPE_DICT[trsf_type]
                check_type(trsf_type, 'trsf_type', int)

                type_msg = "Using transformation type '{}' ({})."
                logging.debug(
                    type_msg.format(TRSF_TYPE_DICT[trsf_type], trsf_type))
                self._c_bal_trsf.type = trsf_type
            else:
                self._c_bal_trsf.type = 0
                logging.warning("No transformation type specified!")
        else:
            init_msg = "Initializing BalTrsf object "
            if isinstance(c_bal_trsf, cBalTrsf):
                init_msg += "from a ``cBalTrsf`` object..."
                logging.debug(init_msg)
                libblockmatching.BAL_AllocTransformation(self.c_ptr,
                                                         c_bal_trsf.type,
                                                         pointer(c_bal_trsf.vx))
                libblockmatching.BAL_CopyTransformation(pointer(c_bal_trsf),
                                                        self.c_ptr)
            elif isinstance(c_bal_trsf, BalTrsf):
                init_msg += "from a ``BalTrsf`` object..."
                logging.debug(init_msg)
                libblockmatching.BAL_AllocTransformation(self.c_ptr,
                                                         c_bal_trsf._c_bal_trsf.type,
                                                         c_bal_trsf.vx.c_ptr)
                libblockmatching.BAL_CopyTransformation(
                    pointer(c_bal_trsf._c_bal_trsf), self.c_ptr)
            else:
                t = type(c_bal_trsf)
                raise TypeError("Unknown type '{}' for 'c_bal_trsf'!".format(t))

        # - Set object new attributes:
        # -- Transformation unit, see TRSF_UNIT_DICT:
        self.trsf_unit = self._c_bal_trsf.transformation_unit
        # -- Transformation type, see TRSF_TYPE_DICT:
        self.trsf_type = self._c_bal_trsf.type
        # -- Linear transformation matrix (BalMatrix):
        self.mat = BalMatrix(c_bal_matrix=self._c_bal_trsf.mat)
        # -- Non-linear transformation matrix (BalImage):
        self.vx = BalImage(c_bal_image=self._c_bal_trsf.vx)
        self.vy = BalImage(c_bal_image=self._c_bal_trsf.vy)
        self.vz = BalImage(c_bal_image=self._c_bal_trsf.vz)

    def __del__(self):
        """Class destructor. """
        self.free()

    @property
    def c_ptr(self):
        """Get the pointer to the C object. """
        return pointer(self._c_bal_trsf)

    @property
    def c_struct(self):
        """Get the ``cBalTrsf`` hidden attribute.

        Returns
        -------
        cBalTrsf
            the hidden attribute ``_c_bal_trsf``
        """
        return self._c_bal_trsf

    def free(self):
        """Free memory allocated to object. """
        if self._c_bal_trsf:
            libblockmatching.BAL_FreeTransformation(self.c_ptr)
            # Remove ref to dynamic fields.
            # If these fields are used outside, memory is managed outside
            # Else, there are no more references to theses fields, gc collect Bal* object and delete it
            # When BalImage and BalMatrix are deleted, memory is released
            self.mat = None
            self.vx = None
            self.vy = None
            self.vz = None

    def c_display(self, name=""):
        """Print information about the object as found in the allocated memory.

        Parameters
        ----------
        name : str
            name of the transformation, used for printing

        Returns
        -------
        str
            information about the BalTrsf object
        """
        libblockmatching.BAL_PrintTransformation(c_stdout, self.c_ptr, name)

    def has_linear_matrix(self):
        """Tests if a linear matrix is defined under ``self._c_bal_trsf.mat``.

        Returns
        -------
        bool
            ``True`` if defined, ``False`` if ``None`` (NULL pointer)
        """
        return bool(self._c_bal_trsf.mat.m)

    def has_nonlinear_matrix(self):
        """Tests if a linear matrix is defined under ``self._c_bal_trsf.vx``.

        Returns
        -------
        bool
            ``True`` if defined, ``False`` if ``None`` (NULL pointer)
        """
        return bool(self._c_bal_trsf.vx.array)

    def read(self, path):
        """Read a transformation given in ``path``.

        .. warning::

           This function:

           * can not differentiate between 'AFFINE' and 'RIGID' transformation, default is 'AFFINE'!
           * can not differentiate between 'REAL' and 'VOXEL' transformation unit for linear transformations, default is 'REAL_UNIT'!
           * will most likely result in a ``Segmentation error (core dumped)``, on some linux systems, when loading non-linear deformation!

        Parameters
        ----------
        path : str
            file path to the transformation to read
        """
        libblockmatching.BAL_ReadTransformation(self.c_ptr, str(path))
        self.trsf_type = self._c_bal_trsf.type
        self.trsf_unit = self._c_bal_trsf.transformation_unit

    def write(self, path):
        """Write a transformation under given ``path``.

        .. warning::

           Using this function on:

           * an empty instance, 'ie.' initialized as ``BalTrsf()``, will NOT write anything!
           * an almost empty instance, 'eg.' initialized as ``BalTrsf()``, will most likely result in a ``Core dumped``!
           * a linear transformation will not save its type (affine/rigid) or unit (real/voxel)!

        Parameters
        ----------
        path : str
            file path to the transformation to write
        """
        libblockmatching.BAL_WriteTransformation(self.c_ptr, str(path))

    def is_linear(self):
        """Test if the transformation matrix is of type 'Linear'.

        Linear transformation matrix are obtained from those types:

           * TRANSLATION_2D, TRANSLATION_3D;
           * TRANSLATION_SCALING_2D, TRANSLATION_SCALING_3D;
           * RIGID_2D, RIGID_3D;
           * SIMILITUDE_2D, SIMILITUDE_3D;
           * AFFINE_2D, AFFINE_3D.

        Returns
        -------
        is_linear : bool
            ``True`` if of type 'Linear', else ``False``
        """
        return libblockmatching.BAL_IsTransformationLinear(self.c_ptr) != 0

    def is_vectorfield(self):
        """Test if the transformation matrix is of type 'VectorField'.

        Non-linear transformation matrix are obtained from those types:

           * VECTORFIELD_2D, VECTORFIELD_3D;

        Returns
        -------
        is_vectorfield : bool
            ``True`` if of type 'VectorField', else ``False``
        """
        return libblockmatching.BAL_IsTransformationVectorField(self.c_ptr) != 0

    def get_type(self):
        """Returns the type of transformation matrix as a string.

        Convert the integer attribute ``trsf_type`` into a human readable type.
        """
        return TRSF_TYPE_DICT[self.trsf_type]

    def get_unit(self):
        """Returns the unit of transformation matrix as a string.

        Convert the integer attribute ``trsf_unit`` into a human readable unit.
        """
        return TRSF_UNIT_DICT[self.trsf_unit]
