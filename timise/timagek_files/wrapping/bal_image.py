# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.bal_image
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
from ctypes import pointer, c_void_p

from ..util import check_type
from .clib import c_stdout, libblockmatching
from .type_conversion import vt_type_to_c_type, vt_type_to_np_type, np_type_to_vt_type
from .c_bal_image import cBalImage
from .bal_common import bal_mystr_c_ptr, bal_mystr_c_struct

__all__ = [
    'bal_image_c_ptr',
    'bal_image_c_struct',
    'free_bal_image',
    'spatial_image_to_bal_image_fields',
    'init_c_bal_image',
    'allocate_c_bal_image',
    'new_c_bal_image',
    'spatial_image_to_c_bal_image',
    'bal_image_to_spatial_image',
    'BalImage'
]

DEFAULT_NAME = 'anonymous_bal_image'
DEFAULT_VT_TYPE = np_type_to_vt_type(np.uint8)


def bal_image_c_ptr(c_or_bal_image):
    """Return a pointer to instance of cBalImage or BalImage.

    Parameters
    ----------
    c_or_bal_image :cBalImage
        instance from which to get the pointer

    Returns
    -------
    c_ptr : pointer
        pointer to cBalImage instance


    Example
    -------
    >>> from timagetk.wrapping.bal_image import bal_image_c_ptr
    >>> c_ptr = bal_image_c_ptr(c_or_bal_image)
    """
    return bal_mystr_c_ptr(c_or_bal_image, cBalImage, BalImage)


def bal_image_c_struct(c_or_bal_image):
    """Return an instance of cBalImage or BalImage.

    Parameters
    ----------
    c_or_bal_image : cBalImage
        instance from which to get the pointer

    Returns
    -------
    cBalImage : cBalImage
        cBalImage instance

    Example
    -------
    >>> from timagetk.wrapping.bal_image import bal_image_c_struct
    >>> c_struct = bal_image_c_struct(c_or_bal_image)
    """
    return bal_mystr_c_struct(c_or_bal_image, cBalImage, BalImage)


def free_bal_image(c_or_bal_image):
    """Memory deallocation.

    Parameters
    ----------
    c_or_bal_image : cBalImage
        instance to deallocate from memory

    Example
    -------
    >>> from timagetk.wrapping.bal_image import free_bal_image
    >>> free_bal_image(c_or_bal_image)
    """
    c_struct = bal_image_c_struct(c_or_bal_image)
    c_ptr = pointer(c_struct)
    if c_ptr is not None:
        # data memory is managed in python side, set pointer to NULL to avoid to free it twice
        # (double free or corruption)
        c_struct.data = None
        libblockmatching.BAL_FreeImage(c_ptr)  # free structure
        del c_or_bal_image
    else:
        logging.warning("Could not retrieve pointer!")


def spatial_image_to_bal_image_fields(spatial_image):
    """Get cBalImage fields from a SpatialImage.

    List of recovered fields are:

     - **shape**: the shape of the image, a len-4 tuple (xdim, ydim, zdim, vdim);
     - **voxelsize**: the voxelsize of the image, a len-3 tuple;
     - **origin**: the origin of the image, a len-3 tuple;
     - **vdim**: here set to 1;
     - **np_type**: bit-depth encoding of the data in the original array.

    Parameters
    ----------
    spatial_image : SpatialImage
        a SpatialImage instance to use to get cBalImage fields

    Returns
    -------
    bal_image_kwargs : dict
        dictionary to use when initializing a new BalImage

    Raises
    ------
    TypeError
        If input ``spatial_image`` is not a ``SpatialImage`` instance.

    Example
    -------
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> img = imread(shared_data("time_3_seg.inr"))
    >>> from timagetk.wrapping.bal_image import spatial_image_to_bal_image_fields
    >>> bal_image_kwargs = spatial_image_to_bal_image_fields(img)
    >>> print(bal_image_kwargs)

    """
    from ..spatial_image import SpatialImage

    check_type(spatial_image, 'spatial_image', SpatialImage)

    # Make sure we have a 3D image:
    if spatial_image.is2D():
        spatial_image.to_3D()

    # Get data information from spatial image:
    bal_image_kwargs = {
        'shape': list(spatial_image.shape) + [1],
        'voxelsize': spatial_image.voxelsize,
        'origin': spatial_image.origin,
        'vdim': 1,
        'np_type': spatial_image.dtype,
        'name': spatial_image.filename if spatial_image.filename != '' else DEFAULT_NAME
    }

    return bal_image_kwargs


def init_c_bal_image(c_bal_image, shape, **kwargs):
    """Initialization of a cBalImage.

    A `cBalImage` is a 3D image.

    Parameters
    ----------
    c_bal_image : cBalImage
        object to initialize
    shape : list(int)
        shape of the array, *ie.* size of each axis

    Other Parameters
    ----------------
    name : str
        name of the image to initialize
    voxelsize : list(int)
        voxelsize of each dimension
    np_type : str
        bit-depth encoding of the data in numpy style
    vt_type : str
        bit-depth encoding of the data in VT style

    Example
    -------
    >>> from timagetk.wrapping import cBalImage
    >>> from timagetk.wrapping.bal_image import init_c_bal_image
    >>> c_bal_image = cBalImage()
    >>> init_c_bal_image(c_bal_image, shape=[5, 5, 3])
    >>> print(c_bal_image.ncols)
    5L
    >>> print(c_bal_image.nrows)
    5L
    >>> print(c_bal_image.nplanes)
    3L
    >>> print(c_bal_image.name)
    anonymous(bal_image
    >>> print(c_bal_image.vx, c_bal_image.vy, c_bal_image.vz)
    1.0 1.0 1.0

    Raises
    ------
    TypeError
        If input ``c_bal_image`` is not a ``cBalImage`` instance.

    """
    check_type(c_bal_image, 'c_bal_image', cBalImage)

    if isinstance(shape, tuple):
        shape = list(shape)
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

    # Check the given shape parameter:
    if len(shape) == 2:
        shape = shape + [1, 1]  # xdim, ydim, zdim, vdim
        # - Get voxelsize:
        vx, vy = kwargs.get('voxelsize', (1., 1.))
        vz = 1.
    elif len(shape) == 3:
        shape = shape + [1]  # xdim, ydim, zdim, vdim
        if shape[-1] == 1:
            vx, vy = kwargs.get('voxelsize', (1., 1.))
            vz = 1.
        else:
            vx, vy, vz = kwargs.get('voxelsize', (1., 1., 1.))
    elif len(shape) == 4:
        assert shape[-1] == 1 # xdim, ydim, zdim, vdim
        vx, vy, vz = kwargs.get('voxelsize', (1., 1., 1.))
    else:
        msg = "Unknown cBalImage initialization method for shape: {}"
        raise NotImplementedError(msg.format(shape))
    xdim, ydim, zdim, vdim = shape

    # - Get type:
    vt_type = kwargs.get('vt_type', None)
    np_type = kwargs.get('np_type', None)
    if np_type is not None:
        vtype = np_type_to_vt_type(np_type)
    elif vt_type is not None:
        vtype = vt_type
    else:
        vtype = DEFAULT_VT_TYPE

    # - Get the name of the object to create:
    name = kwargs.get('name', DEFAULT_NAME)

    # - Initialize a cBalImage:
    libblockmatching.BAL_InitImage(
        pointer(c_bal_image), name, xdim, ydim, zdim, vdim, vtype)
    # - Set the voxelsize in each direction:
    c_bal_image.vx, c_bal_image.vy, c_bal_image.vz = vx, vy, vz


def allocate_c_bal_image(c_bal_image, spatial_image):
    """Memory allocation of a c_bal_image.

    Parameters
    ----------
    c_bal_image : cBalImage
        instance to allocate, this object will contain the SpatialImage
    spatial_image : numpy.array or SpatialImage
        object to allocate in a cBalImage instance

    Example
    -------
    >>> allocate_c_bal_image(c_bal_image, spatial_image)

    Raises
    ------
    TypeError
        If input ``c_bal_image`` is not a ``cBalImage`` instance.

    """
    check_type(c_bal_image, 'c_bal_image', cBalImage)

    # Data pointer cast to a particular c-types object
    _data = spatial_image.ctypes.data_as(c_void_p)
    # update data pointer: bal_image.data -> ctype numpy array
    c_bal_image.data = _data
    # Create ***array from data
    libblockmatching.BAL_AllocArrayImage(pointer(c_bal_image))
    # DO NOT FORGET TO FREE IT LATER


def new_c_bal_image(**kwargs):
    """Create a ``cBalImage`` instance using kwargs.

    Other Parameters
    ----------------
    name : str
        name of the image to initialize
    shape : list
        shape of the array, ie. size of each axis
    voxelsize : int
        voxelsize in y-direction
    np_type : str
        bit-depth encoding of the data in numpy style
    vt_type : str
        bit-depth encoding of the data in VT style
    vdim : int
        here set to 1

    Returns
    -------
    c_bal_image : cBalImage
        a new cBalImage instance

    Example
    -------
    >>> from timagetk.wrapping.bal_image import new_c_bal_image
    >>> c_bal_image = new_c_bal_image(**kwargs)

    """
    c_bal_image = cBalImage()
    init_c_bal_image(c_bal_image, **kwargs)
    return c_bal_image


def spatial_image_to_c_bal_image(spatial_image, **kwargs):
    """Convert ``SpatialImage`` to ``cBalImage``.

    Parameters
    ----------
    spatial_image : SpatialImage
        object to allocate in a cBalImage instance

    Other Parameters
    ----------------
    name : str
        name of the image to initialize
    shape : list
        shape of the array, ie. size of each axis
    voxelsize : int
        voxelsize in y-direction
    np_type : str
        bit-depth encoding of the data in numpy style
    vt_type : str
        bit-depth encoding of the data in VT style
    vdim : int
        here set to 1

    Returns
    -------
    c_bal_image : cBalImage
        converted ``SpatialImage`` instance

    Example
    -------
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.wrapping.bal_image import spatial_image_to_c_bal_image
    >>> sp_img = imread(shared_data('time_0_cut.inr'))
    >>> c_bal_image = spatial_image_to_c_bal_image(sp_img)
    >>> type(sp_img)
    >>> print(sp_img.shape)
    >>> print(c_bal_image.ncols, c_bal_image.nrows, c_bal_image.nplanes)

    >>> from timagetk.components import LabelledImage
    >>> lab_img = LabelledImage(imread(shared_data("time_3_seg.inr")), no_label_id=0)
    >>> c_bal_image = spatial_image_to_c_bal_image(lab_img)
    >>> type(lab_img)

    Raises
    ------
    TypeError
        If input ``spatial_image`` is not a ``SpatialImage`` instance.

    """
    # - Get the cBalImage fields:
    spatial_image_kwargs = spatial_image_to_bal_image_fields(spatial_image)
    # - Update the keyword arguments with previously obtained fields:
    kwargs.update(spatial_image_kwargs)
    # - Use it to create a new cBalImage and allocate the array:
    c_bal_image = new_c_bal_image(**kwargs)
    allocate_c_bal_image(c_bal_image, spatial_image)
    return c_bal_image


def bal_image_to_spatial_image(c_or_bal_image):
    """Convert ``cBalImage`` to ``SpatialImage``.

    Parameters
    ----------
    c_or_bal_image : cBalImage
        C object to convert into ``SpatialImage``

    Returns
    -------
    SpatialImage
        new instance obtained by conversion of ``cBalImage``

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.wrapping.bal_image import spatial_image_to_c_bal_image
    >>> img_path = shared_data('time_0_cut.inr')
    >>> sp_img = imread(img_path)
    >>> c_bal_image = spatial_image_to_c_bal_image(sp_img)
    >>> # Test equality of given ``numpy.array`` to allocated ``cBalMatrix``
    >>> from timagetk.wrapping.bal_image import bal_image_to_spatial_image
    >>> np.array_equal(bal_image_to_spatial_image(c_bal_image), sp_img)

    """
    b = bal_image_c_struct(c_or_bal_image)

    x, y, z, v = b.ncols, b.nrows, b.nplanes, b.vdim
    size = x * y * z * v
    resolution = [b.vx, b.vy, b.vz]

    _cdtype = vt_type_to_c_type(b.type)
    # SR 21/03
    _nptype = vt_type_to_np_type(b.type)

    if b.data is None:
        _np_array = np.ndarray(size)
    else:
        _ct_array = (_cdtype * size).from_address(b.data)
        _np_array = np.ctypeslib.as_array(_ct_array)
    if v == 1:
        # --- SR 21/03
        # arr = np.array(_np_array.reshape(x, y, z, order="F"))
        arr = np.array(_np_array.reshape(x, y, z, order="F"), dtype=_nptype)
    else:
        # --- SR 21/03
        # arr = np.array(_np_array.reshape(x, y, z, v, order="F"))
        arr = np.array(_np_array.reshape(x, y, z, v, order="F"), dtype=_nptype)
    from ..spatial_image import SpatialImage

    return SpatialImage(arr, voxelsize=resolution, origin=[0] * arr.ndim,
                        dtype=arr.dtype)


class BalImage(object):
    """Class representing image objects.

    Parameters
    ----------
    np_array : numpy.array, optional
        numpy array to use when creating ``BalImage`` instance
    c_bal_matrix : cBalMatrix, optional
        C-instance to use when creating ``BalImage`` instance

    Other Parameters
    ----------------
    name : str
        name of the image to initialize
    shape : list
        shape of the array, ie. size of each axis
    voxelsize : int
        voxelsize in y-direction
    np_type : str
        bit-depth encoding of the data in numpy style
    vt_type : str
        bit-depth encoding of the data in VT style
    vdim : int
        here set to 1

    Notes
    -----
    Other parameters are used by:
      * ``spatial_image_to_c_bal_image`` if ``np_array`` is given;
      * ``new_c_bal_image`` if no parameters are given;

    Example
    -------
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.wrapping import BalImage
    >>> from timagetk.wrapping.bal_image import spatial_image_to_c_bal_image
    >>> # Load test ``SpatialImage``:
    >>> img_path = shared_data('time_0_cut.inr')
    >>> sp_img = imread(img_path)
    >>> # Convert it to a ``cBalImage`` type:
    >>> c_bal_image = spatial_image_to_c_bal_image(sp_img)

    >>> # Initialize from ``SpatialImage`` instance:
    >>> bimg = BalImage(spatial_image=sp_img)
    >>> # Test equality with original ``SpatialImage`` instance:
    >>> sp_img.equal_array(bimg.to_spatial_image())
    True

    >>> # Initialize from ``cBalImage`` instance:
    >>> from timagetk.wrapping.bal_image import spatial_image_to_c_bal_image
    >>> cbimg = spatial_image_to_c_bal_image(sp_img)
    >>> bimg = BalImage(c_bal_image=cbimg)
    >>> sp_img.equal_array(bimg.to_spatial_image())
    True

    """

    def __init__(self, spatial_image=None, c_bal_image=None, **kwargs):
        """BalImage object constructor. """
        if spatial_image is not None:
            logging.debug("Initializing BalMatrix from SpatialImage...")
            self._spatial_image = spatial_image
            self._c_bal_image = spatial_image_to_c_bal_image(
                self._spatial_image, **kwargs)
        elif c_bal_image is not None:
            logging.debug("Initializing BalMatrix from cBalImage...")
            self._c_bal_image = c_bal_image
            self._spatial_image = self.to_spatial_image()
        else:
            logging.debug(
                "Initializing empty BalMatrix object from keyword arguments...")
            self._c_bal_image = new_c_bal_image(**kwargs)
            self._spatial_image = self.to_spatial_image()

    def __del__(self):
        """Class destructor. """
        self.free()

    def __eq__(self, other):
        """Test equality between self and ``other``. """
        if not isinstance(other, self.__class__):
            return False
        sp_self = self.to_spatial_image().copy()
        sp_other = other.to_spatial_image().copy()

        elements_ok = np.array_equal(sp_self, sp_other)

        ix, iy, iz = sp_self.voxelsize
        nx, ny, nz = sp_other.voxelsize

        rx_ok = (round(abs(nx - ix), 7) == 0)
        ry_ok = (round(abs(ny - iy), 7) == 0)
        rz_ok = (round(abs(nz - iz), 7) == 0)

        dtype_ok = sp_self.dtype == sp_other.dtype

        return elements_ok and rx_ok and ry_ok and rz_ok and dtype_ok

    def __ne__(self, other):
        """Test non-equality between self and ``other``. """
        return not self.__eq__(other)

    @property
    def c_ptr(self):
        """Get the pointer to the C object ``cBalImage``."""
        return pointer(self._c_bal_image)

    @property
    def c_struct(self):
        """Get the ``cBalImage`` hidden attribute.

        Returns
        -------
        cBalImage
            the hidden attribute ``_c_bal_image``
        """
        return self._c_bal_image

    def free(self):
        """Free memory allocated to object. """
        if self._c_bal_image:
            free_bal_image(self._c_bal_image)
        self._c_bal_image = None

    def c_display(self, name=''):
        """Print information about the object as found in the allocated memory.

        Parameters
        ----------
        name : str
            name of the transformation, used for printing

        Returns
        -------
        str
            information about the BalImage object
        """
        if name == '':
            try:
                name = self._spatial_image.metadata['filename']
            except KeyError:
                pass
        libblockmatching.BAL_PrintImage(c_stdout, self.c_ptr, name)

    def to_spatial_image(self, **kwargs):
        """Convert to a ``SpatialImage``.

        Returns
        -------
        SpatialImage
            converted instance
        """
        return bal_image_to_spatial_image(self._c_bal_image, **kwargs)

    def to_labelled_image(self, **kwargs):
        """Convert to a ``LabelledImage``.

        Returns
        -------
        LabelledImage
            converted instance
        """
        from ..labelled_image import LabelledImage

        return LabelledImage(self.to_spatial_image(**kwargs))
