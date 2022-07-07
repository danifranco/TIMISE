import logging
import copy as cp
import numpy as np

from .util import check_type, get_attributes, get_class_name
from .components.metadata import ImageMetadata, Metadata, ProcessMetadata, IMAGE_MD_TAGS

__all__ = ['SpatialImage']

# - Define default values:
AXIS_ORDER = {'x': 0, 'y': 1, 'z': 2}
EPS = 1e-9
DEC_VAL = 6
DEFAULT_VXS_2D, DEFAULT_VXS_3D = [1.0, 1.0], [1.0, 1.0, 1.0]
DEFAULT_ORIG_2D, DEFAULT_ORIG_3D = [0, 0], [0, 0, 0]
DEFAULT_VALUE_MSG = "Set '{}' to default value: {}"

# - Define possible values for 'dtype':
# For details about numpy types, see:
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
DICT_TYPES = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
              5: np.int32, 6: np.uint64, 7: np.int64, 8: np.float32,
              9: np.float64, 10: np.float_, 11: np.complex64, 12: np.complex128,
              13: np.complex_, 'uint8': np.uint8, 'uint16': np.uint16,
              'ushort': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
              'uint': np.uint64, 'ulonglong': np.uint64, 'int8': np.int8,
              'int16': np.int16, 'short': np.int16, 'int32': np.int32,
              'int64': np.int64, 'int': np.int64, 'longlong': np.int64,
              'float16': np.float16, 'float32': np.float32,
              'single': np.float32, 'float64': np.float64, 'double': np.float64,
              'float': np.float64, 'float128': np.float_,
              'longdouble': np.float_, 'longfloat': np.float_,
              'complex64': np.complex64, 'singlecomplex': np.complex64,
              'complex128': np.complex128, 'cdouble': np.complex128,
              'cfloat': np.complex128, 'complex': np.complex128,
              'complex256': np.complex_, 'clongdouble': np.complex_,
              'clongfloat': np.complex_, 'longcomplex': np.complex_}

AVAIL_TYPES = sorted([k for k in DICT_TYPES if isinstance(k, str)])
# - Define default type:
DEFAULT_DTYPE = DICT_TYPES[0]
# - List of protected attribute or poperties:
PROTECT_PPTY = ['shape', 'min', 'max', 'mean']
# - Array equality testing methods:
EQ_METHODS = ['max_error', 'cum_error']


def dimensionality_test(dim, list2test):
    """Quick testing of dimensionality with print in case of error."""
    d = len(list2test)
    try:
        assert d == dim
    except:
        msg = "Provided values ({}) is not of the same than the array ({})!"
        raise ValueError(msg.format(d, dim))


def compute_extent(voxelsize, shape):
    """Compute new extent of object from shape and voxelsize.

    Parameters
    ----------
    voxelsize : list(float)
        voxelsize of the image
    shape : list(int)
        shape of the image, ie. its size in voxel

    Returns
    -------
    list(float)
        extent of the image

    Examples
    --------
    >>> from timagetk.components.spatial_image import compute_extent
    >>> compute_extent([0.5, 0.5], (10, 10))
    [5., 5.]

    """
    shape = np.array(shape)
    voxelsize = np.array(voxelsize, dtype=np.float)
    return np.multiply(shape, voxelsize).tolist()


def compute_shape(voxelsize, extent):
    """Compute new shape of object from extent and voxelsize.

    Parameters
    ----------
    voxelsize : list(float)
        voxelsize of the image
    extent : list(float)
        extent of the image, ie. its real size

    Returns
    -------
    list(int)
        shape of the image

    Examples
    --------
    >>> from timagetk.components.spatial_image import compute_shape
    >>> compute_shape([0.5, 0.5], [10, 10])
    [20, 20]

    """
    extent = np.array(extent)
    voxelsize = np.array(voxelsize, dtype=np.float)
    return tuple(map(int, np.around(np.divide(extent, voxelsize), 0)))


def _to_list(val):
    """Returns a list from tuple or array, else raise a TypeError. """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, tuple):
        val = list(val)
    if not isinstance(val, list):
        raise TypeError("Accepted type are tuple, list and np.array!")
    else:
        return val

def compute_voxelsize(extent, shape):
    """Compute new voxelsize of object from shape and extent.

    Parameters
    ----------
    extent : list(float)
        extent of the image, ie. its real size
    shape : list(int)
        shape of the image, ie. its size in voxel

    Returns
    -------
    list(float)
        extent of the image

    Examples
    --------
    >>> from timagetk.components.spatial_image import compute_voxelsize
    >>> compute_voxelsize([15., 15.], (10, 10))
    [5., 5.]

    """
    shape = np.array(shape)
    extent = np.array(extent, dtype=np.float)
    return np.divide(extent, shape).tolist()


def new_from_spatialimage(input_array, **kwargs):
    """Check kwargs for known attributes in the SpatialImage ``input_array``.

    If keyword argument and attribute values differs, we replace the kwargs
    values by the attribute, except if attribute is ``None``.
    The list of compared attributes is:

    * origin
    * voxelsize
    * metadata

    Parameters
    ----------
    input_array : SpatialImage
        SpatialImage from which the attribute should be compared against keyword
        arguments

    Other Parameters
    ----------------
    origin : list or numpy.array
        origin given as

    Returns
    -------
    dict
        updated keyword arguments dictionary
    """
    logging.debug("New SpatialImage from SpatialImage...")
    attr_list = ["origin", "voxelsize", "metadata"]
    # Get optional keyword arguments or set to None:
    kwds = [kwargs.get(kwd, None) for kwd in attr_list]

    attr_dict = get_attributes(input_array, attr_list)
    class_name = get_class_name(input_array)
    msg = "Overriding optional keyword arguments '{}' ({}) with defined attribute ({}) in given '{}'!"

    # -- Compare keyword argument values with attribute value:
    for attr_n, kwd in zip(attr_list, kwds):
        attr = attr_dict[attr_n]
        if attr is not None and kwd is not None and attr != kwd:
            logging.debug(msg.format(attr_n, kwd, attr, class_name))
            kwargs[attr_n] = attr
        else:
            logging.debug("Got attribute '{}': {}".format(attr_n, attr))
            logging.debug("Got kwarg '{}': {}".format(attr_n, kwd))

    return kwds


class SpatialImage(np.ndarray):
    """Management of 2D and 3D images.

    A ``SpatialImage`` gathers a numpy array and some metadata (such as voxelsize,
    physical extent, origin, type, etc.).

    Through a ``numpy.ndarray`` inheritance, all usual operations on
    `numpy.ndarray <http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.html>`_
    objects (sum, product, transposition, etc.) are available.

    All image processing operations are performed on this data structure, that
    is also used to solve inputs (read) and outputs (write).

    We `subclass <https://docs.scipy.org/doc/numpy-1.14.0/user/basics.subclassing.html>`_
    ndarray using view casting in the '__new__' section.
    View casting is the standard ``ndarray`` mechanism by which you take an
    ``ndarray`` of any subclass, and return a view of the array as another
    (specified) subclass, here a ``SpatialImage``.

    How to create ``SpatialImage``
    ------------------------------
    With an numpy array as ``input_array``:
      * specify ``origin`` & ``voxelsize``, else use ``DEFAULT_ORIG_2D`` & ``DEFAULT_VXS_2D`` or ``DEFAULT_ORIG_3D`` & ``DEFAULT_VXS_3D`` according to dimensionality;

    With a ``SpatialImage`` instance or any other which inherit it:
      * its attributes 'origin', 'voxelsize' & 'metadata' are used whatever the values given as arguments when calling contructor;

    In any case:
      * specifying ``dtype`` will change the array dtype to the given one;
      * attributes 'shape' and 'ndim' are taken from the array;
      * specifying 'filename' & 'filepath' in the ``metadata`` will allow to define them as attributes;
      * Images tags (``IMAGE_MD_TAGS`` + 'extent') are set or updated if a metadata dictionary is given.

    TODO: defines a 'personal_metadata' attribute to store other metadata ?

    Modifying image properties
    --------------------------
    Modifying 'ndim' property is not possible, be serious now.
    Modifying 'dtype' property directly is not possible, use the ``astype`` method.

    Modifying 'origin' property is not possible, do it at object instantiation.
    FIXME: This is not what its doing... yet!
    Modifying 'voxelsize' property will change the image 'shape' and preserve 'extent'.
    FIXME: This is not what its doing... yet!
    Modifying 'shape' property will change the image 'voxelsize' and preserve 'extent'.
    FIXME: This is not what its doing... yet!
    Modifying 'extent' property should resample the image.
    FIXME: This is not what its doing... yet!

    .. warning: Modification of a property, when possible, will change the object, not return a new one!

    Attributes
    ----------
    metadata_image : ImageMetadata
        A self generated list of attribute, contains basic image information
        such as: 'shape', 'ndim', 'dtype', 'origin', 'voxelsize' & 'extent'.
        See ``ImageMetadata`` class docstring in ``timagetk.components.metadata``.
    metadata_acquisition : Metadata
        Acquisition metadata is a list of attributes related to the measurement
        settings, the machine setup, or even used equipments and parameters.
        Use the 'scan_info' key in ``metadata`` dict to pass them to constructor.
    metadata_processing : ProcessMetadata
        A self generated hierarchical dictionary of the algorithms applied to
        the image. It contains process names, the parameters use as well as their
        values and, if any, the sub-process called with the same degree of info.

    """

    def __new__(cls, input_array, origin=None, voxelsize=None, dtype=None,
                metadata=None, **kwargs):
        """Image object constructor (2D and 3D images)

        Notes
        -----
        To use 'input_array' dtype (from numpy), leave 'dtype' to ``None``, else it
        will modify the dtype to the given values.
        Any key in ``kwargs`` matching a keyword argument (ie. 'origin', 'dtype'
        or 'voxelsize') will be ignored!

        Parameters
        ----------
        input_array : numpy.ndarray
            2D or 3D array defining an image, eg. intensity or labelled image
        origin : list, optional
            coordinates of the origin in the image, default: [0, 0] or [0, 0, 0]
        voxelsize : list, optional.
            image voxelsize, default: [1.0, 1.0] or [1.0, 1.0, 1.0]
        dtype : str, optional
            if given, should be in ``AVAIL_TYPES``, and will modify the
            input_array type
        metadata : dict, optional
            dictionary of image metadata, default is an empty dict

        Returns
        -------
        SpatialImage
            image with metadata

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.random.random_sample((15, 15, 5)).astype(np.float16)
        >>> img = SpatialImage(test_array, voxelsize=[0.5, 0.5, 1.])
        >>> isinstance(img, SpatialImage)
        True
        >>> isinstance(img, np.ndarray)
        True
        >>> print(img.voxelsize)
        [0.5, 0.5]
        """
        # - Accept ``SpatialImage`` or ``numpy.ndarray`` instances:
        if isinstance(input_array, SpatialImage):
            # - Override keyword arguments if defined, except ``dtype``:
            origin, voxelsize, metadata = new_from_spatialimage(input_array,
                                                                origin=origin,
                                                                voxelsize=voxelsize,
                                                                metadata=metadata)
        else:
            # - Test input array type, should be a numpy array :
            check_type(input_array, 'input_array', np.ndarray)

        # ----------------------------------------------------------------------
        # DTYPE:
        # ----------------------------------------------------------------------
        if dtype is None:
            # - Set it to the array type:
            dtype = input_array.dtype
        else:
            # Check it is a known 'dtype' or a numpy.dtype:
            try:
                assert dtype in AVAIL_TYPES or isinstance(dtype, np.dtype)
            except AssertionError:
                msg = "Unknown 'dtype' value '{}', available types are: {}"
                raise ValueError(msg.format(dtype, AVAIL_TYPES))

        # ----------------------------------------------------------------------
        # ARRAY:
        # ----------------------------------------------------------------------
        # - View casting (see class doc) of the array as a SpatialImage subclass:
        # NOTE: it convert array to given dtype if not None:
        # This call ``__array_finalize__``!
        if input_array.flags.f_contiguous:
            obj = np.asarray(input_array, dtype=dtype).view(cls)
        else:
            obj = np.asarray(input_array, dtype=dtype, order='F').view(cls)

        # - Test input array dimensionality, should be of dimension 2 or 3:
        try:
            assert input_array.ndim in [2, 3]
        except AssertionError:
            msg = "Input 'input_array' must be 2D or 3D!"
            for attr in ['ndim', 'shape', 'dtype']:
                try:
                    msg += "Got '{}': {}".format(attr,
                                                 getattr(input_array, attr))
                except:
                    pass
            logging.debug(msg)
            return None  # WEIRD behavior... seems required by some functions
            # Original behavior was to test if the dim was 2D or 3D but was not
            # doing anything otherwise (no else declared!).

        # ----------------------------------------------------------------------
        # Metadata:
        # ----------------------------------------------------------------------
        # - Initialize (empty) or check type of given metadata dictionary:
        if metadata is None:
            metadata = {}
            logging.debug("No metadata dictionary provided!")
        else:
            check_type(metadata, 'metadata', dict)

        # - Round the voxelsize to have consistent voxelsize:
        # FIXME : why do we round 'voxelsize' ?! Also done elsewhere in the code!!
        # voxelsize = around_list(voxelsize, 6)

        # - Get ImageMetadata:
        # NOTE: don't forget to set 'origin' and 'voxelsize' as keyword
        # arguments, otherwise they will be set to their default values.
        obj.metadata_image = ImageMetadata(input_array, origin=origin,
                                           voxelsize=voxelsize)
        # - Update the metadata dictionary for image IMAGE_MD_TAGS & extent:
        metadata = obj.metadata_image.update_metadata(metadata)
        obj._metadata = metadata

        # - Set 'voxelsize', 'origin' & 'extent' hidden attributes:
        obj._voxelsize = _to_list(obj.metadata_image['voxelsize'])
        obj._origin = _to_list(obj.metadata_image['origin'])
        obj._extent = _to_list(obj.metadata_image['extent'])

        # - Get acquisition metadata using 'scan_info' key:
        obj.metadata_acquisition = Metadata(metadata.get('scan_info', {}))
        if obj.metadata_acquisition is None:
            msg = "Could not set 'metadata_acquisition' attribute!"
            raise AttributeError(msg)

        # - Get processing metadata :
        obj.metadata_processing = {}

        # -- Set the 'class' metadata if not defined:
        try:
            assert 'class' in obj.metadata_processing
        except AssertionError:
            obj.metadata_processing.update({'class': 'SpatialImage'})
        else:
            # DO NOT try to change returned class, eg. 'LabelledImage', ...
            pass

        # - Defines some attributes:
        # -- File name and path, if known:
        obj.filename = metadata.get('filename', '')
        obj.filepath = metadata.get('filepath', '')

        return obj

    def __array_finalize__(self, obj):
        """
        This is the mechanism that numpy provides to allow subclasses to handle
        the various ways that new instances get created.

        Parameters
        ----------
        obj :
            the object returned by the __new__ method.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> type(img)
        timagetk.components.spatial_image.SpatialImage
        >>> # Taking the first z-slice automatically return a SpatialImage instance:
        >>> img2d = img[:, :, 0]
        >>> type(img2d)
        timagetk.components.spatial_image.SpatialImage

        """
        self._origin = getattr(obj, '_origin', None)
        self._voxelsize = getattr(obj, '_voxelsize', None)
        self._extent = getattr(obj, '_extent', None)

        self.filename = getattr(obj, 'filename', None)
        self.filepath = getattr(obj, 'filepath', None)

        self._metadata = getattr(obj, '_metadata', None)
        self.metadata_image = getattr(obj, 'metadata_image', None)
        self.metadata_acquisition = getattr(obj, 'metadata_acquisition', None)
        self.metadata_processing = getattr(obj, 'metadata_processing', None)

        self._min = getattr(obj, '_min', None)
        self._max = getattr(obj, '_max', None)
        self._mean = getattr(obj, '_mean', None)

        # Case where we are doing view casting of a SpatialImage and we changed the array:
        md_img = self.metadata_image

        # View Casting DEBUG:
        # if md_img is not None:
        #     print("Got image metadata: {}".format(md_img.get_dict()))
        #     print("With md_img['shape']: {}".format(md_img['shape']))
        #     print("With obj.shape: {}".format(obj.shape))
        # -----

        # if md_img is not None and md_img['shape'] != obj.shape:
        #     # Update image metadata:
        #     self.metadata_image.get_from_image(obj)
        #     self.metadata = self.metadata_image.update_metadata(self.metadata)
        #     # Reset some attributes
        #     for attr in ['min', 'max', 'mean']:
        #         setattr(self, '_' + attr, None)

    def __str__(self):
        """
        Method called when printing the object.
        """
        msg = "SpatialImage object with following metadata:\n"
        md = self.metadata
        msg += '\n'.join(['   - {}: {}'.format(k, v) for k, v in md.items()])
        return msg

    # ##########################################################################
    #
    # SpatialImage properties:
    #
    # ##########################################################################

    @property
    def extent(self):
        """Get ``SpatialImage`` physical extent.

        It is related to the array shape and image voxelsize.

        Returns
        -------
        list
            ``SpatialImage`` physical extent

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> print(image.extent)
        [5.0, 5.0]
        """
        return cp.copy(self._extent)

    @extent.setter
    def extent(self, img_extent):
        """Set ``SpatialImage`` physical extent.

        This will change voxelsize based on array shape.

        Parameters
        ----------
        img_extent : list
            ``SpatialImage`` new physical extent.

        Notes
        -----
        Metadata are updated according to the new physical extent and voxelsize.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> print(image.voxelsize)
        [1.0, 1.0]
        >>> image.extent = [10.0, 10.0]
        Set extent to '[10.0, 10.0]'
        Changed voxelsize to '[2.0, 2.0]'
        >>> print(image.extent)
        [10.0, 10.0]
        >>> print(image.voxelsize)
        [2.0, 2.0]
        """
        logging.warning("You are changing a physical property of the image!")
        logging.warning("This should not happen!")
        logging.warning("You better know what you are doing!")

        dimensionality_test(self.get_dim(), img_extent)
        # - Update 'extent' hidden attribute:
        self._extent = _to_list(img_extent)
        # - Update 'voxelsize' hidden attribute:
        self._voxelsize = compute_voxelsize(self.extent, self.shape)
        # - Update 'extent' & 'voxelsize' metadata:
        self.metadata_image['extent'] = self.extent
        self.metadata_image['voxelsize'] = self.voxelsize
        logging.info("Set extent to '{}'".format(self.extent))
        logging.info("Changed voxelsize to '{}'".format(self.voxelsize))
        return

    @property
    def metadata(self):
        """Get ``SpatialImage`` metadata dictionary.

        Returns
        -------
        dict
            metadata dictionary
        """
        md = {}
        if isinstance(self.metadata_image, ImageMetadata):
            md.update(self.metadata_image.get_dict())
        if isinstance(self.metadata_acquisition, Metadata):
            acq = self.metadata_acquisition.get_dict()
            if acq != {}:
                md['acquisition'] = acq
        if isinstance(self.metadata_processing, ProcessMetadata):
            proc = self.metadata_processing.get_dict()
            if proc != {}:
                md['processing'] = proc

        self._metadata.update(md)
        return cp.copy(self._metadata)

    @metadata.setter
    def metadata(self, img_md):
        """Update ``SpatialImage`` metadata dictionary. """
        self._metadata.update(img_md)

    @property
    def origin(self):
        """Get ``SpatialImage`` origin.

        Returns
        -------
        list
            ``SpatialImage`` origin coordinates

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> print(image.origin)
        [0, 0]
        """
        return cp.copy(self._origin)

    @origin.setter
    def origin(self, img_origin):
        """Set ``SpatialImage`` origin.

        Given list should be of same length than the image dimensionality.

        Parameters
        ----------
        img_origin : list
            ``SpatialImage`` origin coordinates,

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.origin = [2, 2]
        Set origin to '[2, 2]'
        """
        logging.warning("You are changing a physical property of the image!")
        logging.warning("This should not happen!")
        logging.warning("You better know what you are doing!")

        dimensionality_test(self.get_dim(), img_origin)
        # - Update hidden attribute 'origin':
        self._origin = _to_list(img_origin)
        # - Update hidden attribute metadata key 'origin':
        self.metadata_image['origin'] = self.origin
        logging.info("Set origin to '{}'".format(self.origin))
        return

    @property
    def voxelsize(self):
        """Get ``SpatialImage`` voxelsize.

        Returns
        -------
        list
            ``SpatialImage`` voxelsize

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> print(image.voxelsize)
        [1.0, 1.0]
        """
        return cp.copy(self._voxelsize)

    @voxelsize.setter
    def voxelsize(self, img_vxs):
        """Change ``SpatialImage`` voxelsize by preserving 'extent'.

        Parameters
        ----------
        img_vxs : list
            ``SpatialImage`` new voxelsize

        Notes
        -----
        Metadata are updated according to the new physical extent and voxelsize.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.random.random_sample((15,15,10)).astype(np.uint8)
        >>> image = SpatialImage(test_array)
        >>> print(image.voxelsize)
        [1.0, 1.0, 1.0]
        >>> print(image.extent)
        [15.0, 15.0, 10.0]
        >>> print(image.shape)
        (15, 15, 10)
        >>> # - Change image voxelsize:
        >>> image.voxelsize = [0.5, 0.5, 0.5]
        >>> print(image.voxelsize)
        [0.5, 0.5, 0.5]
        >>> print(image.extent)
        [7.5, 7.5, 5.0]
        >>> print(image.shape)
        (15, 15, 10)

        """
        logging.warning("You are changing a physical property of the image!")
        logging.warning("This should not happen!")
        logging.warning("You better know what you are doing!")

        dimensionality_test(self.get_dim(), img_vxs)
        # - Update 'voxelsize' hidden attribute:
        self._voxelsize = _to_list(img_vxs)
        # - Update 'extent' hidden attribute:
        self._extent = compute_extent(self.voxelsize, self.shape)
        # - Update 'extent' & 'voxelsize' metadata:
        self.metadata_image['voxelsize'] = self.voxelsize
        self.metadata_image['extent'] = self.extent
        logging.info("Set voxelsize to '{}'".format(self.voxelsize))
        logging.info("Changed extent to '{}'".format(self.extent))
        return

    @property
    def resolution(self):
        """
        Ensure backward compatibility with older openalea.image package.

        .. deprecated:: 1.2
            `resolution` will be removed in 1.4 and is replaced with `voxelsize`
        """
        msg = "Attribute 'resolution' is deprecated, use 'voxelsize' attribute instead!"
        print(DeprecationWarning(msg))
        return self._voxelsize

    @resolution.setter
    def resolution(self, voxelsize):
        """
        Ensure backward compatibility with older openalea.image package.

        .. deprecated:: 1.2
            `resolution` will be removed in 1.4 and is replaced with `voxelsize`
        """
        msg = "Attribute 'resolution' is deprecated, use 'voxelsize' attribute instead!"
        print(DeprecationWarning(msg))
        self.voxelsize = voxelsize

    # --------------------------------------------------------------------------
    #
    # SpatialImage methods :
    #
    # --------------------------------------------------------------------------

    def is_isometric(self):
        """Test image isometry.

        Tests if the voxelsize values are the same for every axis or not.

        Returns
        -------
        is_iso : bool
            True if the image is isometric, else ``False``.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize anisotropic random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image(voxelsize=[0.5, 0.5, 1.])
        >>> img.is_isometric()
        False
        >>> # Trasnform to isotropic 3D SpatialImage:
        >>> img_iso = img.isometric_resampling()
        >>> img_iso.is_isometric()
        True

        """
        vxs = [self.voxelsize[0]] * self.ndim
        return np.allclose(vxs, self.voxelsize, atol=1.e-6)

    def is2D(self):
        """Returns True if the SpatialImage is 2D, else ``False``.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> img.is2D()
        False
        >>> # Take the first z-slice:
        >>> img2d = img[:, :, 0]
        >>> img2d.is2D()
        True

        """
        return self.get_dim() == 2

    def is3D(self):
        """Returns True if the SpatialImage is 3D, else ``False``.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> img.is3D()
        True

        """
        return self.get_dim() == 3

    def is_available_types(self, dtype):
        """Test if the given type is available.

        Parameters
        ----------
        dtype : str
            name of the type to find in DICT_TYPES

        """
        return dtype in DICT_TYPES.keys()

    # --------------------------------------------------------------------------
    #
    # GETTERs & SETTERs:
    #
    # --------------------------------------------------------------------------

    def get_shape(self, axis=None):
        """Return the shape of the image or of an axis.

        Parameters
        ----------
        axis : str, in ['x', 'y'(, 'z')]
            if specified, axis to use for shape dimension, else the shape of the image

        Returns
        -------

        """
        if axis is None:
            return cp.copy(self.shape)
        else:
            return cp.copy(self.shape[AXIS_ORDER[axis]])

    def get_x_dim(self):
        """Returns the dimension (shape) of the x-axis.

        Returns
        -------
        int
            the dimension of the x-axis
        """
        return self.get_shape('x')

    def get_y_dim(self):
        """Returns the dimension (shape) of the y-axis.

        Returns
        -------
        int
            the dimension of the y-axis
        """
        return self.get_shape('y')

    def get_z_dim(self):
        """Returns the dimension (shape) of the z-axis.

        Returns
        -------
        int
            the dimension of the z-axis
        """
        return self.get_shape('z')

    def set_slice(self, slice_id, array, axis='z'):
        """Set the values of an axis slice to given array.

        Parameters
        ----------
        slice_id : int
            slice number to modify
        array : np.array
            values that should replace those from given slice
        axis : str, in ['x', 'y', 'z']
            axis to use for slicing

        """
        sl = self.get_slice(slice_id, axis)
        assert sl.shape == array.shape

        if axis == 'x':
            self[slice_id, :, :] = array
        elif axis == 'y':
            self[:, slice_id, :] = array
        else:
            self[:, :, slice_id] = array

    def get_slice(self, slice_id, axis='z'):
        """Returns a SpatialImage with only one slice for given axis.

        Parameters
        ----------
        slice_id : int
            slice to return
        axis : str, in ['x', 'y', 'z']
            axis to use for slicing

        Returns
        -------
        SpatialImage
            2D SpatialImage with only the required slice

        Raises
        ------
        ValueError
            if image is not 3D.
            if slice does not exists, **ie.** should satify: `0 < slice < max(len(axis))`.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random [40, 40, 10] (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> # Taking an existing z-slice from a 3D image works fine:
        >>> img_z5 = img.get_slice(5, 'z')
        >>> # Taking an existing x-slice from a 3D image works fine:
        >>> img_x20 = img.get_slice(20, 'x')

        >>> # Taking an NON existing z-slice from a 3D image raises an error:
        >>> img_z50 = img.get_slice(50, 'z')
        ValueError: Required z-slice (50) does not exists (max: 10)

        >>> # Taking a z-slice from a 2D image raises an error:
        >>> img_z5.get_slice(5, 'z')
        ValueError: Can only extract z-slice from 3D images!

        >>> # Taking existing x-slice & y-slice from a 2D image works fine:
        >>> img_z5.get_slice(5, 'x')
        >>> img_z5.get_slice(5, 'y')

        """
        if axis == 'z':
            # Make sure we have a 3D image:
            try:
                assert self.is3D()
            except AssertionError:
                raise ValueError("Can only extract z-slice from 3D images!")

        axis_id = AXIS_ORDER[axis]
        # Make sure this slice exists:
        max_slice = self.shape[axis_id]
        if isinstance(slice_id, int):
            try:
                assert slice_id <= max_slice
            except AssertionError:
                msg = "Required {}-slice ({}) does not exists (max: {})"
                raise ValueError(msg.format(axis, slice_id, max_slice))
        else:
            try:
                assert np.max(slice_id) <= max_slice
            except AssertionError:
                msg = "Required {}-slice range ({}-{}) does not exists (max: {})"
                raise ValueError(msg.format(axis, np.min(slice_id), np.max(slice_id), max_slice))

        vxs = self.voxelsize
        ori = self.origin
        md = self.metadata
        # Remove value corresponding to returned slice axis:
        vxs.pop(axis_id)
        ori.pop(axis_id)
        # Clear image tags, they will be recomputed by SpatialImage.__new__
        for attr in IMAGE_MD_TAGS + ['extent']:
            md.pop(attr)

        if axis == 'x':
            return SpatialImage(self.get_array()[slice_id, :, :],
                                origin=ori, voxelsize=vxs, metadata=md)
        elif axis == 'y':
            return SpatialImage(self.get_array()[:, slice_id, :],
                                origin=ori, voxelsize=vxs, metadata=md)
        else:
            return SpatialImage(self.get_array()[:, :, slice_id],
                                origin=ori, voxelsize=vxs, metadata=md)


    def get_x_slice(self, x_slice):
        """"Returns a 2D SpatialImage with only one x-slice.

        Parameters
        ----------
        x_slice : int
            x-slice to return

        Returns
        -------
        SpatialImage
            2D SpatialImage with only the required slice

        Raises
        ------
        ValueError
            if image is not 3D.
            if x-slice does not exists.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> # Taking an existing x-slice from a 3D image works fine:
        >>> img_z5 = img.get_x_slice(5)

        >>> # Taking an NON existing x-slice from a 3D image raises an error:
        >>> img_z50 = img.get_x_slice(50)
        ValueError: Required x-slice (50) does not exists (max: 10)

        >>> # Taking a x-slice from a 2D image raises an error:
        >>> img_z5.get_x_slice(5)
        ValueError: Can only extract x-slice from 3D images!

        """
        return self.get_slice(x_slice, axis='x')

    def get_y_slice(self, y_slice):
        """"Returns a 2D SpatialImage with only one y-slice.

        Parameters
        ----------
        y_slice : int
            y-slice to return

        Returns
        -------
        SpatialImage
            2D SpatialImage with only the required slice

        Raises
        ------
        ValueError
            if image is not 3D.
            if y-slice does not exists.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> # Taking an existing y-slice from a 3D image works fine:
        >>> img_z5 = img.get_y_slice(5)

        >>> # Taking an NON existing y-slice from a 3D image raises an error:
        >>> img_z50 = img.get_y_slice(50)
        ValueError: Required y-slice (50) does not exists (max: 10)

        >>> # Taking a y-slice from a 2D image raises an error:
        >>> img_z5.get_y_slice(5)
        ValueError: Can only extract y-slice from 3D images!

        """
        return self.get_slice(y_slice, axis='y')

    def get_z_slice(self, z_slice):
        """Returns a 2D SpatialImage with only one z-slice.

        Parameters
        ----------
        z_slice : int
            z-slice to return

        Returns
        -------
        SpatialImage
            2D SpatialImage with only the required slice

        Raises
        ------
        ValueError
            if image is not 3D.
            if z-slice does not exists.

        Examples
        --------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> # Taking an existing z-slice from a 3D image works fine:
        >>> img_z5 = img.get_z_slice(5)

        >>> # Taking an NON existing z-slice from a 3D image raises an error:
        >>> img_z50 = img.get_z_slice(50)
        ValueError: Required z-slice (50) does not exists (max: 10)

        >>> # Taking a z-slice from a 2D image raises an error:
        >>> img_z5.get_z_slice(5)
        ValueError: Can only extract z-slice from 3D images!

        """
        return self.get_slice(z_slice, axis='z')

    def get_available_types(self):
        """Print the available bits type dictionary.

        See Also
        --------
        DICT_TYPES : dict
            dictionary of available bits type

        Returns
        -------
        dict
            dictionary of available bits type

        """
        return DICT_TYPES

    def get_array(self, dtype=None):
        """Get a ``numpy.ndarray`` from a ``SpatialImage``

        Returns
        -------
        numpy.ndarray
            ``SpatialImage`` array

        Example
        -------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> array = img.get_array()
        >>> isinstance(array, SpatialImage)
        False
        >>> isinstance(array, np.ndarray)
        True

        """
        if dtype is None:
            return np.array(self, copy=True)
        else:
            return np.array(self, copy=True).astype(dtype)

    def get_dim(self):
        """Get ``SpatialImage`` number of dimensions (2D or 3D image)

        Returns
        -------
        int
            ``SpatialImage`` dimensionality

        Example
        -------
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> img.get_dim()
        3

        """
        return self.ndim

    def get_min(self):
        """Get ``SpatialImage`` minimum value

        Returns
        -------
        *val*, dtype to self.dtype
            ``SpatialImage`` minimum value.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.get_min()
        1

        """
        try:
            assert self._min is not None
        except AssertionError:
            self._min = np.array([self.min()]).astype(self.dtype)[0]
            self.metadata.update({'min': self._min})
        return self._min

    def get_max(self):
        """Get ``SpatialImage`` maximum value.

        Returns
        -------
        *val*, dtype to self.dtype
            ``SpatialImage`` maximum value.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.get_max()
        1

        """
        try:
            assert self._max is not None
        except AssertionError:
            self._max = np.array([self.max()]).astype(self.dtype)[0]
            self.metadata.update({'max': self._max})
        return self._max

    def get_mean(self):
        """Get ``SpatialImage`` mean value.

        Returns
        -------
        *val*, dtype to self.dtype
            ``SpatialImage`` mean value.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.get_mean()
        1

        """
        try:
            assert self._mean is not None
        except AssertionError:
            self._mean = np.array([self.mean()]).astype(self.dtype)[0]
            self.metadata.update({'mean': self._mean})
        return self._mean

    def get_pixel(self, idx):
        """Get ``SpatialImage`` pixel value at given array coordinates.

        Parameters
        ----------
        idx : list
            indices as list of integers

        Returns
        -------
        value
            pixel value

        Raises
        ------
        TypeError
            If the given `idx` is not a list.
        ValueError
            If the number of `idx` is wrong, should be the image dimensionality.
            If the `idx` coordinates are not within the image boundaries.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.get_pixel([1,1])
        1

        """
        check_type(idx, 'idx', list)
        # Check we have the right number of `idx`:
        ndim = self.get_dim()
        try:
            assert len(idx) == ndim
        except AssertionError:
            msg = "Input 'idx' must have a lenght equal to the image dimensionality!"
            raise ValueError(msg)
        # Check the `ids` are within the image boundaries:
        sh = self.shape
        ids = range(0, ndim)
        try:
            assert all([(idx[i] >= 0) & (idx[i] < sh[i] + 1) for i in ids])
        except AssertionError:
            err = "Input 'idx' {} are not within the image shape {}!"
            raise ValueError(err.format(idx, self.shape))

        if ndim == 2:
            pix_val = self[idx[0], idx[1]]
        else:
            pix_val = self[idx[0], idx[1], idx[2]]

        return pix_val

    def set_pixel(self, idx, value):
        """Change ``SpatialImage`` pixel|voxel value at given array coordinates.

        Parameters
        ----------
        idx : list
            array coordinates as list of integers
        value : array.dtype
            new value for the pixel|voxel, should be compatible with the array dtype

        Raises
        ------
        TypeError
            If the given `idx` is not a list.
        ValueError
            If the number of `idx` is wrong, should be the image dimensionality.
            If the `idx` coordinates are not within the image boundaries.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((3,3), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> image.set_pixel([1, 1], 2)
        >>> image.get_array()
        array([[1, 1, 1],
               [1, 2, 1],
               [1, 1, 1]], dtype=uint8)
        >>> # Trying to set a value not compatible with the array.dtype:
        >>> image.set_pixel([2, 2], 0.5)
        >>> image.get_array()
        array([[1, 1, 1],
               [1, 2, 1],
               [1, 1, 0]], dtype=uint8)
        >>> # Trying to set a value not compatible with the array.dtype:
        >>> image.set_pixel([0, 0], -6)
        >>> image.get_array()
        array([[250,   1,   1],
               [  1,   2,   1],
               [  1,   1,   0]], dtype=uint8)

        """
        check_type(idx, 'idx', list)
        # Check we have the right number of `idx`:
        ndim = self.get_dim()
        try:
            assert len(idx) == ndim
        except AssertionError:
            msg = "Input 'idx' must have a lenght equal to the image dimensionality!"
            raise ValueError(msg)
        # Check the `ids` are within the image boundaries:
        sh = self.shape
        ids = range(0, ndim)
        try:
            assert all([(idx[i] >= 0) & (idx[i] < sh[i] + 1) for i in ids])
        except AssertionError:
            err = "Input 'idx' {} are not within the image shape {}!"
            raise ValueError(err.format(idx, self.shape))

        if ndim == 2:
            self[idx[0], idx[1]] = value
        else:
            self[idx[0], idx[1], idx[2]] = value
        return

    def get_region(self, region):
        """Extract a region using list of start & stop indices.

        There should be two values per image dimension in 'indices'.
        If the image is 3D and in one dimension, the start and stop are differ
        by one (on layer of voxels), the image is transformed to 2D!

        Parameters
        ----------
        region : list
            indices as list of integers

        Returns
        -------
        SpatialImage
            output ``SpatialImage``

        Raises
        ------
        TypeError
            If the given `region` is not a list.
        ValueError
            If the number of `region` is wrong, should be twice the image dimensionality.
            If the `region` coordinates are not within the image boundaries.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> region = [1, 3, 1, 3]
        >>> out_sp_img = image.get_region(region)
        >>> isinstance(out_sp_img, SpatialImage)
        True

        """
        # TODO: use slice instead of 'indices' list?
        check_type(region, 'idx', list)
        # Check we have the right number of `idx`:
        ndim = self.get_dim()
        try:
            assert len(region) == 2 * ndim
        except AssertionError:
            msg = "Input 'idx' must have a lenght equal to twice the number of dimension of the image!"
            raise ValueError(msg)
        # Check the `ids` are within the image boundaries:
        sh = self.shape
        ids = range(0, 2 * ndim, 2)
        try:
            assert all([(region[i] >= 0) & (region[i + 1] < sh[i // 2] + 1) for i in ids])
        except AssertionError:
            err = "Input 'idx' {} are not within the image shape {}!"
            raise ValueError(err.format(region, self.shape))

        bbox = [slice(region[i], region[i + 1]) for i in ids]
        tmp_arr, tmp_vox = self.get_array(), self.voxelsize
        reg_val = tmp_arr[tuple(bbox)]
        if ndim == 3 & 1 in reg_val.shape:  # 3D --> 2D
            if reg_val.shape[0] == 1:
                reg_val = np.squeeze(reg_val, axis=(0,))
                tmp_vox = [tmp_vox[1], tmp_vox[2]]
            elif reg_val.shape[1] == 1:
                reg_val = np.squeeze(reg_val, axis=(1,))
                tmp_vox = [tmp_vox[0], tmp_vox[2]]
            elif reg_val.shape[2] == 1:
                reg_val = np.squeeze(reg_val, axis=(2,))
                tmp_vox = [tmp_vox[0], tmp_vox[1]]

        out_sp_img = SpatialImage(reg_val, voxelsize=tmp_vox)
        return out_sp_img

    def set_region(self, region, value):
        """Replace a region of the image by given value.

        Parameters
        ----------
        region : list
            indices as list of integers
        value : array.dtype or ndarray
            new value for the selected pixels, type of ``SpatialImage`` array

        Returns
        -------
        SpatialImage
            ``SpatialImage`` instance

        Raises
        ------
        TypeError
            If the given `region` is not a list.
        ValueError
            If the number of `region` is wrong, should be twice the image dimensionality.
            If the `region` coordinates are not within the image boundaries.
            If the `region` and `value` shape missmatch.
            If the given `value` is not of the same dtype than the array or not a ndarray.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(test_array)
        >>> region = [1,3,1,3]
        >>> image.set_region(region, 3)
        >>> image.get_array()

        """
        check_type(region, 'idx', list)
        # Check we have the right number of `idx`:
        ndim = self.get_dim()
        try:
            assert len(region) == 2 * ndim
        except AssertionError:
            msg = "Input 'region' must have a lenght equal to twice the number of dimension of the image!"
            raise ValueError(msg)
        # Check the `region` is within the image boundaries:
        sh = self.shape
        ids = range(0, 2 * ndim, 2)
        try:
            assert all([(region[i] >= 0) & (region[i + 1] < sh[i // 2] + 1) for i in ids])
        except AssertionError:
            err = "Input 'idx' {} are not within the image shape {}!"
            raise ValueError(err.format(region, self.shape))

        if isinstance(value, np.ndarray):
            region_shape = [region[i + 1] - region[i] for i in ids]
            try:
                np.testing.assert_equal(region_shape, value.shape)
            except AssertionError:
                msg = "Given region shape ({}) and value shape ({}) missmatch!"
                raise ValueError(msg.format(region_shape, value.shape))

        dtype = self.dtype
        if isinstance(value, np.ndarray):
            if ndim == 2:
                self[region[0]: region[1], region[2]: region[3]] = value[:, :].astype(dtype)
            else:
                self[region[0]: region[1], region[2]: region[3], region[4]: region[5]] = value[:, :, :].astype(dtype)
        elif isinstance(value, dtype):
            if ndim == 2:
                self[region[0]: region[1], region[2]: region[3]] = value
            else:
                self[region[0]: region[1], region[2]: region[3], region[4]: region[5]] = value
        else:
            msg = "The given `value` is not of the same dtype than the array or not a ndarray"
            raise ValueError(msg)

        return

    # --------------------------------------------------------------------------
    #
    # SpatialImage transformation functions:
    #
    # --------------------------------------------------------------------------

    def astype(self, dtype, **kwargs):
        """Copy of the SpatialImage with updated data type.

        Notes
        -----
        kwargs are passed to numpy 'astype' method.

        Parameters
        ----------
        dtype : str
            new type of data to apply

        Returns
        -------
        SpatialImage
            image with the new data type

        Examples
        --------
        >>> import numpy as np
        >>> from timagetk.test import make_random_spatial_image
        >>> # Initialize a random (uint8) 3D SpatialImage:
        >>> img = make_random_spatial_image()
        >>> # Check the type:
        >>> img.dtype
        dtype('uint8')
        >>> # Convert it to 16bits unsigned integers:
        >>> img16 = img.astype(np.uint16)
        >>> # Check the type:
        >>> img16.dtype
        dtype('uint16')

        """
        # - Convert the numpy array:
        array = self.get_array().astype(dtype, **kwargs)
        # - Get 'origin', 'voxelsize' & 'metadata':
        origin = self.origin
        voxelsize = self.voxelsize
        md = self.metadata
        # - Update metadata 'dtype' to new type:
        md['dtype'] = dtype

        return SpatialImage(array, origin=origin, voxelsize=voxelsize,
                            metadata=md)

    def to_2D(self):
        """Convert a 3D SpatialImage to a 2D SpatialImage.

        Conversion is possible only if there is a "flat" axis (ie. with only
        one slice in this axis).

        Returns
        -------
        SpatialImage
            the 2D SpatialImage

        """
        if self.is3D() and 1 in self.shape:
            voxelsize, shape, array = self.voxelsize, self.shape, self.get_array()
            ori, md = self.origin, self.metadata
            if shape[0] == 1:
                new_arr = np.squeeze(array, axis=(0,))
                new_vox = [voxelsize[1], voxelsize[2]]
                new_ori = [ori[1], ori[2]]
            elif shape[1] == 1:
                new_arr = np.squeeze(array, axis=(1,))
                new_vox = [voxelsize[0], voxelsize[2]]
                new_ori = [ori[0], ori[2]]
            else:
                new_arr = np.squeeze(array, axis=(2,))
                new_vox = [voxelsize[0], voxelsize[1]]
                new_ori = [ori[0], ori[1]]
            out_sp_img = SpatialImage(new_arr, voxelsize=new_vox,
                                      origin=new_ori, metadata=md)
            return out_sp_img
        elif self.is2D():
            print("This image is already a 2D image!")
            return
        else:
            print('This 3D SpatialImage can not be reshaped to 2D.')
            return

    def to_3D(self):
        """Convert a 2D SpatialImage to a 3D SpatialImage.

        Obtained 3D SpatialImage has a "flat" z-axis (ie. with only one
        slice in this axis).

        Returns
        -------
        SpatialImage
            the 3D SpatialImage

        """
        if self.is2D():
            voxelsize, shape, array = self.voxelsize, self.shape, self.get_array()
            ori, md = self.origin, self.metadata
            new_arr = np.reshape(array, (shape[0], shape[1], 1))
            new_vox = [voxelsize[0], voxelsize[1], 1.0]
            new_ori = [ori[0], ori[1], 0]
            out_sp_img = SpatialImage(new_arr, voxelsize=new_vox,
                                      origin=new_ori, metadata=md)
            return out_sp_img
        else:
            print('This SpatialImage is not 2D.')
            return

    def transpose(self, *axes):
        """

        Parameters
        ----------
        axes

        Returns
        -------

        """
        # TODO!
        return NotImplementedError

    def revert_axis(self, axis):
        """Revert given axis.

        Notes
        -----
        x, y, or z axis if the object is in 3D, limited to x and y in 2D.

        Parameters
        ----------
        axis : str
            can be either 'x', 'y' or 'z' (if 3D)

        Returns
        -------
        SpatialImage
            reverted array for selected axis

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
        >>> image = SpatialImage(test_array, voxelsize=[0.5, 0.5, 0.5])
        >>> print(image.get_array())
        >>> image.revert_axis(axis='y')
        """
        if self.is2D():
            return self._revert_2d(self, axis)
        else:
            return self._revert_3d(self, axis)

    @staticmethod
    def _revert_3d(array, axis):
        """Revert x, y, or z axis of the given array.

        Parameters
        ----------
        array : numpy.array
            array for which to revert an axis
        axis : str in {'x', 'y', 'z'}
            array axis to revert

        Returns
        -------
        numpy.array
            reverted array for selected axis

        Raises
        ------
        ValueError
            If given `axis` is not in {'x', 'y', 'z'}

        """
        if axis == 'x':
            return array[::-1, :, :]
        elif axis == 'y':
            return array[:, ::-1, :]
        elif axis == 'z':
            return array[:, :, ::-1]
        else:
            raise ValueError("Unknown axis '{}' for a 3D array.".format(axis))

    @staticmethod
    def _revert_2d(array, axis):
        """Revert x or y axis of the given array.

        Parameters
        ----------
        array : numpy.array
            array for which to revert an axis
        axis : str in {'x', 'y'}
            array axis to revert

        Returns
        -------
        numpy.array
            reverted array for selected axis

        Raises
        ------
        ValueError
            If given `axis` is not in {'x', 'y'}

        """
        if axis == 'x':
            return array[::-1, :]
        elif axis == 'y':
            return array[:, ::-1]
        else:
            raise ValueError("Unknown axis '{}' for a 2D array.".format(axis))

    # --------------------------------------------------------------------------
    #
    # Image comparisons
    #
    # --------------------------------------------------------------------------

    def equal(self, sp_img, error=EPS, method='max_error'):
        """Equality test between two ``SpatialImage``.

        Uses array equality and metadata matching.

        Parameters
        ----------
        sp_img : SpatialImage
            another ``SpatialImage`` instance to test for array equality
        error : float, optional
            maximum difference accepted between the two arrays (default=EPS)
        method : str in {'max_error', 'cum_error'}, optional
            type of "error measurement", choose among (default='max_error'):
              - "max_error": max difference accepted for a given pixel
              - "cum_error": max cumulative (sum) difference for the whole array

        Returns
        -------
        bool
            True/False if (array and metadata) are equal/or not

        Notes
        -----
        Metadata equality test compare defined self.metadata keys to their
        counterpart in 'sp_img'. Hence, a missing key in 'sp_im' or a different
        value will return ``False``.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image_1 = SpatialImage(test_array)
        >>> image_1.equal(image_1)
        True
        >>> image_2 = SpatialImage(test_array, voxelsize=[0.5,0.5])
        >>> image_1.equal(image_2)
        SpatialImages metadata are different.
        False
        >>> image_2[1, 1] = 2
        >>> image_1.equal(image_2)
        Max difference between arrays is greater than '1e-09'.
        SpatialImages metadata are different.
        False

        """
        equal = False
        # - Test array equality:
        t_arr = self.equal_array(sp_img, error=error, method=method)
        # - Test metadata equality:
        md_ref = self.metadata
        md = sp_img.metadata
        t_met = all([k in md and v == md[k] for k, v in md_ref.items()])

        # - Combine test and print when fail:
        if t_arr and t_met:
            equal = True
        if not t_arr:
            m = 'Max' if method == 'max_error' else 'Cumulative'
            print("{} difference between arrays is greater than '{}'.".format(
                m, error))
        if not t_met:
            print("SpatialImages metadata are different.")

        return equal

    def equal_array(self, sp_img, error=EPS, method='max_error'):
        """
        Test array equality between two ``SpatialImage``.

        Parameters
        ----------
        sp_img : SpatialImage
            another ``SpatialImage`` instance to test for array equality
        error : float, optional
            maximum difference accepted between the two arrays, should be
            strictly inferior to this value to return True, default: EPS
        method : str in {'max_error', 'cum_error'}, optional
            type of "error measurement", choose among:
              - "max_error": max difference accepted for a given pixel
              - "cum_error": max cumulative (sum) difference for the whole array

        Returns
        -------
        bool
            True/False if arrays are equal/or not

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image_1 = SpatialImage(test_array)
        >>> image_1.equal_array(image_1)
        True
        >>> # - Changing voxelsize does not affect array equality test:
        >>> image_2 = SpatialImage(test_array, voxelsize=[0.5,0.5])
        >>> image_1.equal_array(image_2)
        True
        >>> # - Changing array value does affect array equality test:
        >>> image_2[0, 0] = 0
        >>> image_1.equal_array(image_2)
        False
        >>> # - Changing accepted max difference affect array equality test:
        >>> image_1.equal_array(image_2, error=2)
        True

        """
        check_type(sp_img, 'sp_img', SpatialImage)

        try:
            assert method in EQ_METHODS
        except AssertionError:
            msg = "Unknown method '{}', should be in {}."
            raise ValueError(msg.format(method, EQ_METHODS))

        # - Starts by testing the shapes are equal:
        if self.shape != sp_img.shape:
            msg = "Given 'sp_img' has a different shape ({}) than this one ({})!"
            print(msg.format(sp_img.shape, self.shape))
            return False

        # - Test array equality:
        ori_type = str(self.dtype)
        if ori_type.startswith('u'):
            # unsigned case is problematic for 'np.subtract'
            tmp_type = DICT_TYPES[ori_type[1:]]
        else:
            tmp_type = ori_type
        # - Compute the difference between the two arrays:
        out_img = np.abs(np.subtract(self, sp_img).astype(tmp_type)).astype(
            ori_type)
        # - Try to find non-null values in this array:
        non_null_idx = np.nonzero(out_img)
        if len(non_null_idx[0]) != 0:
            non_null = out_img[non_null_idx]
            if method == 'max_error':
                equal = bool(np.max(non_null) < error)
            else:
                equal = bool(np.sum(non_null) < error)
        else:
            equal = True

        return equal
