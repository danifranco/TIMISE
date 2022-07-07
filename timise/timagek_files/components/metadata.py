# -*- python -*-
# -*- coding: utf-8 -*-
#
#       components.labelled_image
#
#       Copyright 2018 CNRS- ENS - INRIA
#
#       File author(s):
#           Gregoire Malandain <gregoire.malandain@inria.fr>
#
#       File maintainer(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       Distributed under the INRIA License.
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

"""
This module implement timagetk metadata functionnalities.

Use the function to add parameters to metadata when performing image
modifications.

Metadata specifications:

* Use 'timagetk' as the metadata main root key to store these information
* Use 'class' key for class information
* Use a counter to sort operation orders
* Under each number save a dictionary with details of the modification or
about algorithm applied, such as its name and their parameters or the name of
the transformation (or even the transformation if linear!).

Examples
--------
>>> import numpy as np
>>> from timagetk.components import SpatialImage
>>> test_array = np.ones((5,5,10), dtype=np.uint8)
>>> image_1 = SpatialImage(test_array, voxelsize=[1., 1., 2.])
>>> image_iso = image_1.isometric_resampling()
>>> image_iso.metadata['timagetk']
{(1, 'resample_isotropic'):
    {'params': {},
    'called':
        {(1, 'resample'):
            {'params':
                {'image': ['', ''],
                'option': 'gray',
                'voxelsize': [1.0, 1.0, 1.0]},
            'called':
                {(1, 'apply_trsf'):
                    {'dtype': None,
                    'image': ['', ''],
                    'param_str_1': '-linear -template-dim 5 5 20 -template-voxel 1.0 1.0 1.0',
                    'param_str_2': ' -resize -interpolation linear',
                    'template_img': ['', ''],
                    'trsf': None}
                }
            }
        }
    }
}
"""

import os
import sys
import logging
import numpy as np

from ..util import clean_type
from ..wrapping.bal_trsf import BalTrsf

def get_func_name(level=1):
    """Get the name of the function it is embedded in.

    Parameters
    ----------
    level : int, optional
        stack level, set to 1 to get the name of the function where
        ``get_func_name`` is embedded in, or more to go higher in the stack.

    Returns
    -------
    str
        name of the function

    References
    ----------
    [#].. https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string-in-python
    """
    func_name = sys._getframe(level).f_code.co_name
    if func_name.startswith('<') and func_name.endswith('>'):
        return None
    else:
        return sys._getframe(level).f_code.co_name


def get_func_params(level=1):
    """Get the parameters of the function it is embedded in.

    Parameters
    ----------
    level : int, optional
        stack level, set to 1 to get the name of the function where
        ``get_func_name`` is embedded in, or more to go higher in the stack.

    Returns
    -------
    dict
        dictionary with the name of the parameters and their values at given ``level``

    References
    ----------
    [#].. https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string-in-python
    """
    frame = sys._getframe(level)
    var_names = get_func_params_names_from_frame(frame)
    # - Make a dictionary of name and values:
    params = {name: get_func_param_value_from_frame(frame, name) for name in
              var_names if name not in ['verbose']}
    return params


def get_func_params_names_from_frame(frame):
    """Get the parameters' name of the function in the given ``frame``.

    Parameters
    ----------
    level : int, optional
        stack level, set to 1 to get the name of the function where
        ``get_func_name`` is embedded in, or more to go higher in the stack.

    Returns
    -------
    str
        name of the function

    References
    ----------
    [#].. https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string-in-python
    """
    fc = frame.f_code
    return fc.co_varnames[:fc.co_argcount]


def get_func_param_value_from_frame(my_frame, v_name):
    """Get the value of a variable in the given ``frame``.

    Notes
    -----
    If a SpatialImage instance is found, we return the metadata
    'filepath'/'filename' instead of the whole object!
    If a BalTrsf is found and it is linear, we return the transformation matrix
    as a numpy array.

    Parameters
    ----------
    my_frame : frame
        frame with a variable ``v_name``
    v_name : str
        name of a variable in the given frame ``my_frame``

    Returns
    -------
    any
        value of the variable

    References
    ----------
    """
    from ..spatial_image import SpatialImage
    # from ..wrapping.c_bal_trsf import BalTrsf

    val = my_frame.f_locals.get(v_name)

    # - When a SpatialImage is given, do not return the whole object but its 'filename' & 'filepath':
    if isinstance(val, SpatialImage):
        stype = clean_type(val)
        fun_name = my_frame.f_code.co_name

        # - Try to get the path:
        try:
            filepath = val.filepath
        except AttributeError:
            msg = "Could not find a 'filepath' in {} given to ``{}``!"
            logging.warning(msg.format(stype, fun_name))
            filepath = "unknown"

        # - Try to get the name:
        try:
            filename = val.filename
        except AttributeError:
            msg = "Could not find a 'filename' in {} given to ``{}``!"
            logging.warning(msg.format(stype, fun_name))
            filename = "unknown"

        # - Join the path and name:
        val = {
            'path': os.path.join(filepath, filename),
            'titk_md': val.metadata.get('timagetk', None)
        }

    # - When a BalTrsf is given and it is linear, return the matrix as an array:
    if isinstance(val, BalTrsf) and val.is_linear():
        val = val.mat.to_np_array()

    return val


def get_last_id(md_dict):
    """Get the last id in the metadata dictionary.

    Parameters
    ----------
    md_dict : dict
        metadata dictionary

    Returns
    -------
    id : int
        last id if found else 1
    """
    ids = md_dict.keys()
    try:
        ids.remove('class')
    except ValueError:
        warn_msg = "Initializing from a metadata dictionary without 'class' entry in 'timagetk' metadata!"
        logging.debug(warn_msg)
    if ids == []:
        msg = "Initialising from a 'timagetk' metadata dict with no keys!"
        logging.debug(msg)
        return 0

    lid = sorted([i[0] for i in ids])[-1]
    if lid == 0:
        return 0
    else:
        return lid


def single_fn_md(lid, fname, fpars):
    """Metadata format when using low level-function.

    Use it when no higher level function is known.

    Parameters
    ----------
    lid : int
        last id
    fname : str
        name of the function
    fpars : dict
        parameter dictionary associated to function call

    Returns
    -------
    dict
        {(lid+1, fname): fpars}
    """
    return {(lid + 1, fname): fpars}


def called_fn_md(lid, hfname, fname, fpars):
    """Metadata format when using function calling others sub-processes.

    Parameters
    ----------
    lid : int
        last id
    hfname : str
        name of the function of higher level calling the function
    fname : str
        name of the function
    fpars : dict
        parameter dictionary associated to function call

    Returns
    -------
    dict
       {(lid+1, hfname): {'params': {}, 'called': {(1, fname): fpars}}}
    """
    return {(lid + 1, hfname): {'params': {}, 'called': {(1, fname): fpars}}}


def add2md(image, extra_params=None, from_level=2):
    """Add parameters to 'timagetk' metadata attribute of given image.

    Parameters
    ----------
    image : SpatialImage, LabelledImage, TissueImage
        image with a metadata attribute
    extra_params : dict, optional
        dictionary of additional parameter to save to the metadata.

    Returns
    -------
    image : SpatialImage, LabelledImage, TissueImage
        image with updated metadata dictionary
    """
    # - Get the metadata dictionary:
    try:
        md_dict = image.metadata['timagetk']
    except KeyError:  # TODO: change this to error later
        warn_msg = "No 'timagetk' entry in '{}' instance metadata!"
        logging.debug(warn_msg.format(clean_type(image)))
        md_dict = {}
        image.metadata.update({'timagetk': md_dict})
    except AttributeError:
        raise TypeError("No metadata attribute is found!")

    # - Get the function name:
    fun_name = get_func_name(level=from_level)
    logging.debug("Found function name: '{}'".format(fun_name))
    # - Create new metadata from function name and used parameters:
    fun_param = get_func_params(level=from_level)
    logging.debug("Found function parameters: '{}'".format(fun_param))
    if extra_params is not None:
        # - Do not save 'verbose' kwarg:
        try:
            extra_params.pop('verbose')
        except KeyError:
            pass
        # - Update the function parameters dict:
        fun_param.update(extra_params)

    # - Find the last id:
    lid = get_last_id(md_dict)
    logging.debug("Got last id: {}".format(lid))

    higher_fun_name = get_func_name(level=from_level + 1)
    md = {}
    if higher_fun_name is None:
        logging.debug(
            "No higher level function calling '{}'!".format(higher_fun_name))
        if (lid, fun_name) in md_dict and md_dict[(lid, fun_name)]['params'] == {}:
            msg = "Updating 'params' for ({}, {})."
            logging.debug(msg.format(lid, fun_name))
            image.metadata['timagetk'][(lid, fun_name)][
                'params'] = fun_param  # update params
        else:
            # - Low-level function case:
            md = single_fn_md(lid, fun_name, fun_param)
    else:
        if (lid, higher_fun_name) in md_dict:
            msg = "Appending a new sub-process '{}' called by '{}'."
            logging.debug(msg.format(fun_name, higher_fun_name))
            # - Case where same ``higher_fun_name`` call multiple sub-processes:
            assert md_dict[(lid, higher_fun_name)]['params'] == {}
            # Need to know the last id in already 'called' processes to update properly:
            called_md = md_dict[(lid, higher_fun_name)]['called']
            clid = get_last_id(called_md)
            md = called_md.update(single_fn_md(clid, fun_name, fun_param))
        elif (lid, fun_name) in md_dict:
            # - Case where a higher level function already called other processes that are registered
            if md_dict[(lid, fun_name)]['params'] == {}:
                msg = "Updating 'params' for ({}, {})."
                logging.debug(msg.format(lid, fun_name))
                image.metadata['timagetk'][(lid, fun_name)][
                    'params'] = fun_param  # update params
            msg = "Moving '{}' as called by '{}'!"
            logging.debug(msg.format(fun_name, higher_fun_name))
            sub_md = md_dict.pop((lid, fun_name))
            md = {(lid, higher_fun_name): {'params': {},
                                           'called': {(1, fun_name): sub_md}}}
        else:
            msg = "Registering a new sub-process '{}' called by '{}'."
            logging.debug(msg.format(fun_name, higher_fun_name))
            # - Case where a new process is registered:
            md = called_fn_md(lid, higher_fun_name, fun_name, fun_param)

    if md != {} and md is not None:
        logging.debug("Updating metadata with following dict:\n{}".format(md))
        # - Update the 'timagetk' metadata:
        image._metadata['timagetk'].update(md)
    logging.debug("New metadata:\n{}\n".format(image.metadata['timagetk']))
    return image


def sort_ops(md):
    """Return a list of sorted operations from metadata root.

    Parameters
    ----------
    md : dict
        metadata dictionary.

    Returns
    -------
    list
        list of sorted operations from metadata root
    """
    ops_tuple = md.keys()
    # Remove 'class' key, not referring to list of operations:
    if 'class' in ops_tuple:
        ops_tuple.remove('class')
    # TODO: should test that key are len-2 tuples? -> more generic filtering!
    if len(ops_tuple) == 0:
        return None
    elif len(ops_tuple) == 1:
        return [ops_tuple[0][1]]
    else:
        ops_id = [op[0] for op in ops_tuple]
        ops_name = [op[1] for op in ops_tuple]

    return ops_name[np.argsort(ops_id)]


def get_params_called(md, op_id, op_name):
    """Get 'params' and 'called' values in metadata dictionary.

    Parameters
    ----------
    md : dict
        metadata dictionary.
    op_id : int
        sorting id
    op_name : str
        name of the algorithm or function

    Returns
    -------
    params : dict
        dictionary of function parameters
    called : None|dict
        dictionary of called sub-functions
    """
    try:
        assert (op_id, op_name) in md
    except AssertionError:
        msg = "Unknown metadata entry '({}, {})'."
        raise KeyError(msg.format(op_id, op_name))

    if md[(op_id, op_name)].has_key('params'):
        params = md[(op_id, op_name)]['params']
        called = md[(op_id, op_name)]['called']
    else:
        params = md[(op_id, op_name)]
        called = None

    return params, called


def print_md(metadata, indent_lvl=0):
    """

    Parameters
    ----------
    metadata

    Returns
    -------

    """
    try:
        md = metadata['timagetk']
    except KeyError:
        md = metadata

    s = ""
    sorted_ops = sort_ops(md)
    if sorted_ops is None:
        return s

    op_id = 0
    lid = get_last_id(md)
    for op_name in sorted_ops:
        op_id += 1
        params, called = get_params_called(md, op_id, op_name)
        s += md_str(op_id, lid, op_name, params, called, indent_lvl)
        if called is not None:
            s += print_md(called, indent_lvl + 1)

    s = s[:-1]  # remove the last '\n'
    return s


def md_str(op_id, op_max, op_name, params, called_md, indent_lvl, indent='  '):
    """

    Parameters
    ----------
    op_id
    op_max
    op_name
    params
    called_md
    indent_lvl
    indent

    Returns
    -------

    """
    s = ""
    ind = indent * indent_lvl
    if op_max == 1:
        s += "{}# {}\n".format(ind, op_name)
    else:
        s += "{}#{}. {}\n".format(ind, op_id, op_name)

    s_p = "{}*params:\n{}\n"
    if params == {}:
        s += s_p.format(ind + indent, ind + indent * 2 + 'EMPTY')
    else:
        s += s_p.format(ind + indent, print_params(params, indent_lvl))

    if called_md is not None:
        s += "{}*called:\n".format(ind + indent)

    return s


def print_params(params, indent_lvl, p_indent='--', indent='  '):
    """

    Parameters
    ----------
    params
    indent_lvl

    Returns
    -------
    str
        formated parameters

    Examples
    --------
    >>> import numpy as np
    >>> from timagetk.components import SpatialImage
    >>> from timagetk.plugins.resampling import resample_isotropic
    >>> from timagetk.components.metadata import print_md
    >>> test_array = np.ones((5,5,10), dtype=np.uint8)
    >>> img = SpatialImage(test_array, voxelsize=[1., 1., 2.])
    >>> output_image = resample_isotropic(img, voxelsize=0.4)
    >>> print(print_md(output_image.metadata))

    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.plugins import linear_filtering, h_transform, region_labeling, segmentation
    >>> from timagetk.components.metadata import print_md
    >>> image_path = shared_data('segmentation_src.inr')
    >>> input_image = imread(image_path)
    >>> smooth_image = linear_filtering(input_image, sigma=2.0, method='gaussian_smoothing')
    >>> regext_image = h_transform(smooth_image, h=5, method='min')
    >>> seeds_image = region_labeling(regext_image, low_threshold=1, high_threshold=3, method='connected_components')
    >>> segmented_image = segmentation(smooth_image, seeds_image, control='first', method='seeded_watershed')
    >>> segmented_image.metadata['timagetk']
    >>> print(print_md(segmented_image.metadata))
    """
    s = ""
    ind = indent * indent_lvl + indent
    for k in params.keys():
        val = params.get(k, None)
        if val == {}:
            v = "EMPTY"
        elif val is None:
            v = "None"
        elif k == 'titk_md':
            v = ""
            # - case when a SpatialImage instance has been found and metadata have been returned:
            titk_md = params.get(k, {})
            assert isinstance(titk_md, dict)
            # Get 'class' metadata:
            c = {'class': titk_md.get('class', None)}
            if c['class'] is not None:
                titk_md.pop('class')
                v += "\n" + print_params(c, indent_lvl, '*-')
            # Get SpatialImage metadata:
            if titk_md == {}:
                v += " with no metadata!"
            else:
                v += "\n" + print_md(titk_md, indent_lvl + 2)
        elif isinstance(val, dict):
            v = "\n" + print_params(val, indent_lvl + 1, '*-')
        else:
            v = "{}".format(val)

        s += "{}{}{}: {}\n".format(ind, p_indent, k, v)
    if s.endswith('\n'):
        s = s[:-1]  # remove the last '\n'
    return s


class Metadata(object):
    """Basic metadata class.

    Take a dictionary as input and make object attributes out of it.
    """

    def __init__(self, md=None):
        """Metadata constructor.

        Notes
        -----
        Uni-dimensional numpy arrays are changed to list.
        Other type are unchanged. See example.

        Parameters
        ----------
        md : dict, optional
            metadata dictionary

        Examples
        --------
        >>> import numpy as np
        >>> from timagetk.components.metadata import Metadata
        >>> md = {'origin': [0, 0], 'shape': (10, 10), 'voxelsize':np.array([0.2, 0.2])}
        >>> md = Metadata(md)
        >>> md.origin
        [0, 0]
        >>> md.shape
        (10, 10)
        >>> md.get_dict()
        {'origin': [0, 0], 'shape': (10, 10), 'voxelsize': [0.2, 0.2]}

        """
        # - Declare hidden attribute:
        self._dict = {}

        # - Initialize the object if metadata is given:
        if md is not None:
            self._test_input_md(md)
            self._lazy_loading(md)

    def _lazy_loading(self, md):
        """Set the attributes. """
        for attr_name, attr_val in md.items():
            # - uni-dimensional arrays to list:
            if isinstance(attr_val, np.ndarray) and attr_val.ndim == 1:
                attr_val = attr_val.tolist()
            # - Set the object attribute:
            setattr(self, attr_name, attr_val)
            # - Set the dict key and value
            self._dict[attr_name] = attr_val

    def _test_input_md(self, md):
        """Check the input is a dictionary. """
        try:
            assert isinstance(md, dict)
        except AssertionError:
            raise TypeError("Metadata class input should be a dictionary!")

    def _empty_md(self, default_tags):
        """Initialise empty metadata dictionary.

        Parameters
        ----------
        default_tags : list(str)
            list of tag names

        Returns
        -------
        dict
            the empty metadata dictionary
        """
        n_tags = len(default_tags)
        return dict(zip(default_tags, [None] * n_tags))

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, key, value):
        self.update_attributes({key: value})


    def get_dict(self):
        """Returns the metadata dictionary. """
        return {k: v for k, v in self._dict.items()}

    def update_attributes(self, md):
        """Update the metadata.

        Parameters
        ----------
        md : dict, optional
            metadata dictionary to use for updating metadata attribute

        """
        self._test_input_md(md)
        for k, v in md.items():
            self._dict[k] = v
            setattr(self, k, v)


IMAGE_MD_TAGS = ['shape', 'ndim', 'dtype', 'origin', 'voxelsize']
IMAGE_MD_TYPES = {'shape': 'list', 'ndim': 'int', 'dtype': 'str', 'origin': 'list', 'voxelsize': 'list', 'extent': 'list'}

# NOTE: 'extent' is computed as: 'shape' * 'voxelsize'

class ImageMetadata(Metadata):
    """Metadata associated to ``SpatialImage`` attributes.

    With a ``numpy.array`` get: 'shape', 'ndim', & 'dtype'.
    Also, you must provide 'origin' and 'voxelsize' as keyword arguments,
    otherwise they will be set to their default values:  ``DEFAULT_ORIG_2D``
    & ``DEFAULT_VXS_2D`` or ``DEFAULT_ORIG_3D`` & ``DEFAULT_VXS_3D`` according
    to dimensionality.

    With a ``SpatialImage`` instance or any other which inherit it, also get:
    'voxelsize' & 'origin'.

    Notes:
    ------
    Defined attributes for ``ImageMetadata`` are in ``IMAGE_MD_TAGS``.
    Attribute 'extent' is computed as: 'shape' * 'voxelsize'.
    It is possible to update a metadata dictionary with the known tags (and
    'extent') using ``self.update_metadata()``.

    Attributes:
    -----------
    shape : list(int)
        list of integers defining the shape of the image, ie. its size in voxel
    ndim : int
        number of dimension of the image, should be 2 or 3
    dtype : np.dtype
        bit depth and size of the image
    origin : list(int)
        coordinates of the origin of the image
    voxelsize : list(float)
        size of the voxel in each direction of the image

    """

    def __init__(self, image, **kwargs):
        """ImageMetadata constructor.

        Notes
        -----
        If image is an array, you must provide 'origin' and 'voxelsize' as
        keyword arguments, otherwise they will be set to their default values,
        see: ``self.get_from_array()``.

        Parameters
        ----------
        image : ndarray | SpatialImage
            image to use to define metadata
        image_md : dict
            dictionary of metadata, may contain 'voxelsize' and/or 'origin' if
            using an array instead of a SpatialImage

        Other Parameters
        ----------------
        origin : list, optional
            coordinates of the origin in the image, default: [0, 0] or [0, 0, 0]
        voxelsize : list, optional.
            image voxelsize, default: [1.0, 1.0] or [1.0, 1.0, 1.0]

        Examples
        --------
        >>> import numpy as np
        >>> from timagetk.components.metadata import ImageMetadata
        >>> # - Create a random array:
        >>> img = np.random.random_sample((15, 15))
        >>> # - Do NOT specify 'voxelsize' & 'origin':
        >>> img_md = ImageMetadata(img)
        >>> # - Take a look at the obtained metadata:
        >>> img_md.get_dict()
        {'dtype': dtype('float64'),
         'ndim': 2,
         'origin': [0, 0],
         'shape': (15, 15),
         'voxelsize': [1.0, 1.0]}
        >>> # - Specify 'voxelsize' & 'origin':
        >>> md = {'voxelsize': [0.5, 0.5], 'origin': [0, 0]}
        >>> img_md = ImageMetadata(img, md)
        >>> img_md.get_dict()
        {'dtype': dtype('float64'),
         'ndim': 2,
         'origin': [0, 0],
         'shape': (15, 15),
         'voxelsize': [0.5, 0.5]}
        >>> # - Use a SpatialImage as input:
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> from timagetk.components.metadata import ImageMetadata
        >>> # - Create a random array:
        >>> img = np.random.random_sample((15, 15))
        >>> img = SpatialImage(img)
        >>> # - Do NOT specify 'voxelsize' & 'origin':
        >>> img_md = ImageMetadata(img)

        """
        from ..spatial_image import SpatialImage

        # - Initialize Metadata class:
        Metadata.__init__(self, None)
        # - Declare attributes:
        self.ndim = None  # int
        self.dtype = None  # np.dtype
        self.shape = None  # tuple(int)
        self.origin = None  # list(int)
        self.voxelsize = None  # list(float)
        self.extent = None  # list(float)

        # - Define attribute values from input type:
        if isinstance(image, SpatialImage):
            self.get_from_image(image)
        elif isinstance(image, np.ndarray):
            vxs = kwargs.get('voxelsize', None)
            ori = kwargs.get('origin', None)
            self.get_from_array(image, voxelsize=vxs, origin=ori)
        else:
            msg = "Unknown input `image` type ({})!"
            raise TypeError(msg.format(clean_type(image)))

    def _compute_extent(self):
        """Compute the extent of the image based on 'shape' & 'voxelsize'.

        Returns
        -------
        list
            extent of the image, ie. its size in real units
        """
        from ..spatial_image import compute_extent
        return compute_extent(self.voxelsize, self.shape)

    def _get_default(self, param, default_value_2d, default_value_3d):
        """Set default value for origin or voxelsize.

        Parameters
        ----------
        param : list|None
            list parameter to
        default_value_2d : list
            length-2 array for 2D image case
        default_value_3d : list
            length-3 array for 2D image case

        Returns
        -------
        param
            the correct parameter value
        """
        if param is None or param == []:
            if self.ndim == 2:
                param = default_value_2d
            elif self.ndim == 3:
                param = default_value_3d
            else:
                raise ValueError("Weird dimensionality: {}".format(self.ndim))
        else:
            try:
                assert len(param) == self.ndim
            except AssertionError:
                msg = "Given parameter ({}) is {} than array dimensionality ({})!"
                if len(param) > self.ndim:
                    raise ValueError(msg.format(param, 'bigger', self.ndim))
                else:
                    raise ValueError(msg.format(param, 'smaller', self.ndim))

        return param

    def get_from_array(self, array, voxelsize=None, origin=None):
        """Get image metadata values from an array.

        Notes
        -----
        Only 'shape', 'ndim' & 'type' are accessible from an array!
        Directly update the dictionary and attributes.

        Parameters
        ----------
        array : numpy.array
            array to use to get metadata
        voxelsize : list, optional
            array voxelsize, if ``None`` (default) use the default values
            (either DEFAULT_VXS_2D or DEFAULT_VXS_3D)
        origin : list, optional
            array origin, if ``None`` (default) use the default values
            (either DEFAULT_ORIG_2D or DEFAULT_ORIG_3D)

        Examples
        --------
        >>> import numpy as np
        >>> from timagetk.components.metadata import ImageMetadata
        >>> # - Create a random array:
        >>> img = np.random.random_sample((15, 15))
        >>> # - Do NOT specify 'voxelsize' & 'origin':
        >>> img_md = ImageMetadata(img)
        >>> # - Take a look at the obtained metadata:
        >>> img_md.get_dict()

        """
        from ..spatial_image import DEFAULT_VXS_2D, DEFAULT_VXS_3D, DEFAULT_ORIG_2D, DEFAULT_ORIG_3D

        self.update_attributes({'shape': array.shape})
        self.update_attributes({'ndim': array.ndim})
        self.update_attributes({'dtype': array.dtype})

        defval = self._get_default(voxelsize, DEFAULT_VXS_2D, DEFAULT_VXS_3D)
        self.update_attributes({'voxelsize': defval})

        defval = self._get_default(origin, DEFAULT_ORIG_2D, DEFAULT_ORIG_3D)
        self.update_attributes({'origin': defval})

        # - Update the Metadata to get 'extent' with 'self.get_dict()':
        self.update_attributes({'extent': self._compute_extent()})

        return

    def get_from_image(self, image):
        """Get image metadata values from a SpatialImage instance.

        Notes
        -----
        Directly update the dictionary and attributes.

        Parameters
        ----------
        image : SpatialImage
            image to use to get metadata

        """
        from ..spatial_image import DEFAULT_VXS_2D, DEFAULT_VXS_3D, DEFAULT_ORIG_2D, DEFAULT_ORIG_3D 

        self.update_attributes({'shape': image.shape})
        self.update_attributes({'ndim': image.ndim})
        self.update_attributes({'dtype': image.dtype})

        val = self._get_default(image.voxelsize, DEFAULT_VXS_2D, DEFAULT_VXS_3D)
        self.update_attributes({'voxelsize': list(map(float, val))})

        val = self._get_default(image.origin, DEFAULT_ORIG_2D, DEFAULT_ORIG_3D)
        self.update_attributes({'origin': val})

        # - Update the Metadata to get 'extent' with 'self.get_dict()':
        self.update_attributes({'extent': self._compute_extent()})

        return

    def update_metadata(self, metadata):
        """Update a metadata dictionary with its basics image infos.

        Parameters
        ----------
        metadata : dict
            a metadata dictionary to compare to the object attributes

        Returns
        -------
        metadata : dict
            a verified metadata dictionary
        """
        # --- Update metadata with object value:
        # --------------------------------------
        for attr in IMAGE_MD_TAGS + ['extent']:
            attr_val = getattr(self, attr)
            try:
                if IMAGE_MD_TYPES[attr] == 'list':
                    np.testing.assert_array_almost_equal(metadata[attr],
                                                         attr_val, decimal=6)
                else:
                    assert metadata[attr] == attr_val
            except KeyError:
                # Case where attribute is not defined in metadata: add it
                metadata[attr] = attr_val
            except AssertionError:
                # Case where attribute and metadata value differ: update metadata
                msg = "Metadata '{}' {} do not match the image {} {}."
                logging.warning(
                    msg.format(attr, metadata[attr], attr, attr_val))
                metadata[attr] = attr_val
                logging.warning("--> updated!")

        return metadata

    def similar_metadata(self, md):
        """Compare this ImageMetadata values to another one.

        Parameters
        ----------
        md : dict | ImageMetadata
            dictionary or image metadata to compare to self

        Returns
        -------
        bool
            ``True`` if all metadata values are equal, else ``False``

        """
        if isinstance(md, ImageMetadata):
            md = md.get_dict()

        return all([self._dict[tag] == md[tag] for tag in IMAGE_MD_TAGS])


class ProcessMetadata(object):
    """

    """

    def __init__(self, image):
        """ProcessMetadata constructor.

        Parameters
        ----------
        image : SpatialImage|LabelledImage|TissueImage
            image to use to
        """
        self._dict = {}

        # - Save the image class:
        self.img_class = clean_type(image)

    def get_dict(self):
        """

        Returns
        -------

        """
        pass


    def __str__(self):
        return self._print_md()

    def _print_md(self, indent_lvl=0):
        """Formatter for metadata printing.

        Parameters
        ----------
        indent_lvl : int, optional
            level of indentation to use

        Returns
        -------
        str
        """
        md = self.get_dict()

        s = ""
        sorted_ops = sort_ops(md)
        if sorted_ops is None:
            return s

        op_id = 0
        lid = get_last_id(md)
        for op_name in sorted_ops:
            op_id += 1
            params, called = get_params_called(md, op_id, op_name)
            s += md_str(op_id, lid, op_name, params, called, indent_lvl)
            if called is not None:
                s += print_md(called, indent_lvl + 1)

        s = s[:-1]  # remove the last '\n'
        return s


def _check_fname_def(sp_img, filename, filepath):
    """Hidden function verifying 'filename' & 'filepath' definition in
    SpatialImage metadata.

    Parameters
    ----------
    sp_img : SpatialImage
        image with metadata
    filename : str
        string giving the name of the file (with its extension)
    filepath : str
        string giving the path to the file

    Returns
    -------
    SpatialImage
        image and updated metadata

    """
    # -- Check 'filename' & 'filepath' definition in metadata:
    try:
        # Try to load the 'filename' metadata:
        md_filepath = sp_img.metadata['filepath']
        md_filename = sp_img.metadata['filename']
    except KeyError:
        # If not defined in metadata, use path splitted `fname`:
        sp_img._metadata.update({'filepath': filepath})
        sp_img._metadata.update({'filename': filename})
        # sp_img.metadata['filepath'] = filepath
        # sp_img.metadata['filename'] = filename
    else:
        # If defined, compare given `fname` and metadata:
        (md_filepath, md_filename) = os.path.split(md_filename)

        # -- Check the given filepath and the one found in metadata (if any)
        if md_filepath == '':
            md_filepath = filepath
        elif md_filepath != filepath:
            logging.warning(
                "Metadata 'filepath' differ from the one given to the reader!")
            logging.warning("Updated metadata 'filepath'!")
            md_filepath = filepath
        elif md_filepath == filepath:
            pass
        else:
            print("Got filepath: {}".format(filepath))
            raise ValueError("Undefined 'filepath' condition!")

        # -- Check the given filename and the one found in metadata (if any)
        if md_filename == '':
            md_filename = filename
        elif md_filename != filename:
            logging.warning(
                "Metadata 'filename' differ from the one given to the reader!")
            logging.warning("Updated metadata 'filename'!")
        elif md_filename == filename:
            pass
        else:
            print("Got filename: {}".format(filename))
            raise ValueError("Undefined 'filename' condition!")

        # Update metadata keys:
        sp_img._metadata.update({'filepath': filepath})
        sp_img._metadata.update({'filename': filename})

    sp_img.filepath = sp_img.metadata['filepath']
    sp_img.filename = sp_img.metadata['filename']

    return sp_img


def _check_class_def(sp_img):
    """Check timagetk class definition in metadata.

    Parameters
    ----------
    sp_img : SpatialImage
        image with metadata

    Returns
    -------
    SpatialImage
        image with updated metadata

    """
    from .components import LabelledImage

    # -- Use 'class' definition in 'timagetk' metadata to return correct instance type:
    try:
        md_class = sp_img.metadata['timagetk']['class']
    except KeyError:
        warn_msg = "Initializing from an image without 'class' entry in 'timagetk' metadata!"
        logging.debug(warn_msg)
        sp_img.metadata.update({'timagetk': {'class': 'SpatialImage'}})
        logging.debug("Updated to 'SpatialImage' entry.")
    else:
        if md_class == 'LabelledImage':
            sp_img = LabelledImage(sp_img)
        else:
            sp_img.metadata.update({'timagetk': {'class': 'SpatialImage'}})

    return sp_img


def _check_physical_ppty(metadata):
    """Make sure physical properties are coherent.

    Parameters
    ----------
    metadata : dict
        metadata dictionary, should contain 'shape', 'extent' & 'voxelsize' info

    Raises
    ------
    AssertionError
        if saved and computed 'extent' values are not the same

    """
    from ..spatial_image import compute_extent

    sh = metadata['shape']
    vxs = metadata['voxelsize']
    ext = metadata['extent']

    ext2 = compute_extent(vxs, sh)

    try:
        np.testing.assert_array_equal(ext, ext2)
    except AssertionError:
        msg = "Metadata 'extent' is wrong: should be '{}', but read '{}'!"
        raise ValueError(msg.format(ext2, ext))

    return

