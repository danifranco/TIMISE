import logging
import pandas as pd
import numpy as np

from .labelled_image import LabelledImage
from .spatial_image import SpatialImage, compute_voxelsize, _to_list, compute_shape
from .trsf import apply_trsf, LABEL_INTERPOLATION_METHODS, GRAY_INTERPOLATION_METHODS, DEF_GRAY_INTERP_METHOD, INTERPOLATION_METHODS, DEF_LABEL_INTERP_METHOD


ISO_RESAMPLE_METHODS = ['min', 'max']
RESAMPLE_OPTIONS = ['gray', 'label']


def _interpolate_params(image, method=None):
    """Check or define correct definition of interpolation method to pass to `apply_trsf`.

    Parameters
    ----------
    image : Image type
        the image to interpolate
    method : str in INTERPOLATION_METHODS, optional
        If None (default) try to guess the interpolation method from Image type,
        else check if its an adequate or defined method.

    Returns
    -------
    str
        The string to use with `param_str_1` or `param_str_2` in `apply_trsf`.

    See Also
    --------
    apply_trsf : the wrapped interpolation algorithm
    INTERPOLATION_METHODS : the list of available interpolation methods

    """
    method_msg = "WARNING: Input `image` is a {}, option should be in {} but got '{}'!"

    if isinstance(image, LabelledImage):
        if method is None:
            method = 'label'
        else:
            allowed_methods = ['label'] + LABEL_INTERPOLATION_METHODS
            try:
                assert method in allowed_methods
            except AssertionError:
                print(
                    method_msg.format("LabelledImage", allowed_methods, method))
                method = 'label'
                print("Changed to '{}'!".format(method))
    elif isinstance(image, SpatialImage):
        if method is None:
            method = 'gray'
        else:
            allowed_methods = ['gray', 'grey'] + GRAY_INTERPOLATION_METHODS
            try:
                assert method in allowed_methods
            except AssertionError:
                print(method_msg.format("SpatialImage", allowed_methods, method))
                method = 'gray'
                print("Changed to '{}'!".format(method))
    else:
        msg = "Input `image` is not an Image type: {}!"
        raise NotImplementedError(msg.format(type(image)))

    param_str = ""
    if method == 'gray' or method == 'grey':
        param_str += ' -interpolation {}'.format(DEF_GRAY_INTERP_METHOD)
    elif method == 'label':
        param_str += ' -interpolation {}'.format(DEF_LABEL_INTERP_METHOD)
    elif method in INTERPOLATION_METHODS:
        param_str += ' -interpolation {}'.format(method)
    else:
        msg = "Given interpolation 'method' ({}) is not available!\n".format(method)
        msg += "Choose among: {}".format(RESAMPLE_OPTIONS+INTERPOLATION_METHODS)
        raise NotImplementedError(msg)

    return param_str


def resample(image, voxelsize=None, shape=None, option=None, **kwargs):
    """Resample an image to the given voxelsize.

    Parameters
    ----------
    image : SpatialImage
        The image to resample.
    voxelsize : list, optional
        The voxelsize to which the image should be resampled to.
    shape : list, optional
        The shape to which the image should be resampled to.
    option : str in {'gray', 'label'}, optional
        Use 'gray' with a grayscale image or 'label' with a labelled image.
        By default (None) try to guess it from the type of `image`.

    Other Parameters
    ----------------
    verbose : bool
        if ``True``, increase the level of verbosity

    Returns
    -------
    out_img : SpatialImage
        the resampled image

    See Also
    --------
    GRAY_INTERPOLATION_METHODS : the list of available interpolation methods for grayscale images.
    LABEL_INTERPOLATION_METHODS : the list of available interpolation methods for labelled images.
    DEF_GRAY_INTERP_METHOD : the default interpolation methods for grayscale images.
    DEF_LABEL_INTERP_METHOD : the default interpolation methods for labelled images.

    Notes
    -----
    Use 'option' to control type of interpolation applied to the image.
    Note that you have to provide either a target ``voxelsize`` or ``shape``.
    Using 'linear' interpolation method might decrease the contrast.

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.components import SpatialImage
    >>> from timagetk.plugins.resampling import resample
    >>> test_array = np.ones((5,5,10), dtype=np.uint8)
    >>> img = SpatialImage(test_array, voxelsize=[1., 1., 2.])
    >>> print(img)
    >>> # Performs resampling on SpatialImage:
    >>> out_img = resample(img, voxelsize=[0.4, 0.3, 0.2], verbose=True)
    >>> # Performs resampling on LabelledImage:
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.components import LabelledImage
    >>> img = LabelledImage(imread(shared_data("time_3_seg.inr")), no_label_id=0)
    >>> out_img = resample(img, voxelsize=[vxs * 2. for vxs in img.voxelsize], verbose=True)

    """
    verbose = kwargs.get('verbose', False)
    ndim = image.ndim

    try:
        assert voxelsize != [] or shape != [] and not (
                voxelsize is not None and shape is not None)
    except AssertionError:
        msg = "You have to provide either `voxelsize` or `shape` as input!"
        raise ValueError(msg)

    if shape is not None and shape != []:
        voxelsize = compute_voxelsize(image.extent, shape)

    try:
        assert image.voxelsize != []
    except AssertionError:
        raise ValueError("Input image has an EMPTY voxelsize attribute!")
    try:
        assert len(voxelsize) == ndim
    except AssertionError:
        msg = "Given 'voxelsize' ({}) ".format(voxelsize)
        msg += "does not match the dimension of the image ({}).".format(ndim)
        raise ValueError(msg)

    # - Make sure voxelsize is a list:
    voxelsize = _to_list(voxelsize)
    # - Compute the new shape of the object using image extent & new voxelsize:
    extent = np.array(image.extent)
    new_shape = compute_shape(voxelsize, extent)
    # - Initialise a new metadata dictionary matching the template array:
    new_md = image._metadata
    # -- UPDATE the keys and values of 'NEW' properties: 'shape' & 'voxelsize'
    new_md['shape'] = new_shape
    new_md['voxelsize'] = voxelsize

    if np.allclose(image.voxelsize, voxelsize, atol=1e-06):
        print("Image already has correct voxelsize!")
        return image

    # - Initialise a SpatialImage from a template array, new voxelsize and metadata dictionary:
    tmp_img = SpatialImage(np.zeros(new_shape, dtype=image.dtype),
                           voxelsize=voxelsize, origin=image.origin,
                           metadata=new_md)
    if verbose:
        print("Template initialized: {}".format(tmp_img.metadata_image.get_dict()))
        print("Image resampling will change:")
        print("  - 'shape': {} -> {}".format(image.shape, tmp_img.shape))
        print("  - 'voxelsize': {} -> {}".format(image.voxelsize, tmp_img.voxelsize))

    param_str_2 = ' -resize'
    param_str_2 += _interpolate_params(image, method=option)

    # - Performs resampling using 'apply_trsf':
    out_img = apply_trsf(image, trsf=None, template_img=tmp_img,
                         param_str_2=param_str_2)

    # - Since 'apply_trsf' only works on 3D images, it might have converted it to 3D:
    if 1 in out_img.shape:
        out_img = out_img.to_2D()

    if verbose:
        msg = "Resampled image is a {}"
        print(msg.format(out_img))

    # add2md(out_img)
    return out_img



def fast_image_overlap3d(mother_seg, daughter_seg, method='target_mother', ds=1,
                         mother_label=None, daughter_label=None, verbose=True,
                         decimal=5):
    """ Compute all the possible overlap between cells from two images. Different overlap estimation are available
        This method is way faster than image_overlap().

        Available criterion are:

      - **Jaccard coefficient**: J(A, B) = |A n B|/|A u B| [1]
      - **Target overlap (mother/daughter)**: T(A, B) = |A n B|/|B| OR T(B, A) = |A n B|/|A|

    Parameters
    ----------
    mother_seg : LabelledImage
        Image labelled, voxelsize and size should be the same than daughter_seg
    daughter_seg: LabelledImage
        Image labelled, voxelsize and size should be the same than mother_seg
    method : str, optional
        estimation method used for the overlap: {‘target_mother','target_daughter','jaccard'}
        'target_mother': the overlap is estimated as: volume(mother_cell).intersection(volume(daughter_cell)) / voluem(mother_cell)
        'target_daughter': the overlap is estimated as: volume(mother_cell).intersection(volume(daughter_cell)) / voluem(daughter_cell)
        'jaccard': the overlap is estimated as: volume(mother_cell).intersection(volume(daughter_cell)) / volume(mother_cell).union(volume(daughter_cell))
        'intersection' : the overlap intersection
    ds : int, optional
        downsampling factor
    mother_label : list, optional
        list of mother labels where the overlap need to be computed. default: None (all the mother labels)
    daughter_label : list, optional
        list of daughter labels where the overlap need to be computed. default: None (all the daughter labels)
    verbose : bool, optional
        display the couples found

    Returns
    -------
    ov_df : dataframe

    Example
    -------
    >>> from timagetk.components.labelled_image import LabelledImage
    >>> from ctrl.algorithm.image_overlap import fast_image_overlap3d
    >>> import numpy as np
    >>> I = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 3, 3, 3, 3, 1, 1, 1], [1, 1, 1, 3, 3, 3, 3, 1, 1, 1], [1, 1, 1, 3, 3, 3, 3, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    >>> J = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    >>> I = LabelledImage(I, not_a_label = 0)
    >>> J = LabelledImage(J, not_a_label = 0)
    >>> ov_df = fast_image_overlap3d(I, J, method = 'target_daughter')
    >>> print(ov_df)

    """
    # - Check the input
    try:
        assert isinstance(mother_seg, LabelledImage) and isinstance(daughter_seg, LabelledImage)
    except:
        raise ValueError

    try:
        assert method in {'target_mother', 'target_daughter', 'jaccard', 'intersection'}
    except:
        raise ValueError

    # - Downsampling segmented images in order to accelerate computing
    if ds != 1:
        new_voxelsize = [vox * ds for vox in daughter_seg.voxelsize]
        mother_seg = resample(mother_seg, voxelsize=new_voxelsize, interpolation='cellbased', cell_based_sigma=1)
        daughter_seg = resample(daughter_seg, voxelsize=new_voxelsize, interpolation='cellbased', cell_based_sigma=1)

        mother_seg = LabelledImage(mother_seg, not_a_label=0)
        daughter_seg = LabelledImage(daughter_seg, not_a_label=0)

    if mother_label is None:
        mother_label = mother_seg.labels()

    if daughter_label is None:
        daughter_label = daughter_seg.labels()

    # - Get the bounding boxes
    mother_bboxes = mother_seg.boundingbox(labels=mother_label)
    daughter_bboxes = daughter_seg.boundingbox(labels=daughter_label)

    # - Precompute the volume of the cells: do not recalculate at each iteration later
    #   Volume are estimated in voxel units
    mother_volume_cell = {}
    for lab, mbbox in mother_bboxes.items():
        mother_volume_cell[lab] = np.sum(mother_seg[mbbox].get_array() == lab)

    daughter_volume_cell = {}
    for lab, dbbox in daughter_bboxes.items():
        daughter_volume_cell[lab] = np.sum(daughter_seg[dbbox].get_array() == lab)

    ov_df = []
    # - Loop over the mother cells to compute all the possible overlap with the daughter cells
    for mlab, mbbox in mother_bboxes.items():
        for dlab, dbbox in daughter_bboxes.items():
            # - check if overlap between both bboxes (common borders)
            coord_min = [max(mb.start, db.start) for mb, db in zip(mbbox, dbbox)]
            coord_max = [min(mb.stop, db.stop) for mb, db in zip(mbbox, dbbox)]

            if all([dim_min < dim_max for dim_min, dim_max in zip(coord_min, coord_max)]):
                # - get the coordinates that corresponds to the intersection of both volumes: volume(mother_cell).intersection(volume(daughter_cell))
                bbox_intersection = tuple([slice(dim_min, dim_max, None) for dim_min, dim_max in zip(coord_min, coord_max)])

                # - get the subimage corresponding to the intersection
                mother_sub = np.array(mother_seg[bbox_intersection] == mlab)
                daughter_sub = np.array(daughter_seg[bbox_intersection] == dlab)

                # - calculate the intersection between both cells (voxel units)
                intersection = (mother_sub & daughter_sub).sum()

                # - according to the method, calculate the overlap coefficient
                if method == 'target_mother':
                    val = intersection / mother_volume_cell[mlab]
                elif method == 'target_daughter':
                    val = intersection / daughter_volume_cell[dlab]
                elif method == 'jaccard':  # jaccard: in this case the union is estimated from the respective cell volume and their intersection
                    val = intersection / (mother_volume_cell[mlab] + daughter_volume_cell[dlab] - intersection)
                elif method == 'intersection':
                    val = intersection
                else:
                    logging.error('Unknown method!')

                val = np.around(val, decimals=decimal)

                if verbose:
                    logging.info("--> Couple (" + str(mlab) + ", " + str(dlab) + ") : " + str(val))

                ov_df.append({'mother_label': mlab, 'daughter_label': dlab, method: val})

    return pd.DataFrame(ov_df)