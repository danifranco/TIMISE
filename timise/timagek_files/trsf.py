from .wrapping.bal_image import BalImage
from .wrapping.clib import libblockmatching


__all__ = [
    'DEF_INV_TRSF', 'inv_trsf',
    'DEF_APPLY_TRSF', 'apply_trsf',
    'DEF_COMPOSE_TRSF', 'compose_trsf',
    'DEF_CREATE_TRSF', 'create_trsf',
    'mean_trsfs'
]

# Methods 'linear' & 'cspline' are for grayscale images:
GRAY_INTERPOLATION_METHODS = ['linear', 'cspline']
# Methods 'nearest' is for grayscale images:
LABEL_INTERPOLATION_METHODS = ['nearest']

INTERPOLATION_METHODS = GRAY_INTERPOLATION_METHODS + LABEL_INTERPOLATION_METHODS
# Default interpolation methods per type of images:
DEF_GRAY_INTERP_METHOD = 'cspline'  # 'linear' decrease contrast and blur image!
DEF_LABEL_INTERP_METHOD = 'nearest'

DEF_INV_TRSF = ''
DEF_APPLY_TRSF = '-' + DEF_GRAY_INTERP_METHOD
DEF_COMPOSE_TRSF = ''
DEF_CREATE_TRSF = '-identity'

import logging
from ctypes import POINTER
from ctypes import pointer

import numpy as np

from .components.components import assert_image, image_class
# from .util import check_type
from .spatial_image import SpatialImage

def apply_trsf(image, trsf=None, template_img=None,
               param_str_1=DEF_APPLY_TRSF, param_str_2=None, dtype=None):
    """Apply a ``BalTrsf`` transformation to a ``SpatialImage`` image.

    To apply a transformation to a ``LabelledImage``, uses '-nearest' in
    ``param_str_2``, default is ``DEF_APPLY_TRSF``.

    Parameters
    ----------
    image : Image
        input image to transform.
    trsf : BalTrsf, optional
        input transformation, default is identity.
    template_img : Image or list or dict, optional
        Default is ``None``, used for output image geometry, can be a ``SpatialImage``,
        a list or tuple of dimensions or a dictionary.
        If a list or tuple of integers, its length should match ``image.ndim``,
        voxelsize will be defined by ``image``.
        If a dictionary, should match the fields from `spatial_image_to_bal_image_fields()`.
        If ``None``, attributes 'shape', 'voxelsize' & 'np_type' will be obtained from ``image``.
    param_str_1 : str, optional
        see ``DEF_APPLY_TRSF``
    param_str_2 : str, optional.
        optional parameters.
    dtype : np.dtype, optional
        output image type, by default output type is equal to input type.

    Returns
    -------
    SpatialImage
        output image with metadata.

    See Also
    --------
    wrapping.bal_image.spatial_image_to_bal_image_fields: the function extracting
    the required keyword arguments to defines a template from an image.

    Example
    -------
    >>> from timagetk.util import shared_data
    >>> from timagetk.io import imread
    >>> from timagetk.algorithms.trsf import apply_trsf
    >>> from timagetk.algorithms.trsf import create_trsf
    >>> image = imread(shared_data('p58-t0-a0.lsm'))
    >>> trsf = create_trsf(template_img=image, param_str_2='-random')
    >>> print("Random {} transformation:".format(trsf.get_type()))
    >>> print(trsf.mat.to_np_array())
    >>> output_image = apply_trsf(image, trsf)

    >>> # Display results:
    >>> from timagetk.visu.mplt import grayscale_imshow
    >>> grayscale_imshow([image, output_image], title=['Original', 'Transformed'], suptitle="Arbitrary transformation")

    >>> # - Effect of resampling method: 'linear' vs. 'cspline'
    >>> from timagetk.plugins.resampling import isometric_template
    >>> iso_tmp = isometric_template(image)
    >>> iso_img_linear = apply_trsf(image, template_img=iso_tmp, param_str_2='-linear')
    >>> iso_img_cspline = apply_trsf(image, template_img=iso_tmp, param_str_2='-cspline')

    >>> from timagetk.visu.mplt import orthogonal_view
    >>> xs, ys, zs = iso_tmp.get_shape('x')/2, iso_tmp.get_shape('y')/2, iso_tmp.get_shape('z')/2
    >>> orthogonal_view(iso_img_linear, xs, ys, zs, title="Isometric resampling - linear", figname=shared_data("p58-t0-iso_linear.png"))
    >>> orthogonal_view(iso_img_cspline, xs, ys, zs, title="Isometric resampling - cspline", figname=shared_data("p58-t0-iso_cspline.png"))

    >>> orthogonal_view(iso_img_linear, xs, title="Isometric resampling - linear", figname=shared_data("p58-t0-x_slice-iso_linear.png"), cmap='viridis')
    >>> orthogonal_view(iso_img_cspline, xs, title="Isometric resampling - cspline", figname=shared_data("p58-t0-x_slice-iso_cspline.png"), cmap='viridis')

    >>> from timagetk.components import LabelledImage
    >>> image = LabelledImage(imread(shared_data("time_3_seg.inr")), no_label_id=0)
    >>> trsf = create_trsf(template_img=image, param_str_2='-random')
    >>> print("Random {} transformation:".format(trsf.get_type()))
    >>> print(trsf.mat.to_np_array())
    >>> out_image = apply_trsf(image, trsf, param_str_2=' -interpolation nearest')
    >>> print(type(image))
    >>> print(type(out_image))

    """
    assert_image(image, obj_name='image')
    Image = image_class(image)
    no_label_id=getattr(image, 'no_label_id', None)
    logging.debug('Attribute `no_label_id` from input `image`: {}'.format(no_label_id))
    background=getattr(image, 'background', None)
    logging.debug('Attribute `background` from input `image`: {}'.format(background))

    # - API_applyTrsf works only on 3D images:
    if image.is2D():  # 2D management
        image = image.to_3D()

    # - If a transformation is given, make sure it's a ``BalTrsf``:
    # if trsf is not None:
    #     check_type(trsf, 'trsf', BalTrsf)

    # - Initialize the OUTPUT BalImage to use with API_applyTrsf:
    if isinstance(template_img, SpatialImage):
        # -- Initialize BalImage directly:
        if template_img.is2D():
            # - API_applyTrsf works only on 3D images:
            template_img = template_img.to_3D()
        # - If a dtype is specified, we convert the `template_img` before creation of BalImage:
        if dtype:
            bal_out_image = BalImage(spatial_image=template_img.copy().astype(dtype))
        else:
            bal_out_image = BalImage(spatial_image=template_img.copy())
    # else:
    #     # -- Defines "bal_fields" to create cBalImage object:
    #     bal_fields = {}
    #     # MUST contains same fields than defined in `spatial_image_to_bal_image_fields()`
    #     if template_img is None:
    #         # - If no template image is given, create "bal_fields" from the SpatialImage:
    #         bal_fields = spatial_image_to_bal_image_fields(image)
    #         # - If a dtype is specified, we override the BalImage field 'np_type':
    #         if dtype:
    #             bal_fields['np_type'] = dtype
    #     elif isinstance(template_img, list) or isinstance(template_img, tuple):
    #         # - If a shape is given as template, use it as "bal_fields":
    #         # -- First, make sure image dimensionality and template shape are compatible:
    #         try:
    #             assert len(template_img) == image.ndim
    #         except AssertionError:
    #             msg = "Template shape {} and image dimensionality {} do not match!"
    #             raise ValueError(msg.format(template_img, image.ndim))
    #         # -- Then, create "bal_fields" from the SpatialImage and override the shape:
    #         bal_fields = spatial_image_to_bal_image_fields(image)
    #         bal_fields['shape'] = template_img + [1]  # 'vdim' is also here
    #     elif isinstance(template_img, dict):
    #         # If a dictionary, should match fields returned by `spatial_image_to_bal_image_fields`
    #         # -- Check both 'shape' & 'voxelsize' are defined:
    #         assert all([k in template_img for k in ['shape', 'voxelsize']])
    #         # -- Check 'vt_type' OR 'np_type' is defined:
    #         assert sum([k in template_img for k in ['vt_type', 'np_type']]) == 1
    #         # -- Then, make sure image dimensionality and template shape are compatible:
    #         if (len(template_img['shape']) == image.ndim + 1) and template_img['shape'][-1] != 1:
    #             template_img['shape'] = template_img['shape'] + [1]  # add 'vdim'
    #         try:
    #             assert len(template_img['shape']) == image.ndim + 1  # 'vdim'
    #         except AssertionError:
    #             msg = "Template shape {} and image dimensionality {} do not match!"
    #             raise ValueError(msg.format(template_img['shape'], image.ndim))
    #         # -- Then, create "bal_fields" from the dictionary:
    #         bal_fields['shape'] = list(template_img['shape'])
    #         bal_fields['voxelsize'] = template_img['voxelsize']
    #         if 'vt_type' in template_img:
    #             bal_fields['vt_type'] = template_img['vt_type']
    #         else:
    #             bal_fields['np_type'] = template_img['np_type']
    #     else:
    #         msg = INPUT_TYPE_ERROR.format(template_img, 'template_img',
    #                                       SpatialImage)
    #         raise TypeError(msg)
    #     # -- If a dtype is specified, we override the BalImage field 'np_type':
    #     if dtype:
    #         bal_fields['np_type'] = dtype
    #         # Try to remove any defined 'vt_type' if `dtype` is given:
    #         try:
    #             bal_fields.pop('vt_type')
    #         except:
    #             pass
    #     # -- Use "bal_fields" to initialize a cBalImage:
    #     c_out_image = cBalImage()
    #     template_img = np.ndarray(bal_fields['shape'],
    #                               dtype=bal_fields['np_type'])
    #     init_c_bal_image(c_out_image, bal_fields.pop('shape'), **bal_fields)
    #     allocate_c_bal_image(c_out_image, template_img)
    #     # -- Initialize output BalImage:
    #     bal_out_image = BalImage(c_bal_image=c_out_image)


    # - Initialise the INPUT BalImage:
    bal_image = BalImage(spatial_image=image)

    # - Call API_applyTrsf from library libblockmatching:
    if param_str_1:
        param_str_1 = param_str_1.encode('utf-8')
    if param_str_2:
        param_str_2 = param_str_2.encode('utf-8')

    libblockmatching.API_applyTrsf(bal_image.c_ptr, bal_out_image.c_ptr,
                                   trsf.c_ptr if trsf else None,
                                   param_str_1, param_str_2)

    # - Get output SpatialImage:
    out_image = bal_out_image.to_spatial_image()
    out_image = Image(out_image,
                      no_label_id=no_label_id,
                      background=background)

    # - Convert back to 2D image if input was converted to 3D
    if 1 in out_image.shape:
        out_image = out_image.to_2D()

    # - Free memory allocated to input and output BalImages:
    bal_image.free()
    bal_out_image.free()

    # - Add processing metadata:
    # add2md(out_image)
    return out_image
