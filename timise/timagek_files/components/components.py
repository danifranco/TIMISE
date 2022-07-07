
from ..spatial_image import SpatialImage
from ..labelled_image import LabelledImage
from ..util import clean_type

IMAGE_TYPES = (SpatialImage, LabelledImage)

def assert_image(obj, obj_name=None):
    """Tests whether given object is an `Image`.

    Parameters
    ----------
    obj : instance
        object to test
    obj_name : str, optional
        if given used as object name for TypeError printing
    """
    if obj_name is None:
        try:
            obj_name = obj.filename
        except AttributeError:
            obj_name = clean_type(obj)

    err = "Input '{}' is not an Image instance."
    try:
        assert isinstance(obj, IMAGE_TYPES)
    except AssertionError:
        raise TypeError(err.format(obj_name))

    return


def image_class(image):
    """Returns the class corresponding to the given image. """
    if isinstance(image, LabelledImage):
        return LabelledImage
    elif isinstance(image, SpatialImage):
        return SpatialImage
    else:
        msg = "Input `image` is not an Image type: {}!"
        raise NotImplementedError(msg.format(type(image)))
