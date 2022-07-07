
import re
import time

INPUT_TYPE_ERROR = "Input '{}' should be a ``{}`` instance, got: {}"


def clean_type(obj):
    """Get a clean string of ``obj`` type."""
    s = str(type(obj))
    if s.startswith("<type"):
        regexp = "<type \'(.+)\'>"
        m = re.search(regexp, s)
        try:
            ct = m.group(1)
        except AttributeError:
            msg = "Could not find regex '{}' in '{}'!"
            raise ValueError(msg.format(regexp, s))
    else:
        regexp = "<class \'(.+)\'>"
        m = re.search(regexp, s)
        try:
            ct = m.group(1).split(".")[-1]
        except AttributeError:
            msg = "Could not find regex '{}' in '{}'!"
            raise ValueError(msg.format(regexp, s))
    return ct


def check_type(obj, obj_name, obj_type):
    """Check ``obj`` is the right instance against ``obj_type``.

    Parameters
    ----------
    obj : any
        an instance to test against ``obj_type``
    obj_name : str
        name of the obj, use for printing
    obj_type : any|list(any)
        type of instance that ``obj`` should match, can also be a list of types

    Raises
    ------
    TypeError
        standard error message for input type
    """
    if isinstance(obj_type, list) and len(obj_type) >= 2:
        try:
            assert True in [isinstance(obj, ot) for ot in obj_type]
        except AssertionError:
            raise TypeError(
                INPUT_TYPE_ERROR.format(obj_name, obj_type, type(obj)))
    else:
        try:
            assert isinstance(obj, obj_type)
        except AssertionError:
            raise TypeError(
                INPUT_TYPE_ERROR.format(obj_name, obj_type, type(obj)))



def get_attributes(obj, attr_list):
    """
    Return a dictionary with attributes values from 'obj'.
    By default they are set to ``None`` if not defined.

    Parameters
    ----------
    obj : any
        an object from which to try to get attributes
    attr_list : list(str)
        list of attributes to get from the object

    Returns
    -------
    dict
        attr_list as keys and the attribute value as their values.
    """
    return {attr: getattr(obj, attr, None) for attr in attr_list}



def get_class_name(obj):
    """
    Returns a string defining the class name.
    No module & package hierarchy returned.

    Parameters
    ----------
    obj : any
        any object for which you want to get the class name

    Returns
    -------
    str
        the name of the class
    """
    return str(type(obj))[:-2].split('.')[-1]



def elapsed_time(start, stop=None, round_to=3):
    """Return a rounded elapsed time float.

    Parameters
    ----------
    start : float
        start time
    stop : float, optional
        stop time, if ``None``, get it now
    round : int, optional
        number of decimals to returns

    Returns
    -------
    float
        rounded elapsed time
    """
    if stop is None:
        stop = time.time()
    t = round(stop - start, round_to)

    return "done in {}s".format(t)

