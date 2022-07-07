# -*- python -*-
# -*- coding: utf-8 -*-
#
#       wrapping.clib
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
Module linking external C libraries using ``ctypes``.
"""

import sys
import platform
from ctypes import cdll, c_void_p, c_char_p, POINTER
from ctypes.util import find_library

from .c_bal_trsf import cBalTrsf 

# - Defines list of supported platforms:
SUPP_PLATFORMS = ['Linux', 'Darwin']

# - Load libraries according to platform specificity:
if platform.system() == 'Linux':
    try:
        libbasic = cdll.LoadLibrary('libbasic.so')
        libblockmatching = cdll.LoadLibrary('libblockmatching.so')
        libvt = cdll.LoadLibrary('libvt.so')
        libvp = cdll.LoadLibrary('libvp.so')
        libvtexec = cdll.LoadLibrary('libvtexec.so')
        libio = cdll.LoadLibrary('libio.so')
        libdavid = cdll.LoadLibrary('libdavid.so')
    except ImportError:
        print('Error: unable to load shared libraries')
        sys.exit(-1)
elif platform.system() == 'Darwin':
    try:
        libbasic = cdll.LoadLibrary('libbasic.dylib')
        libblockmatching = cdll.LoadLibrary('libblockmatching.dylib')
        libvt = cdll.LoadLibrary('libvt.dylib')
        libvp = cdll.LoadLibrary('libvp.dylib')
        libvtexec = cdll.LoadLibrary('libvtexec.dylib')
        libio = cdll.LoadLibrary('libio.dylib')
        libdavid = cdll.LoadLibrary('libdavid.dylib')
    except ImportError:
        print('Error: unable to load shared libraries')
        sys.exit(-1)
else:
    print("Supported platforms: {}".format(SUPP_PLATFORMS))
    sys.exit(-1)

# - Get libc:
try:
    libc = cdll.LoadLibrary(find_library('c'))
except ImportError:
    print('Error: unable to find libc')
    sys.exit(-1)

# - Determines standard output according to platform specificity:
if platform.system() == 'Linux':
    c_stdout = c_void_p.in_dll(libc, "stdout")
elif platform.system() == "Darwin":
    c_stdout = c_void_p.in_dll(libc, '__stdoutp')
else:
    print("Supported platforms: {}".format(SUPP_PLATFORMS))
    sys.exit(-1)

libblockmatching.API_blockmatching.restype = POINTER(cBalTrsf)


def return_value(values, rcode):
    """Returns given ``values`` or ``None`` depending on ``rcode`` value.

    Parameters
    ----------
    values : any
        any given instance
    rcode : int
        return code, if equal to '-1', return ``None``, else returns ``values``

    Returns
    -------
    any
        If ``rcode == -1``, return ``None``, else returns ``values``.
    """
    if rcode == -1:
        return None
    else:
        return values


def add_doc(py_func, c_func):
    c_func.restype = c_char_p
    py_doc = py_func.__doc__
    doc = """
%s

parameter_str:

%s
""" % (py_doc, c_func(1))
    py_func.__doc__ = doc
