
import time
import logging
import numpy as np
import scipy.ndimage as nd
from scipy.cluster.vq import vq

# from .util import clean_type
# from .util import stuple
from .util import elapsed_time
from .util import get_attributes
from .util import get_class_name
# from .util import percent_progress

# from timagetk.components.slices import dilation_by
# from timagetk.components.slices import real_indices

from .spatial_image import SpatialImage


def topological_elements_extraction2D(img, elem_order=None):
    """
    Extract the topological elements of order 2 (ie. wall) and 1 (ie. wall-edge)
    by returning their coordinates grouped by pair or triplet of labels.

    Parameters
    ----------
    img : numpy.array
        numpy array representing a labelled image
    elem_order : list, optional
        list of dimensional order of the elements to return, should be in [2, 1]

    Returns
    -------
    topological_elements : dict
        dictionary with topological elements order as key, each containing
        dictionaries of n-uplets as keys and coordinates array as values

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.components.labelled_image import topological_elements_extraction2D
    >>> im = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 2, 2, 3, 3, 3, 3, 1],
                       [1, 2, 2, 2, 3, 3, 3, 1],
                       [1, 2, 2, 2, 3, 3, 3, 1],
                       [1, 2, 2, 2, 3, 3, 3, 1],
                       [1, 4, 4, 4, 4, 4, 4, 1],
                       [1, 4, 4, 4, 4, 4, 4, 1],
                       [1, 4, 4, 4, 4, 4, 4, 1]])
    >>> im.shape
    (8, 8)
    >>> # Extract topological elements coordinates:
    >>> elem = topological_elements_extraction2D(im)
    >>> # Get the wall-edge voxel coordinates between labels 1, 2 and 3:
    >>> elem[1][(2, 3, 4)]
    array([[ 0.5,  2.5,  0.5],
           [ 1.5,  2.5,  0.5],
           [ 1.5,  3.5,  0.5],
           [ 2.5,  3.5,  0.5],
           [ 3.5,  3.5,  0.5]])
    >>> # Get the wall voxel coordinates between labels 2 and 3:
    >>> elem[2][(2, 43)]
    array([[ 5.5,  0.5,  0.5],
           [ 5.5,  1.5,  0.5],
           [ 5.5,  2.5,  0.5],
           [ 5.5,  3.5,  0.5],
           [ 5.5,  4.5,  0.5],
           [ 5.5,  5.5,  0.5],
           [ 5.5,  6.5,  0.5],
           [ 6.5,  0.5,  0.5],
           [ 6.5,  1.5,  0.5],
           [ 6.5,  2.5,  0.5],
           [ 6.5,  3.5,  0.5],
           [ 6.5,  4.5,  0.5],
           [ 6.5,  5.5,  0.5],
           [ 6.5,  6.5,  0.5]])
    """
    n_nei_vox = 4
    print("# - Detecting cell topological elements:")
    sh = np.array(img.shape)
    n_voxels = (sh[0] - 1) * (sh[1] - 1)
    print(
        "  - Computing the neighborhood matrix of non-marginal voxels (n={})...".format(
            n_voxels))
    start_time = time.time()
    # - Create the neighborhood matrix of each voxels:
    neighborhood_img = []
    for x in np.arange(-1, 1):
        for y in np.arange(-1, 1):
            neighborhood_img.append(img[1 + x:sh[0] + x, 1 + y:sh[1] + y])

    # - Reshape the neighborhood matrix in a N_voxel x n_nei_vox:
    neighborhood_img = np.sort(
        np.transpose(neighborhood_img, (1, 2, 0))).reshape((sh - 1).prod(),
                                                           n_nei_vox)
    elapsed_time(start_time)

    print("  - Filtering out voxels surrounded only by similar label...")
    # - Detect the voxels that are not alone (only neighbors to themself, or same label around):
    non_flat = np.sum(
        neighborhood_img == np.tile(neighborhood_img[:, :1], (1, n_nei_vox)),
        axis=1) != n_nei_vox
    # - Keep only these "non flat" neighborhood:
    neighborhood_img = neighborhood_img[non_flat]
    elapsed_time(start_time)

    n_voxels_elem = neighborhood_img.shape[0]
    pc = float(n_voxels - n_voxels_elem) / n_voxels * 100
    print("\tRemoved {}% of the initial voxel matrix!".format(round(pc, 3)))

    print("  - Creating the associated coordinate matrix...")
    start_time = time.time()
    # - Create the coordinate matrix associated to each voxels of the neighborhood matrix:
    vertex_coords = np.transpose(np.mgrid[0:sh[0] - 1, 0:sh[1] - 1],
                                 (1, 2, 0)).reshape((sh - 1).prod(), 2) + 0.5
    # - Keep only these "non flat" coordinates:
    vertex_coords = vertex_coords[non_flat]
    elapsed_time(start_time)

    print("  - Computing the neighborhood size...")
    start_time = time.time()
    # - Keep the "unique values" in each neighborhood:
    neighborhoods = list(map(np.unique, neighborhood_img))
    neighborhoods = np.array(neighborhoods)
    # - Compute the neighborhood size of each voxel:
    neighborhood_size = list(map(len, neighborhoods))
    neighborhood_size = np.array(neighborhood_size)
    elapsed_time(start_time)

    print(
        "  - Creating dictionary of voxels coordinates detected as topological elements (n={})...".format(
            n_voxels_elem))
    if elem_order is None:
        elem_order = [2, 1]
    # - Separate the voxels depending on the size of their neighborhood:
    #   "wall" is a dimension 2 element with a neighborhood size == 2;
    #   "wall-edge" is a dimension 1 element with a neighborhood size == 3;
    topological_elements = {}
    start_time = time.time()
    for dimension in elem_order:
        try:
            assert dimension in [2, 1]
        except AssertionError:
            print("Given element orders should be in [1, 2]!")
            continue
        # - Make a mask indexing desired 'neighborhood_size':
        dim_mask = neighborhood_size == 4 - dimension
        msg = "  --Sorting {} voxels as topological elements of order {} ({})..."
        if dimension == 2:
            elem_type = "faces"
        elif dimension == 1:
            elem_type = "edges"
        else:
            elem_type = "nodes"
        print(msg.format(sum(dim_mask), dimension, elem_type))
        # - Get all coordinates corresponding to selected 'neighborhood_size':
        element_points = vertex_coords[dim_mask]
        # - Get labels list for this given 'neighborhood_size':
        element_cells = np.array([p for p in neighborhoods[dim_mask]], int)

        if element_cells != np.array([]):
            # - Remove duplicate of labels n-uplets, with 'n = 4 - dim':
            unique_cell_elements = array_unique(element_cells)
            # - ??
            element_matching = vq(element_cells, unique_cell_elements)[0]
            # - Make a dictionary of all {(n-uplet) : np.array(coordinates)}:
            topological_elements[dimension] = dict(
                zip([tuple(e) for e in unique_cell_elements],
                    [element_points[element_matching == e] for e, _ in
                     enumerate(unique_cell_elements)]))
        else:
            print(
                "WARNING: Could not find topological elements of order {}!".format(
                    dimension))
            topological_elements[dimension] = None

    elapsed_time(start_time)
    return topological_elements


def topological_elements_extraction3D(img, elem_order=None):
    """
    Extract the topological elements of order 2 (ie. wall), 1 (ie. wall-edge)
    and 0 (ie. cell vertex) by returning their coordinates grouped by pair,
    triplet and quadruplet of labels.

    Parameters
    ----------
    img : numpy.array
        numpy array representing a labelled image
    elem_order : list, optional
        list of dimensional order of the elements to return, should be in [2, 1, 0]

    Returns
    -------
    topological_elements : dict
        dictionary with topological elements order as key, each containing
        dictionaries of n-uplets as keys and coordinates array as values

    Example
    -------
    >>> import numpy as np
    >>> from timagetk.components.labelled_image import topological_elements_extraction3D
    >>> a = np.array([[2, 2, 2, 3, 3, 3, 3, 3],
                      [2, 2, 2, 3, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [2, 2, 2, 2, 3, 3, 3, 3],
                      [4, 4, 4, 4, 4, 4, 4, 4],
                      [4, 4, 4, 4, 4, 4, 4, 4],
                      [4, 4, 4, 4, 4, 4, 4, 4]])
    >>> bkgd_im = np.ones_like(a)
    >>> # Create an image by adding a background and repeat the previous array 6 times as a Z-axis:
    >>> im = np.array([bkgd_im, a, a, a, a, a, a]).transpose(1, 2, 0)
    >>> im.shape
    (8, 8, 7)
    >>> # Extract topological elements coordinates:
    >>> elem = topological_elements_extraction3D(im)
    >>> # Get the cell-vertex coordinates between labels 1, 2, 3 and 4
    >>> elem[0]
    {(1, 2, 3, 4): array([[ 4.5,  3.5,  0.5]])}
    >>> # Get the wall-edge voxel coordinates between labels 1, 2 and 3:
    >>> elem[1][(1, 2, 3)]
    array([[ 0.5,  2.5,  0.5],
           [ 1.5,  2.5,  0.5],
           [ 1.5,  3.5,  0.5],
           [ 2.5,  3.5,  0.5],
           [ 3.5,  3.5,  0.5]])
    >>> # Get the wall voxel coordinates between labels 1 and 4:
    >>> elem[2][(1, 4)]
    array([[ 5.5,  0.5,  0.5],
           [ 5.5,  1.5,  0.5],
           [ 5.5,  2.5,  0.5],
           [ 5.5,  3.5,  0.5],
           [ 5.5,  4.5,  0.5],
           [ 5.5,  5.5,  0.5],
           [ 5.5,  6.5,  0.5],
           [ 6.5,  0.5,  0.5],
           [ 6.5,  1.5,  0.5],
           [ 6.5,  2.5,  0.5],
           [ 6.5,  3.5,  0.5],
           [ 6.5,  4.5,  0.5],
           [ 6.5,  5.5,  0.5],
           [ 6.5,  6.5,  0.5]])
    """
    n_nei_vox = 8
    print("# - Detecting cell topological elements:")
    sh = np.array(img.shape)
    n_voxels = (sh[0] - 1) * (sh[1] - 1) * (sh[2] - 1)
    print(
        "  - Computing the neighborhood matrix of non-marginal voxels (n={})...".format(
            n_voxels))
    start_time = time.time()
    # - Create the neighborhood matrix of each pointel:
    neighborhood_img = []
    for x in np.arange(-1, 1):
        for y in np.arange(-1, 1):
            for z in np.arange(-1, 1):
                neighborhood_img.append(
                    img[1 + x:sh[0] + x, 1 + y:sh[1] + y, 1 + z:sh[2] + z])

    # - Reshape the neighborhood matrix in a N_voxel x n_nei_vox:
    neighborhood_img = np.sort(
        np.transpose(neighborhood_img, (1, 2, 3, 0))).reshape((sh - 1).prod(),
                                                              n_nei_vox)
    elapsed_time(start_time)

    print("  - Filtering out voxels surrounded only by similar label...")
    # - Detect the voxels that are not alone (only neighbors to themself, or same label around):
    non_flat = np.sum(
        neighborhood_img == np.tile(neighborhood_img[:, :1], (1, n_nei_vox)),
        axis=1) != n_nei_vox
    # - Keep only these "non flat" neighborhood:
    neighborhood_img = neighborhood_img[non_flat]
    elapsed_time(start_time)

    n_voxels_elem = neighborhood_img.shape[0]
    pc = float(n_voxels - n_voxels_elem) / n_voxels * 100
    print("\tRemoved {}% of the initial voxel matrix!".format(round(pc, 3)))

    print("  - Creating the associated coordinate matrix...")
    start_time = time.time()
    # - Create the coordinate matrix associated to each voxels of the neighborhood matrix:
    vertex_coords = np.transpose(
        np.mgrid[0:sh[0] - 1, 0:sh[1] - 1, 0:sh[2] - 1], (1, 2, 3, 0)).reshape(
        (sh - 1).prod(), 3) + 0.5
    # - Keep only these "non flat" coordinates:
    vertex_coords = vertex_coords[non_flat]
    elapsed_time(start_time)

    print("  - Computing the neighborhood size...")
    start_time = time.time()
    # - Keep the "unique values" in each neighborhood:
    neighborhoods = list(map(np.unique, neighborhood_img))
    neighborhoods = np.array(neighborhoods)
    # - Compute the neighborhood size of each voxel:
    neighborhood_size = list(map(len, neighborhoods))
    neighborhood_size = np.array(neighborhood_size)
    elapsed_time(start_time)

    print(
        "  - Creating dictionary of voxels coordinates detected as topological elements (n={})...".format(
            n_voxels_elem))
    if elem_order is None:
        elem_order = range(3)
    # - Separate the voxels depending on the size of their neighborhood:
    #   "wall" is a dimension 2 element with a neighborhood size == 2;
    #   "wall-edge" is a dimension 1 element with a neighborhood size == 3;
    #   "cell-vertex" is a dimension 0 element with a neighborhood size == 4+;
    topological_elements = {}
    start_time = time.time()
    for dimension in elem_order:
        try:
            assert dimension in range(3)
        except AssertionError:
            print("Given element orders should be in [0, 1, 2]!")
            continue
        # - Make a mask indexing desired 'neighborhood_size':
        dim_mask = neighborhood_size == 4 - dimension
        msg = "  --Sorting {} voxels as topological elements of order {} ({})..."
        if dimension == 2:
            elem_type = "faces"
        elif dimension == 1:
            elem_type = "edges"
        else:
            elem_type = "nodes"
        print(msg.format(sum(dim_mask), dimension, elem_type))
        # - Get all coordinates corresponding to selected 'neighborhood_size':
        element_points = vertex_coords[dim_mask]
        # - Get labels list for this given 'neighborhood_size':
        element_cells = np.array([p for p in neighborhoods[dim_mask]], int)

        # - In case of "cell-vertex" try to find 5-neighborhood:
        if (dimension == 0) & ((neighborhood_size >= 5).sum() > 0):
            # - Make a mask indexing 'neighborhood_size == 5':
            mask_5 = neighborhood_size == 5
            # - Get all coordinates for 'neighborhood_size == 5':
            clique_vertex_points = np.concatenate(
                [(p, p) for p in vertex_coords[mask_5]])
            # - Get labels list for 'neighborhood_size == 5':
            clique_vertex_cells = np.concatenate(
                [[p[:4], np.concatenate([[p[0]], p[2:]])] for p in
                 neighborhoods[mask_5]]).astype(int)
            # - Add them to the 4-neighborhood arrays of coordinates and labels:
            element_points = np.concatenate(
                [element_points, clique_vertex_points])
            element_cells = np.concatenate([element_cells, clique_vertex_cells])

        if element_cells != np.array([]):
            # - Remove duplicate of labels n-uplets, with 'n = 4 - dim':
            unique_cell_elements = array_unique(element_cells)
            # - ??
            element_matching = vq(element_cells, unique_cell_elements)[0]
            # - Make a dictionary of all {(n-uplet) : np.array(coordinates)}:
            topological_elements[dimension] = dict(
                zip([tuple(e) for e in unique_cell_elements],
                    [element_points[element_matching == e] for e, _ in
                     enumerate(unique_cell_elements)]))
        else:
            print(
                "WARNING: Could not find topological elements of order {}!".format(
                    dimension))
            topological_elements[dimension] = None

    elapsed_time(start_time)
    return topological_elements


def array_unique(array, return_index=False):
    """
    Return an array made of the unique occurrence of each rows.

    Parameters
    ----------
    array : numpy.array
        the array to compare by rows
    return_index : bool, optional
        if ``False`` (default) do NOT return the index of the unique rows, else do
        return them

    Returns
    -------
    array_unique : numpy.array
        the array made of unique rows
    unique_rows : numpy.array, if ``return_index == True``
        index of the unique rows

    Example
    -------
    >>> from timagetk.components.labelled_image import array_unique
    >>> a = np.array([[0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5]])
    >>> array_unique(a)
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5]])
    """
    _, unique_rows = np.unique(np.ascontiguousarray(array).view(
        np.dtype((np.void, array.dtype.itemsize * array.shape[1]))),
        return_index=True)
    if return_index:
        return array[unique_rows], unique_rows
    else:
        return array[unique_rows]


def real_indices(slices, resolutions):
    """
    Transform the discrete (voxels based) coordinates of the bounding box
    (slices) into their real-world size using resolutions.

    Parameters
    ----------
    slices : list
        list of slices or bounding boxes found using scipy.ndimage.find_objects
    resolutions : list
        length-2 (2D) or length-3 (3D) vector of float indicating the size of a
        voxel in real-world units

    Returns
    -------
    list
        list of slice objects
    """
    return [(s.start * r, s.stop * r) for s, r in zip(slices, resolutions)]

class LabelledImage(SpatialImage):
    """
    Class to manipulate labelled SpatialImage, eg. a segmented image.
    """

    def __new__(cls, image, **kwargs):
        """
        LabelledImage construction method.

        Parameters
        ----------
        image : numpy.array or SpatialImage
            a numpy array or a SpatialImage containing a labelled array

        kwargs
        ------
        origin : list, optional
            coordinates of the origin in the image, default: [0,0] or [0,0,0]
        voxelsize : list, optional.
            image voxelsize, default: [1.0,1.0] or [1.0,1.0,1.0]
        dtype : str, optional
            image type, default dtype = input_array.dtype
        metadata= : dict, optional
            dictionary of image metadata, default is an empty dict

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> from timagetk.components import LabelledImage
        >>> test_array = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
        >>> test_array[0,:] = np.ones((5,), dtype=np.uint8)
        >>> # - Construct from a NumPy array:
        >>> lab_image = LabelledImage(test_array, voxelsize=[0.5,0.5], no_label_id=0)
        >>> # - Construct from a SpatialImage:
        >>> image_1 = SpatialImage(test_array, voxelsize=[0.5,0.5])
        >>> lab_image = LabelledImage(image_1, no_label_id=0)
        >>> isinstance(lab_image, np.ndarray)
        True
        >>> isinstance(lab_image, SpatialImage)
        True
        >>> isinstance(lab_image, LabelledImage)
        True
        >>> print(lab_image.voxelsize)
        [0.5, 0.5]
        >>> print(lab_image.no_label_id)
        0
        """
        # logging.debug('LabelledImage.__new__ got a {} instance!'.format(
        #     clean_type(image)))
        # logging.debug('LabelledImage.__new__ got kwargs: {}.'.format(kwargs))
        # - Get variables for LabelledImage instantiation:
        if isinstance(image, SpatialImage):
            # -- Can be a SpatialImage or any class inheriting from it:
            return super(LabelledImage, cls).__new__(cls, image, **kwargs)
        elif isinstance(image, np.ndarray):
            # -- Case where constructing from a NumPy array:
            origin = kwargs.pop('origin', None)
            voxelsize = kwargs.pop('voxelsize', None)
            dtype = kwargs.pop('dtype', image.dtype)
            metadata = kwargs.pop('metadata', None)
            return super(LabelledImage, cls).__new__(cls, image,
                                                     origin=origin,
                                                     voxelsize=voxelsize,
                                                     dtype=dtype,
                                                     metadata=metadata,
                                                     **kwargs)
        else:
            msg = "Undefined construction method for type '{}'!"
            raise NotImplementedError(msg.format(type(image)))

    def __init__(self, image, no_label_id=None, **kwargs):
        """
        LabelledImage initialisation method.

        Parameters
        ----------
        image : numpy.array or SpatialImage
            a numpy array or a SpatialImage containing a labelled array
        no_label_id : int, optional
            if given define the "unknown label" (ie. not a label)
        """
        # - In case a LabelledImage is contructed from a LabelledImage, get the attributes values:
        if isinstance(image, LabelledImage):
            attr_list = ["no_label_id"]
            attr_dict = get_attributes(image, attr_list)
            class_name = get_class_name(image)
            msg = "Overriding optional keyword arguments '{}' ({}) with defined attribute ({}) in given '{}'!"
            # -- Check necessity to override 'origin' with attribute value:
            if attr_dict['no_label_id'] is not None:
                if no_label_id is not None and no_label_id != attr_dict[
                    'no_label_id']:
                    print(msg.format('no_label_id', no_label_id,
                                     attr_dict['no_label_id'],
                                     class_name))
                no_label_id = attr_dict['no_label_id']

            # -- Check 'class' definition in 'timagetk' metadata:
            try:
                md_class = image.metadata['timagetk']['class']
            except KeyError:
                warn_msg = "Initializing from a 'LabelledImage' without 'class' entry in 'timagetk' metadata!"
                # logging.warning(warn_msg)
                self.metadata.update({'timagetk': {'class': 'LabelledImage'}})
            else:
                if md_class != 'LabelledImage':
                    warn_msg = "Initializing from a 'LabelledImage' without correct 'class' definition in 'timagetk' metadata!"
                    warn_msg += "\n\{'timagetk': \{'class': {}\}\}".format(
                        md_class)
                    logging.warning(warn_msg)
                    # TODO: update 'class' entry to 'LabelledImage' ?!
        else:
            # - Adding class to metadata:
            self.metadata.update({'timagetk': {'class': 'LabelledImage'}})

        # - Initializing EMPTY hidden attributes:
        # -- Property hidden attributes:
        self._no_label_id = None  # id refering to the absence of label

        # -- Topological element of order 3 are called 'labels':
        self._labels = None  # list of labels
        self._label_bboxes = {}  # dict of label bounding boxes
        self._neighbors = {}  # unfiltered neighborhood label-dict {vid_i: neighbors(vid_i)}

        # -- Topological element of order 2 are called 'faces':
        self._faces = None  # list of faces
        self._face_bboxes = {}  # dict of face bounding boxes
        self._face_voxels = {}  # dict of face voxel coordinates

        # -- Topological element of order 1 are called 'edges':
        self._edges = None  # list of edges
        self._edge_bboxes = {}  # dict of edge bounding boxes
        self._edge_voxels = {}  # dict of edge voxel coordinates

        # -- Topological element of order 0 are called 'nodes':
        self._nodes = None  # list of nodes
        self._node_bboxes = {}  # dict of node bounding boxes
        self._node_voxels = {}  # dict of node voxel coordinates

        # - Initialise object property and most used hidden attributes:
        # -- Define the "no_label_id" value, if any (can be None):
        self.no_label_id = no_label_id
        # -- Get the list of labels found in the image:
        self.labels()
        n_lab = len(self.labels())
        if kwargs.get('verbose', False):
            print("Initialized `LabelledImage` object with {} labels!".format(n_lab))
            if n_lab <= 15:
                print("Found list of labels: {}".format(self.labels()))

        # -- Get the boundingbox dictionary:
        self.boundingbox()

    def __str__(self):
        """
        Method called when printing the object.
        """
        msg = "LabelledImage object with following metadata:\n"
        md = self.metadata
        msg += '\n'.join(['   - {}: {}'.format(k, v) for k, v in md.items()])
        return msg

    @property
    def no_label_id(self):
        """
        Get the value associated to no label.
        This is used as "unknown label" or "erase value".

        Returns
        -------
        no_label_id : int
            the value defined as the "no_label_id"

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        WARNING : no value defined for the 'no label' id!
        >>> im = LabelledImage(a, no_label_id=1)
        >>> im.labels()
        [2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        1
        """
        if self._no_label_id is None:
            print("WARNING: no value defined for the 'no label' id!")
        return self._no_label_id

    @no_label_id.setter
    def no_label_id(self, value):
        """
        Set the value of the label indicating "unknown" label.

        Parameters
        ----------
        value : int
            value to be defined as the "no_label_id"

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        WARNING : no value defined for the 'no label' id!
        >>> im.no_label_id = 1
        >>> im.labels()
        [2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        1
        """
        if not isinstance(value, int) and value is not None:
            print("Provided value '{}' is not an integer!".format(value))
            return
        else:
            self._no_label_id = value
        self.metadata = {'no_label_id': self.no_label_id}

    def _defined_no_label_id(self):
        """
        Tests if '_no_label_id' attribute is defined, if not raise a ValueError.
        """
        try:
            assert self._no_label_id is not None
        except AssertionError:
            msg = "Attribute 'no_label_id' is not defined (None)."
            msg += "Please set it (integer) before calling this function!"
            raise ValueError(msg)
        return

    def topological_elements(self, element_order=None):
        """
        Extract the coordinates of topological elements of order 2 (ie. face),
        1 (ie. edge) and 0 (ie. node). Return their coordinates grouped by pair,
        triplet and quadruplet of labels.

        Parameters
        ----------
        element_order : int or list, optional
            list of dimensional order of the elements to return

        Returns
        -------
        topo_elem : dict
            dictionary with topological elements order as key, each containing
            dictionaries of n-uplets as keys and coordinates array as values

        Notes
        -----
        The order of the labels in the tuple defining the key is irrelevant, ie.
        coordinates of face (2, 5) is the same than (5, 2).
        """
        import copy as cp
        if isinstance(element_order, int):
            element_order = [element_order]
        if element_order is None:
            if self.is2D():
                element_order = [2, 1]
            else:
                element_order = range(3)
        if 0 in element_order and self.is2D():
            print("There is no elements of order 0 in a 2D image.")
            element_order.remove(0)
            if element_order == []:
                return

        # - List missing order of topological element dictionary
        elem_order = cp.copy(element_order)
        if element_order is not None:
            # remove potential duplicates:
            element_order = list(set(element_order))
            # remove already computed elements order:
            if 2 in element_order and self._face_voxels != {}:
                elem_order.remove(2)
            if 1 in element_order and self._edge_voxels != {}:
                elem_order.remove(1)
            if 0 in element_order and self._node_voxels != {}:
                elem_order.remove(0)

        # - If element are missing, compute them and save them to attributes:
        if elem_order != []:
            if self.is2D():
                topo_elem = topological_elements_extraction2D(self, elem_order)
            else:
                topo_elem = topological_elements_extraction3D(self, elem_order)

            # - Get the face coordinates:
            if 2 in topo_elem:
                self._face_voxels = topo_elem[2]
                self._faces = self._face_voxels.keys()
            # - Get the edge coordinates:
            if 1 in topo_elem:
                self._edge_voxels = topo_elem[1]
                self._edges = self._edge_voxels.keys()
            # - Get the node coordinates:
            if 0 in topo_elem:
                self._node_voxels = topo_elem[0]
                self._nodes = self._node_voxels.keys()
        else:
            topo_elem = {}

        # - Get required but already computed dict of topological elements:
        if 2 in element_order and 2 not in topo_elem:
            topo_elem[2] = self._face_voxels
        if 1 in element_order and 1 not in topo_elem:
            topo_elem[1] = self._edge_voxels
        if 0 in element_order and 0 not in topo_elem:
            topo_elem[0] = self._node_voxels

        return topo_elem

    # --------------------------------------------------------------------------
    # LABEL based methods:
    # --------------------------------------------------------------------------
    def labels(self, labels=None):
        """
        Get the list of labels found in the image, or make sure provided labels
        exists.

        Parameters
        ----------
        labels : int or list, optional
            if given, used to filter the returned list, else return all labels
            defined in the image by default

        Returns
        -------
        list
            list of label found in the image, except for 'no_label_id'
            (if defined)

        Notes
        -----
        Value defined for 'no_label_id' is removed from the returned list of
        labels since it does not refer to one.

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        """
        if isinstance(labels, int):
            labels = [labels]
        # - If the hidden label attribute is None, list all labels in the array:
        if self._labels is None:
            self._labels = list(map(int, np.unique(self.get_array())))

        # - Remove value attributed to 'no_label_id':
        unwanted_set = {self._no_label_id}
        label_set = set(self._labels) - unwanted_set
        if labels:
            label_set = list(label_set & set(labels))

        return list(map(int, label_set))

    def nb_labels(self):
        """
        Return the number of labels.

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.nb_labels()
        7
        >>> im = LabelledImage(a, no_label_id=1)
        >>> im.nb_labels()
        6
        """
        return len(self.labels())

    def is_label_in_image(self, label):
        """
        Returns ``True`` if the label is found in the image, else ``False``.

        Parameters
        ----------
        label : int
            label to check

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.is_label_in_image(7)
        True
        >>> im.is_label_in_image(10)
        False
        """
        return label in self.get_array()

    def boundingbox(self, labels=None, real=False, verbose=False):
        """
        Return the bounding-box of a cell for given 'labels'.

        Parameters
        ----------
        labels : None|int|list(int) or str, optional
            if ``None`` (default) returns all labels.
            if an integer, make sure it is in self.labels()
            if a list of integers, make sure they are in self.labels()
            if a string, should be in LABEL_STR to get corresponding
            list of cells (case insensitive)
        real : bool, optional
            if ``False`` (default), return the bounding-boxes in voxel units, else
            in real units.
        verbose : bool, optional
            control verbosity of the function

        Example
        -------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.boundingbox(7)
        (slice(0, 3), slice(2, 4), slice(0, 1))
        >>> im.boundingbox([7,2])
        [(slice(0, 3), slice(2, 4), slice(0, 1)), (slice(0, 3), slice(0, 2), slice(0, 1))]
        >>> im.boundingbox()
        [(slice(0, 4), slice(0, 6), slice(0, 1)),
        (slice(0, 3), slice(0, 2), slice(0, 1)),
        (slice(1, 3), slice(4, 6), slice(0, 1)),
        (slice(3, 4), slice(3, 4), slice(0, 1)),
        (slice(1, 2), slice(2, 3), slice(0, 1)),
        (slice(1, 2), slice(1, 2), slice(0, 1)),
        (slice(0, 3), slice(2, 4), slice(0, 1))]
        """
        if labels is None:
            labels = self.labels()

        # - Starts with integer case since it is the easiest:
        if isinstance(labels, int):
            try:
                assert labels in self._label_bboxes
            except AssertionError:
                image = self.get_array()
                bbox = nd.find_objects(image == labels, max_label=1)[0]
                self._label_bboxes[labels] = bbox
            return self._label_bboxes[labels]

        # - Create a dict of bounding-boxes using 'scipy.ndimage.find_objects':
        known_bbox = [l in self._label_bboxes for l in labels]
        image = self.get_array()
        if self._label_bboxes is None or not all(known_bbox):
            max_lab = max(labels)
            if verbose:
                print("Computing {} bounding-boxes...".format(max_lab))
            bbox = nd.find_objects(image, max_label=max_lab)
            # NB: scipy.ndimage.find_objects start at 1 (and python index at 0), hence to access i-th element, we have to use (i-1)-th index!
            self._label_bboxes = {n: bbox[n - 1] for n in range(1, max_lab + 1)}

        # - Filter returned bounding-boxes to the (cleaned) given list of labels
        bboxes = {l: self._label_bboxes[l] for l in labels}
        if real:
            vxs = self.voxelsize
            bboxes = {l: real_indices(bbox, vxs) for l, bbox in bboxes.items()}

        return bboxes

    # def label_array(self, label, dilation=None):
    #     """
    #     Returns an array made from the labelled image cropped around the label
    #     bounding box.

    #     Parameters
    #     ----------
    #     label : int
    #         label for which to extract the neighbors
    #     dilation : int, optional
    #         if defined (default is ``None``), use this value as a dilation factor
    #         (in every directions) to be applied to the label boundingbox

    #     Returns
    #     -------
    #     numpy.array
    #         labelled array cropped around the label bounding box

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 1, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a)
    #     >>> im.label_array(7)
    #     array([[7, 7],
    #            [5, 7],
    #            [1, 7]])
    #     >>> im.label_array(7, dilation=1)
    #     array([[2, 7, 7, 1],
    #            [6, 5, 7, 3],
    #            [2, 1, 7, 3],
    #            [1, 1, 4, 1]])
    #     """
    #     # - Get the slice for given label:
    #     label_slice = self.boundingbox(label)
    #     # - Create the cropped image when possible:
    #     if label_slice is None:
    #         crop_img = self.get_array()
    #     else:
    #         if dilation:
    #             label_slice = dilation_by(label_slice, dilation)
    #         crop_img = self.get_array()[label_slice]

    #     return crop_img

    # def _neighbors_with_mask(self, label):
    #     """
    #     Sub-function called when only one label is given to self.neighbors

    #     Parameters
    #     ----------
    #     label : int
    #         the label for which to compute the neighborhood

    #     Returns
    #     -------
    #     list
    #         list of neighbors for given label
    #     """
    #     # - Compute the neighbors and update the unfiltered neighbors dict:
    #     if not label in self._neighbors:
    #         crop_img = self.label_array(label)
    #         self._neighbors[label] = neighbors_from_image(crop_img, label)

    #     return self._neighbors[label]

    # def _neighborhood_with_mask(self, labels, **kwargs):
    #     """
    #     Sub-function called when a list of 'labels' is given to self.neighbors()

    #     Parameters
    #     ----------
    #     label : int
    #         the label for which to compute the neighborhood

    #     Returns
    #     -------
    #     dict
    #         neighborhood dictionary for given list of labels
    #     """
    #     verbose = kwargs.get('verbose', False)

    #     # - Check we have all necessary bounding boxes...
    #     self.boundingbox(labels, verbose=verbose)

    #     # - Try a shortcut: 'self._neighbors' might have all required 'labels'...
    #     miss_labels = [l for l in labels if not l in self._neighbors]
    #     n_miss = len(miss_labels)

    #     # - Compute the neighborhood for labels without (unfiltered) neighbors list:
    #     if miss_labels:
    #         t_start = time.time()
    #         if verbose:
    #             print("-- Computing the neighbors list for {} labels...".format(n_miss))
    #         progress = 0
    #         nb_labels = len(miss_labels)
    #         for n, label in enumerate(miss_labels):
    #             progress = percent_progress(progress, n, nb_labels)
    #             # compute the neighborhood for the given label
    #             self._neighbors[label] = neighbors_from_image(self.label_array(label, dilation=1), label)

    #         elapsed_time(t_start)

    #     neighborhood = {l: self._neighbors[l] for l in labels}
    #     return neighborhood

    # def neighbors(self, labels=None, verbose=True):
    #     """Return the neighbors dictionary of each label.

    #     Parameters
    #     ----------
    #     labels : None|int|list(int) or str, optional
    #         if None (default) returns all labels.
    #         if an integer, make sure it is in self.labels()
    #         if a list of integers, make sure they are in self.labels()
    #         if a string, should be in LABEL_STR to get corresponding
    #         list of cells (case insensitive)
    #     verbose : bool, optional
    #         control verbosity

    #     Returns
    #     -------
    #     dict
    #         {i: neighbors(i)} for i in `labels`

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 2, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a)
    #     >>> im.neighbors(7)
    #     [1, 2, 3, 4, 5]
    #     >>> im.neighbors([7,2])
    #     {7: [1, 2, 3, 4, 5], 2: [1, 6, 7] }
    #     >>> im.neighbors()
    #     {1: [2, 3, 4, 5, 6, 7],
    #      2: [1, 6, 7],
    #      3: [1, 7],
    #      4: [1, 7],
    #      5: [1, 6, 7],
    #      6: [1, 2, 5],
    #      7: [1, 2, 3, 4, 5] }
    #     >>> im = LabelledImage(a, no_label_id=1)
    #     >>> im.neighbors(1)
    #     [2, 3, 4, 7]
    #     >>> im.neighbors([1, 2])
    #     {1: [2, 3, 4, 7], 2: [1, 5, 7]}

    #     """
    #     if labels is None:
    #         labels = self.labels()

    #     # - Transform length-1 list to integers
    #     if isinstance(labels, list) and len(labels) == 1:
    #         labels = labels[0]

    #     # - Neighborhood computing:
    #     if isinstance(labels, int):
    #         try:
    #             assert self.is_label_in_image(labels)
    #         except AssertionError:
    #             raise ValueError(MISS_LABEL.format('', 'is', labels))
    #         if verbose:
    #             print("Extracting neighbors for label {}...".format(labels))
    #         return self._neighbors_with_mask(labels)
    #     else:  # list case:
    #         try:
    #             assert labels != []
    #         except AssertionError:
    #             raise ValueError(MISS_LABEL.format('s', 'are', labels))
    #         if verbose:
    #             n_lab = len(labels)
    #             print("Extracting neighbors for {} labels...".format(n_lab))
    #         return self._neighborhood_with_mask(labels, verbose=verbose)

    # # --------------------------------------------------------------------------
    # # FACE based methods:
    # # --------------------------------------------------------------------------
    # def faces(self, faces=None):
    #     """
    #     Get the list of faces found in the image, or make sure provided faces
    #     exists.

    #     Parameters
    #     ----------
    #     faces : len-2 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all faces
    #         defined in the image by default

    #     Returns
    #     -------
    #     list
    #         list of faces found in the image
    #     """
    #     if isinstance(faces, tuple):
    #         faces = [faces]
    #     if isinstance(faces, list):
    #         ok = [isinstance(tuple, f) and len(f) == 2 for f in faces]
    #         try:
    #             assert all(ok)
    #         except AssertionError:
    #             msg = "Input 'faces' should be a list of length-2 tuples!"
    #             raise TypeError(msg)

    #     # - If the hidden label attribute is None, list all labels in the array:
    #     if self._faces is None:
    #         self.topological_elements(element_order=2)

    #     if faces:
    #         # need to reorder given list of 'faces', might not be label sorted:
    #         faces = [stuple(f) for f in faces]
    #         return list(self._faces & set(faces))
    #     else:
    #         return list(self._faces)

    # def face_coordinates(self, faces=None):
    #     """
    #     Get a dictionary of face coordinates.

    #     Parameters
    #     ----------
    #     faces : len-2 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all faces
    #         defined in the image by default

    #     Returns
    #     -------
    #     dict
    #         dictionary of len-2 labels with their voxel coordinates
    #     """
    #     faces_list = self.faces(faces)
    #     if faces:
    #         return {f: self._face_voxels[f] for f in faces_list}
    #     else:
    #         return self._face_voxels

    # # --------------------------------------------------------------------------
    # # EDGE based methods:
    # # --------------------------------------------------------------------------
    # def edges(self, edges=None):
    #     """
    #     Get the list of edges found in the image, or make sure provided edges
    #     exists.

    #     Parameters
    #     ----------
    #     edges : len-3 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all edges
    #         defined in the image by default

    #     Returns
    #     -------
    #     list
    #         list of edges found in the image
    #     """
    #     if isinstance(edges, tuple):
    #         edges = [edges]
    #     if isinstance(edges, list):
    #         ok = [isinstance(tuple, e) and len(e) == 3 for e in edges]
    #         try:
    #             assert all(ok)
    #         except AssertionError:
    #             msg = "Input 'edges' should be a list of length-2 tuples!"
    #             raise TypeError(msg)

    #     # - If the hidden label attribute is None, list all labels in the array:
    #     if self._edges is None:
    #         self.topological_elements(element_order=1)

    #     if edges:
    #         # need to reorder given list of 'edges', might not be label sorted:
    #         edges = [stuple(e) for e in edges]
    #         return list(self._edges & set(edges))
    #     else:
    #         return list(self._edges)

    # def edge_coordinates(self, edges=None):
    #     """
    #     Get a dictionary of edge coordinates.

    #     Parameters
    #     ----------
    #     edges : len-3 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all edges
    #         defined in the image by default

    #     Returns
    #     -------
    #     dict
    #         dictionary of len-3 labels with their voxel coordinates
    #     """
    #     edges_list = self.edges(edges)
    #     if edges:
    #         return {e: self._edge_voxels[e] for e in edges_list}
    #     else:
    #         return self._edge_voxels

    # # --------------------------------------------------------------------------
    # # NODE based methods:
    # # --------------------------------------------------------------------------
    # def nodes(self, nodes=None):
    #     """
    #     Get the list of nodes found in the image, or make sure provided nodes
    #     exists.

    #     Parameters
    #     ----------
    #     nodes : len-4 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all nodes
    #         defined in the image by default

    #     Returns
    #     -------
    #     list
    #         list of nodes found in the image
    #     """
    #     if isinstance(nodes, tuple):
    #         nodes = [nodes]
    #     if isinstance(nodes, list):
    #         ok = [isinstance(tuple, n) and len(n) == 4 for n in nodes]
    #         try:
    #             assert all(ok)
    #         except AssertionError:
    #             msg = "Input 'nodes' should be a list of length-2 tuples!"
    #             raise TypeError(msg)

    #     # - If the hidden label attribute is None, list all labels in the array:
    #     if self._nodes is None:
    #         self.topological_elements(element_order=0)

    #     if nodes:
    #         # need to reorder given list of 'nodes', might not be label sorted:
    #         nodes = [stuple(n) for n in nodes]
    #         return list(self._nodes & set(nodes))
    #     else:
    #         return list(self._nodes)

    # def node_coordinates(self, nodes=None):
    #     """
    #     Get a dictionary of node coordinates.

    #     Parameters
    #     ----------
    #     nodes : len-4 tuple or list(tuple), optional
    #         if given, used to filter the returned list, else return all nodes
    #         defined in the image by default

    #     Returns
    #     -------
    #     dict
    #         dictionary of len-4 labels with their voxel coordinates
    #     """
    #     nodes_list = self.nodes(nodes)
    #     if nodes:
    #         return {n: self._node_voxels[n] for n in nodes_list}
    #     else:
    #         return self._node_voxels

    # # --------------------------------------------------------------------------
    # # LabelledImage edition functions:
    # # --------------------------------------------------------------------------
    # def isometric_resampling(self, method='min', **kwargs):
    #     """
    #     Performs isometric resampling of the image using either the min, the max
    #     or a a given voxelsize value.

    #     Parameters
    #     ----------
    #     method : str or float, optional
    #         change voxelsize to 'min' (default) or 'max' value of original
    #         voxelsize or to a given value.

    #     Returns
    #     -------

    #     """
    #     return SpatialImage.isometric_resampling(self, method, option='label')

    # def get_image_with_labels(self, labels):
    #     """
    #     Returns a LabelledImage with only the selected 'labels', the rest are
    #     replaced by "self._no_label_id".

    #     Parameters
    #     ----------
    #     labels : int or list(int)
    #         if None (default) returns all labels.
    #         if an integer, make sure it is in self.labels()
    #         if a list of integers, make sure they are in self.labels()
    #         if a string, should be in LABEL_STR to get corresponding
    #         list of cells (case insensitive)

    #     Returns
    #     -------
    #     LabelledImage
    #         labelled image with 'labels' only

    #     Notes
    #     -----
    #     Require property 'no_label_id' to be defined!

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 1, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a, no_label_id=0)
    #     >>> im.get_image_with_labels([2, 5])
    #     LabelledImage([[0, 2, 0, 0, 0, 0],
    #                    [0, 0, 5, 0, 0, 0],
    #                    [2, 2, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 0]])
    #     """
    #     self._defined_no_label_id()
    #     all_labels = self.labels()
    #     labels = self.labels(labels)
    #     off_labels = list(set(all_labels) - set(labels))

    #     if len(off_labels) == 0:
    #         print("WARNING: you selected ALL label!")
    #         return self
    #     if len(labels) == 0:
    #         print("WARNING: you selected NO label!")
    #         return

    #     if len(labels) < len(off_labels):
    #         template_im = image_with_labels(self, labels)
    #     else:
    #         template_im = image_without_labels(self, off_labels)

    #     return template_im

    # def get_image_without_labels(self, labels):
    #     """Returns a LabelledImage without the selected labels.

    #     Parameters
    #     ----------
    #     labels : int|list or str
    #         label or list of labels to keep in the image.
    #         strings might be processed trough 'self.labels_checker()'

    #     Returns
    #     -------
    #     LabelledImage
    #         labelled image with 'labels' only

    #     Notes
    #     -----
    #     Require property 'no_label_id' to be defined!

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 1, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a, no_label_id=0)
    #     >>> im.get_image_without_labels([2, 5])
    #     LabelledImage([[1, 0, 7, 7, 1, 1],
    #                    [1, 6, 0, 7, 3, 3],
    #                    [0, 0, 1, 7, 3, 3],
    #                    [1, 1, 1, 4, 1, 1]])
    #     """
    #     all_labels = self.labels()
    #     labels = self.labels(labels)
    #     off_labels = list(set(all_labels) - set(labels))
    #     return self.get_image_with_labels(off_labels)

    # def get_wall_image(self, labels=None, **kwargs):
    #     """Get a labelled image with walls only (hollowed out cells).

    #     Parameters
    #     ----------
    #     labels : list, optional
    #         list of labels to return in the wall image, by default (None) return
    #         all labels
    #     kwargs : dict, optional
    #         given to 'hollow_out_cells', 'verbose' accepted

    #     Returns
    #     -------
    #     LabelledImage
    #         the labelled wall image

    #     Notes
    #     -----
    #     The "inside" of each label is replaced with `self.no_label_id`.

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 2, 2, 2, 3, 3, 3],
    #                       [1, 2, 2, 2, 2, 3, 3, 3],
    #                       [1, 2, 2, 2, 2, 3, 3, 3],
    #                       [1, 2, 2, 2, 2, 3, 3, 3]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a, no_label_id=0)
    #     >>> im.get_wall_image([2, 3])

    #     """
    #     if labels is not None:
    #         image = self.get_image_with_labels(labels)
    #     else:
    #         image = self

    #     return hollow_out_labels(image, **kwargs)

    # def fuse_labels_in_image(self, labels, new_value='min', verbose=True):
    #     """
    #     Fuse the provided list of labels to a given new_value, or the min or max
    #     of the list of labels.

    #     Parameters
    #     ----------
    #     labels : list
    #         list of labels to fuse
    #     new_value : int or str, optional
    #         value used to replace the given list of labels, by default use the
    #         min value of the ``labels`` list. Can also be the max value.
    #     verbose : bool, optional
    #         control verbosity

    #     Returns
    #     -------
    #     Nothing, modify the LabelledImage array (re-instantiate the object)

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 1, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a, no_label_id=0)
    #     >>> im.fuse_labels_in_image([6, 7], new_value=8)
    #     LabelledImage([[1, 2, 8, 8, 1, 1],
    #                    [1, 8, 5, 8, 3, 3],
    #                    [2, 2, 1, 8, 3, 3],
    #                    [1, 1, 1, 4, 1, 1]])
    #     >>> im.fuse_labels_in_image([6, 7], new_value='min')
    #     LabelledImage([[1, 2, 6, 6, 1, 1],
    #                    [1, 6, 5, 6, 3, 3],
    #                    [2, 2, 1, 6, 3, 3],
    #                    [1, 1, 1, 4, 1, 1]])
    #     >>> im.fuse_labels_in_image([6, 7], new_value='max')
    #     LabelledImage([[1, 2, 7, 7, 1, 1],
    #                    [1, 7, 5, 7, 3, 3],
    #                    [2, 2, 1, 7, 3, 3],
    #                    [1, 1, 1, 4, 1, 1]])
    #     """
    #     if isinstance(labels, np.ndarray):
    #         labels = labels.tolist()
    #     elif isinstance(labels, set):
    #         labels = list(labels)
    #     else:
    #         assert isinstance(labels, list) and len(labels) >= 2

    #     # - Make sure 'labels' is correctly formatted:
    #     labels = self.labels(labels)
    #     nb_labels = len(labels)
    #     # - If no labels to remove, its over:
    #     if nb_labels == 0:
    #         print('No labels to fuse!')
    #         return

    #     # - Define the integer value of 'new_value':
    #     if new_value == "min":
    #         new_value = min(labels)
    #         labels.remove(new_value)
    #     elif new_value == "max":
    #         new_value = max(labels)
    #         labels.remove(new_value)
    #     elif isinstance(new_value, int):
    #         if self.is_label_in_image(new_value) and not new_value in labels:
    #             msg = "Given new_value is in the image and not in the list of labels."
    #             raise ValueError(msg)
    #         if new_value in labels:
    #             labels.remove(new_value)
    #     else:
    #         raise NotImplementedError(
    #             "Unknown 'new_value' definition for '{}'".format(new_value))

    #     t_start = time.time()  # timer
    #     # - Label "fusion" loop:
    #     no_bbox = []
    #     progress = 0
    #     if verbose:
    #         print(
    #             "Fusing the following {} labels: {} to new_value '{}'.".format(
    #                 nb_labels, labels, new_value))
    #     for n, label in enumerate(labels):
    #         if verbose:
    #             progress = percent_progress(progress, n, nb_labels)
    #         # - Try to get the label's boundingbox:
    #         try:
    #             bbox = self.boundingbox(label)
    #         except KeyError:
    #             no_bbox.append(label)
    #             bbox = None
    #         # - Performs value replacement:
    #         array_replace_label(self, label, new_value, bbox)

    #     # - If some boundingbox were missing, print about it:
    #     if no_bbox:
    #         n = len(no_bbox)
    #         print("Could not find boundingbox for {} labels: {}".format(n,
    #                                                                     no_bbox))

    #     # - RE-INITIALIZE the object attributes to match new labels:
    #     self.__init__(self)

    #     # - May print about elapsed time:
    #     if verbose:
    #         elapsed_time(t_start)

    #     return

    # def remove_labels_from_image(self, labels, verbose=True):
    #     """
    #     Remove 'labels' from self.image using 'no_label_id'.

    #     Parameters
    #     ----------
    #     labels : list
    #         list of labels to remove from the image
    #     verbose : bool, optional
    #         control verbosity

    #     Returns
    #     -------
    #     Nothing, modify the LabelledImage array (re-instantiate the object)

    #     Notes
    #     -----
    #     Require property 'no_label_id' to be defined!

    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> a = np.array([[1, 2, 7, 7, 1, 1],
    #                       [1, 6, 5, 7, 3, 3],
    #                       [2, 2, 1, 7, 3, 3],
    #                       [1, 1, 1, 4, 1, 1]])
    #     >>> from timagetk.components import LabelledImage
    #     >>> im = LabelledImage(a, no_label_id=0)
    #     >>> im.remove_labels_from_image([6, 7])
    #     LabelledImage([[1, 2, 0, 0, 1, 1],
    #                    [1, 0, 5, 0, 3, 3],
    #                    [2, 2, 1, 0, 3, 3],
    #                    [1, 1, 1, 4, 1, 1]])
    #     """
    #     if isinstance(labels, int):
    #         labels = [labels]
    #     elif isinstance(labels, np.ndarray):
    #         labels = labels.tolist()
    #     elif isinstance(labels, set):
    #         labels = list(labels)
    #     else:
    #         assert isinstance(labels, list)

    #     # - Make sure 'labels' is correctly formatted:
    #     labels = self.labels(labels)
    #     nb_labels = len(labels)
    #     # - If no labels to remove, its over:
    #     if nb_labels == 0:
    #         print('No labels to remove!')
    #         return

    #     t_start = time.time()  # timer
    #     # - Remove 'labels' using bounding boxes to speed-up computation:
    #     no_bbox = []
    #     progress = 0
    #     if verbose:
    #         print("Removing {} labels.".format(nb_labels))
    #     for n, label in enumerate(labels):
    #         if verbose:
    #             progress = percent_progress(progress, n, nb_labels)
    #         # Try to get the label's boundingbox:
    #         try:
    #             bbox = self.boundingbox(label)
    #         except KeyError:
    #             no_bbox.append(label)
    #             bbox = None
    #         # Performs value replacement:
    #         array_replace_label(self, label, self.no_label_id, bbox)

    #     # - If some boundingbox were missing, print about it:
    #     if no_bbox:
    #         n = len(no_bbox)
    #         print("Could not find boundingbox for {} labels: {}".format(n,
    #                                                                     no_bbox))

    #     # - RE-INITIALIZE the object attributes to match new labels:
    #     self.__init__(self)

    #     # - May print about elapsed time:
    #     if verbose:
    #         elapsed_time(t_start)

    #     return
