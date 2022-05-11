import os
import h5py
import numpy as np
import pandas as pd
from skimage.io import imread, imsave

def prepare_files(directory, verbose=False):
    files = sorted(next(os.walk(directory))[2])
    if len(files) > 2 or len(files) == 0:
        raise ValueError("Two files are expected in {} (.tif and .h5 file). Found {} files".format(directory, len(files)))
    if len(files) == 2:
        for f in files:
            if not f.endswith('.h5') and not f.endswith('.tif'):
                raise ValueError("Only .h5 or .tif files are expected. Given {}".format(os.path.join(directory,f)))

        if files[0].endswith('.h5'):
            h5_file = os.path.join(directory, files[0])
            tif_file = os.path.join(directory, files[1])
        else:
            h5_file = os.path.join(directory, files[1])
            tif_file = os.path.join(directory, files[0])
    else:
        if files[0].endswith('.h5'):
            h5_file = os.path.join(directory, files[0])
            tif_file = 'none'
        else:
            tif_file = os.path.join(directory, files[0])
            h5_file = 'none'

    # Create .h5/.tif files if needed
    if h5_file == 'none':
        if verbose: print("Creating H5 file from file {}".format(tif_file))
        h5_file = tif_to_h5(tif_file, directory)
        if verbose: print("New file H5 created: {}".format(h5_file))
    if tif_file == 'none':
        if verbose: print("Creating TIF file from file {}".format(h5_file))
        tif_file = h5_to_tif(h5_file, directory)
        if verbose: print("New file TIF created: {}".format(tif_file))
    return h5_file, tif_file


def tif_to_h5(tif_path, out_dir):
    """Save tif file into h5 format."""
    filename = os.path.basename(tif_path)
    h5file_name = os.path.join(os.path.dirname(tif_path),filename.split('.')[0]+'.h5')
    img = imread(tif_path)

    # Create the h5 file (using lzf compression to save space)
    h5f = h5py.File(h5file_name, 'w')
    h5f.create_dataset('main', data=img, compression="lzf")
    h5f.close()

    return h5file_name

def h5_to_tif(h5_path, out_dir, dtype=np.uint16):
    """Save h5 file into tif format."""
    h5f = h5py.File(h5_path, 'r')
    k = list(h5f.keys())
    data = (h5f[k[0]])

    if data.ndim == 3:
        data = np.expand_dims(data, -1)

    # Data needs to be like this: (500, 4096, 4096, 1),  (z,x,y,c)
    if data.ndim != 4:
        raise ValueError("Data should be 4 dimensional, given {}".format(data.shape))

    filename = os.path.basename(h5_path)
    tiffile_name = os.path.join(os.path.dirname(h5_path),filename.split('.')[0]+'.tif')
    aux = np.expand_dims(data.transpose((0,3,1,2)), -1).astype(dtype)
    imsave(tiffile_name, aux, imagej=True, metadata={'axes': 'ZCYXS'}, check_contrast=False)

    return tiffile_name

def cable_length(vertices, edges, res = [1,1,1]):
    # make sure vertices and res have the same order of zyx
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    if len(edges) == 0:
        return 0
    v1 = vertices[edges[:,0]]
    v2 = vertices[edges[:,1]]

    delta = (v2 - v1) * res
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)

def mAP_out_to_dataframe(input_file, output_file, verbose=True):
    if verbose: print("Parsing {} file to build a dataframe . . .".format(input_file))
    search = open(input_file)
    pred_id = []
    pred_size = []
    iou = []
    gt_id = []
    gt_size = []
    for line in search:
        line = line.strip()
        if line:
            if "#" not in line:
                line = line.split()
                pred_id.append(int(line[0]))
                pred_size.append(int(line[1]))
                iou.append(float(line[4]))
                gt_id.append(int(line[2]))
                gt_size.append(int(line[3]))
                #print("{},{},{},{}".format(gt_id,gt_size,iou,pred_id))

    # Create the dataframe
    data_tuples = list(zip(gt_id,gt_size,pred_id,pred_size,iou))
    df = pd.DataFrame(data_tuples, columns=['gt_id','gt_size','pred_id','pred_size','iou'])
    df = df.sort_values(by=['gt_id','iou'])

    # Drop background id
    indexNames = df[df['gt_id'] == 0].index
    df.drop(indexNames, inplace=True)

    df.to_csv(output_file, index=False)
    if verbose: print("Dataframe stored in {} . . .".format(output_file))

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

