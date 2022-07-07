import os
import h5py
import numpy as np
import pandas as pd
from skimage.io import imread, imsave

def check_files(directory, verbose=False):
    files = sorted(next(os.walk(directory))[2])
    if len(files) == 0:
        raise ValueError("No files found in {}".format(directory))
    if len(files) != 1:
        raise ValueError("Only one file expected. Found {}".format(len(files)))

    f = files[0]
    if not f.endswith('.h5') and not f.endswith('.tif'):
        raise ValueError("Only a .h5 or .tif file is expected. Given {}".format(os.path.join(directory,f)))
    return os.path.join(directory, f)


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

def create_map_aux_file_from_stats(stats_file, out_file, cat=['small','medium','large']):
    """Create an auxiliary file for mAP calculation"""
    aa = open(stats_file)
    bb = aa.readlines()
    result = np.zeros([len(bb)-1,2])

    # Create categories
    cat_codes = {}
    for i, c in enumerate(cat):
        cat_codes[c] = i
    
    for i in range(1,len(bb)):
        line = bb[i].replace('\n','').split(',')
        result[i-1, 0] = int(line[0])
        result[i-1, 1] = cat_codes[line[-1]]
    aa.close()
    np.savetxt(out_file, result, '%d')

def str_list_to_ints_list(df, col_name):
    """Return a list converting strings to list of ints. This happens when creating 
       the dataframe from dictionary containing lists."""
    list = df[col_name]
    new_list = []
    for l in list:
        new_l = l.replace(']','').replace('[','').split(", ")
        if new_l[0] == '':
            new_list.append([-1])
        else:
            new_list.append([int(a) for a in new_l])
    return new_list

def create_map_aux_file_from_associations(pred_stats_file, gt_stats_file, association_file,  
    out_file, cat=['small','medium','large']):
    """Create an auxiliary file for mAP calculation based on gt categories using the association info"""
    df_assoc = pd.read_csv(association_file, index_col=False)
    df_pred = pd.read_csv(pred_stats_file, index_col=False)
    df_gt = pd.read_csv(gt_stats_file, index_col=False)

    # Change columns from str to list of ints
    df_assoc['predicted'] = str_list_to_ints_list(df_assoc, 'predicted')
    df_assoc['gt'] = str_list_to_ints_list(df_assoc, 'gt')

    # Create categories  
    cat_codes = {}
    for i, c in enumerate(cat):
        cat_codes[c] = i

    pred_instances = df_pred['label'].tolist()
    result = np.zeros([len(pred_instances),2])
    
    # Capture prediction instances categories looking the instance 
    # they are associated with in the gt
    for i, pred_ins in enumerate(pred_instances):
        query = []
        for l in df_assoc['predicted']:
            if pred_ins in l:
                query.append(True)
            else:
                query.append(False)
        line = df_assoc[query]

        if line.size == 0:
            c = df_pred[df_pred['label']==pred_ins]['category'].iloc[0]
            pred_category = cat_codes[c]
        else:
            gt_instances = line['gt'].iloc[0]
            pred_category = 0
            for gt_ins in gt_instances:
                c = df_gt[df_gt['label'] == gt_ins]['category'].iloc[0]
                if cat_codes[c] > pred_category:
                    pred_category = cat_codes[c]

        result[i, 0] = pred_ins
        result[i, 1] = pred_category
    np.savetxt(out_file, result, '%d')
