from bdb import set_trace
import os
from tkinter.tix import ROW
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

def mAP_out_arrays_to_dataframes(pred_arr, missing_arr, matching_out_file, out_assoc_file, verbose=True):
    gt_ids, gt_sizes, pred_ids, pred_sizes, ious = [], [], [], [], []

    gt_to_pred = {}
    gt_keys = {}
    pred_to_gt = {}
    for i in range(pred_arr.shape[0]):
        row = pred_arr[i]
        pred_id = int(row[0])
        gt_id = int(row[2])

        # Matching stats
        gt_ids.append(gt_id)
        gt_sizes.append(int(row[3]))
        pred_ids.append(pred_id)
        pred_sizes.append(int(row[1]))
        ious.append(float(row[4]))

        if gt_id != 0:
            # 'over-segmentation'
            if gt_id in gt_to_pred:
                gt_to_pred[gt_id].append(pred_id)
            # 'one-to-one'
            else:
                gt_to_pred[gt_id] = [pred_id]
            pred_to_gt[pred_id] = [gt_id]

    # Save matching stats
    data_tuples = list(zip(gt_ids,gt_sizes,pred_ids,pred_sizes,ious))
    df = pd.DataFrame(data_tuples, columns=['gt_id','gt_size','pred_id','pred_size','iou'])
    indexNames = df[df['gt_id'] == 0].index # Drop background id
    df.drop(indexNames, inplace=True)
    df = df.sort_values(by=['gt_id','iou'])
    df.to_csv(matching_out_file, index=False)
    if verbose: print("Matching dataframe stored in {} . . .".format(matching_out_file))

    for i in range(missing_arr.shape[0]):
        row = missing_arr[i]
        gt_id = int(row[2])
        iou = float(row[4])

        # 'missing'
        if iou == 0:
            gt_to_pred[gt_id] = []
        # 'under-segmentation' and 'many-to-many'
        else:
            pred_id = int(row[0])

            # Obtain the instance in the gt that matches it
            gt_match_inst = pred_to_gt[pred_id]
            old_key = gt_match_inst[0]
        
            # Insert the relation if it is not registered yet
            if not gt_id in gt_keys:
                gt_keys[gt_id] = str([gt_id])

            # Build the new dict key and update the old_key if it was already changed
            if old_key in gt_keys:
                inst_to_update = str(gt_keys[gt_id] + gt_keys[old_key])
                old_key = gt_keys[old_key]
            else:
                inst_to_update = str(gt_keys[gt_id] + str(gt_match_inst))
            inst_to_update = inst_to_update.replace('[',' ').replace(']',' ').replace(',','').split()
            inst_to_update = [int(x) for x in inst_to_update]
            new_dic_key = str(inst_to_update)

            # Update all the instances with the new dict key for future matchings
            for ins in inst_to_update:
                gt_keys[ins] = str(inst_to_update)

            # Update the old entry with the new one
            gt_to_pred[new_dic_key] = gt_to_pred.pop(old_key)

    # Save associations
    data_tuples = list( zip( gt_to_pred.values(), gt_to_pred.keys() ) )
    df_assoc = pd.DataFrame(data_tuples, columns=['predicted', 'gt'])
    df_assoc.to_csv(out_assoc_file, index=False)
    if verbose: print("Association dataframe stored in {} . . .".format(out_assoc_file))


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

def str_list_to_ints_list(df, col_name, void_to_number=True):
    """Return a list converting strings to list of ints. This happens when creating 
       the dataframe from dictionary containing lists."""
    list = df[col_name]
    new_list = []
    for l in list:
        new_l = l.replace(']','').replace('[','').split(", ")
        if new_l[0] == '':
            if void_to_number:
                new_list.append([-1])
            else:
                new_list.append([])
        else:
            new_list.append([int(a) for a in new_l])
    return new_list

def create_map_groups_from_associations(map_aux_dir, gt_stats_file, association_file, out_file, 
        cat=['small','medium','large'], verbose=True):
    """Create an auxiliary file for mAP calculation based on gt categories using the association info"""
    df_assoc = pd.read_csv(association_file, index_col=False)
    pred_instances = np.load(os.path.join(map_aux_dir, "pred_labels.npy")).tolist()
    df_gt = pd.read_csv(gt_stats_file, index_col=False)
    
    # Change columns from str to list of ints
    df_assoc['predicted'] = str_list_to_ints_list(df_assoc, 'predicted')
    df_assoc['gt'] = str_list_to_ints_list(df_assoc, 'gt')

    # Create categories  
    cat_codes = {}
    for i, c in enumerate(cat):
        cat_codes[c] = i

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
            # For the FP leave them as the first category, e.g. small
            pred_category = cat_codes[cat[0]]
        else:
            gt_instances = line['gt'].iloc[0]
            pred_category = 0
            for gt_ins in gt_instances:
                c = df_gt[df_gt['label'] == gt_ins]['category'].iloc[0]
                if cat_codes[c] > pred_category:
                    pred_category = cat_codes[c]

        result[i, 0] = pred_ins
        result[i, 1] = pred_category
    if verbose: print("MAP group file created in {} . . .".format(out_file))
    np.savetxt(out_file, result, '%d')
