"""Adapted from https://github.com/stardist/stardist"""
import os
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import namedtuple
import pandas as pd
from prettytable import PrettyTable

def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)

def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))

def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True

def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x,'x',True)
        _check_label_array(y,'y',True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)

def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _safe_divide(x,y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out

def intersection_over_union(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))

def precision(tp,fp,fn):
    return tp/(tp+fp) if tp > 0 else 0
def recall(tp,fp,fn):
    return tp/(tp+fn) if tp > 0 else 0
def accuracy(tp,fp,fn):
    return tp/(tp+fp+fn) if tp > 0 else 0
def f1(tp,fp,fn):
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0

def calculate_matching_metrics(gt_file, pred_file, out_file, precomputed_matching_file=None, gt_stats_file=None, 
                               pred_stats_file=None, thresh=0.5, report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.
       Currently, the following metrics are implemented:
       'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 
       'mean_matched_score', 'panoptic_quality'. Corresponding objects of y_true and y_pred are counted as true 
       positives (tp), false positives (fp), and false negatives (fn) whether their intersection over union (IoU) >= thresh 
           * mean_matched_score is the mean IoUs of matched true positives
           * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects
           * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

       Parameters
       ----------
       pred_file : str
           Path to the prediction file. 

       gt_file : str
           Path to the ground truth file. 

       out_file : str
           Path to store the output csv file. 
           
       precomputed_matching_file : str, optional
           Path to a previously computed matching csv file.
           
       gt_stats_file : str, optional
           Path to the ground truth statistics file. Needed when 'precomputed_matching_file' is not None. 
                  
       pred_stats_file : str, optional
           Path to the prediction statistics file. Needed when 'precomputed_matching_file' is not None. 

       thresh: float
           threshold for matching criterion (default 0.5)

       report_matches: bool
           if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')
    """
    # Not make again the matching if previously done
    if not os.path.exists(precomputed_matching_file):
        # Load pred
        if str(pred_file).endswith('.h5'):
            h5f = h5py.File(pred_file, 'r')
            k = list(h5f.keys())
            y_pred = np.array(h5f[k[0]])
            del h5f, k
        else:
            y_pred = imread(pred_file)

        # Load gt
        if str(gt_file).endswith('.h5'):
            h5f = h5py.File(gt_file, 'r')
            k = list(h5f.keys())
            y_true = np.array(h5f[k[0]])
            del h5f, k
        else:
            y_true = imread(gt_file)

        # Checks
        _check_label_array(y_true,'y_true')
        _check_label_array(y_pred,'y_pred')
        y_true.shape == y_pred.shape or _raise(ValueError("y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true, y_pred=y_pred)))
        if thresh is None: thresh = 0
        thresh = float(thresh) if np.isscalar(thresh) else map(float,thresh)

        y_true, _, map_rev_true = relabel_sequential(y_true)
        y_pred, _, map_rev_pred = relabel_sequential(y_pred)

        overlap = label_overlap(y_true, y_pred, check=False)
        scores = intersection_over_union(overlap)
    else:
        if gt_stats_file is None or pred_stats_file is None:
            _raise(ValueError("'gt_stats_file' and 'pred_stats_file' need to be provided when 'precomputed_matching_file' in not None"))
        
        print("Using matching previously calculated with mAP . . .")
        df = pd.read_csv(precomputed_matching_file, index_col=False)
        df = df.sort_values(by=['gt_id'], ascending=True)

        df_gt = pd.read_csv(gt_stats_file, index_col=False)
        df_gt = df_gt.sort_values(by=['label'], ascending=True)

        df_pred = pd.read_csv(pred_stats_file, index_col=False)
        df_pred = df_pred.sort_values(by=['label'], ascending=True)
        
        l_true = df_gt['label'].tolist()
        l_pred = df_pred['label'].tolist()

        # Relabel instances and map them to run it faster
        gt_mapping = {}
        c = 1
        last_number = l_true[0]
        gt_mapping[l_true[0]] = c
        l_true[0] = c     
        for i in range(1,len(l_true)):
            if last_number != l_true[i]:
                c += 1
            gt_mapping[l_true[i]] = c
            l_true[i] = c   

        # Relabel instances and map them to run it faster
        pred_mapping = {}
        c = 1
        last_number = l_pred[0]
        pred_mapping[l_pred[0]] = c
        l_pred[0] = c
        for i in range(1,len(l_pred)):
            if last_number != l_pred[i]:
                c += 1
            pred_mapping[l_pred[i]] = c
            l_pred[i] = c

        scores = np.zeros((len(l_true)+1, len(l_pred)+1), dtype=np.float32)

        # Calculate the scores
        df = df.reset_index()
        for index, row in df.iterrows():
            i = int(row['gt_id'])
            pi = int(row['pred_id'])
            scores[gt_mapping[i], pred_mapping[pi] ] = float(row['iou'])

    assert 0 <= np.min(scores) <= np.max(scores) <= 1
    scores = scores[1:,1:]  # ignoring background
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind,pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score    = _safe_divide(sum_matched_score, n_true)
        panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

        stats_dict = dict (
            thresh             = thr,
            fp                 = fp,
            tp                 = tp,
            fn                 = fn,
            precision          = precision(tp,fp,fn),
            recall             = recall(tp,fp,fn),
            accuracy           = accuracy(tp,fp,fn),
            f1                 = f1(tp,fp,fn),
            n_true             = n_true,
            n_pred             = n_pred,
            mean_true_score    = mean_true_score,
            mean_matched_score = mean_matched_score,
            panoptic_quality   = panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update (
                    # int() to be json serializable
                    matched_pairs  = tuple((int(map_rev_true[i]),int(map_rev_pred[j])) for i,j in zip(1+true_ind,1+pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return namedtuple('Matching',stats_dict.keys())(*stats_dict.values())

    if np.isscalar(thresh):
        value = _single(thresh)
        out_results = pd.DataFrame.from_dict(value)  
    else:
        value = tuple(map(_single,thresh))  
        out_results = pd.DataFrame.from_dict([v for v in value])   

    out_results.to_csv(out_file, index=False)


def print_matching_stats(stats_csv, show_categories=False):
    """Print matching statistics.

       Parameters
       ----------
       stats_csv : str
           Path where the statistics of the matching are stored.
        
       show_categories : bool, optional
           Not used (added just for convention).
    """
    df = pd.read_csv(stats_csv, index_col=False)
    df = df.round(3)
    
    t = PrettyTable(df.columns.tolist()[1:])
    df = df.reset_index()
    for index, row in df.iterrows():
        t.add_row(row.tolist()[2:])
    
    txt = "Matching metrics"
    txt = txt.center(t.get_string().find(os.linesep))
    print(txt)
    print(t)


# copied from scikit-image master for now (remove when part of a release)
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.
    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).
    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.
    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.
    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.
    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.
    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max()) # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map
    
    
    