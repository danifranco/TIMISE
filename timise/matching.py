"""Adapted from https://github.com/stardist/stardist"""
import os
import numpy as np
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

def calculate_matching_metrics(out_file, pred_group_file, categories=None, precomputed_matching_file=None, gt_stats_file=None, 
                               thresh=0.5, report_matches=False):
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
       out_file : str
           Path to store the output csv file. 

       categories: List of str     
           Categories the instances are divided on.

       precomputed_matching_file : str, optional
           Path to a previously computed matching csv file.
           
       gt_stats_file : str, optional
           Path to the ground truth statistics file. Needed when 'precomputed_matching_file' is not None. 

       thresh: float
           threshold for matching criterion (default 0.5)

       report_matches: bool
           if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')
    """
    if gt_stats_file is None:
        _raise(ValueError("'gt_stats_file' need to be provided"))
    
    print("Using matching previously calculated with mAP . . .")
    df = pd.read_csv(precomputed_matching_file, index_col=False)
    df = df.sort_values(by=['gt_id'], ascending=True)

    df_gt = pd.read_csv(gt_stats_file, index_col=False)
    #df_gt = df_gt.sort_values(by=['label'], ascending=True)

    df_pred = pd.read_csv(pred_group_file, names=['pred_id', 'category'], index_col=False, sep=' ')

    final_values = []
    _categories = categories.copy()
    _categories.append('total')
    for k, cat in enumerate(_categories):
        if cat != 'total':
            l_true = df_gt[df_gt['category'] == cat]['label'].tolist()
            l_pred = df[df['gt_id'].isin(l_true)]['pred_id'].tolist()
            df_aux = df[df['gt_id'].isin(l_true)]

            # Take the FPs
            df_pred_aux = df_pred[df_pred['category'] == k]['pred_id']
            l_pred += df_pred_aux[~(df_pred_aux.isin(l_pred))].tolist()
        else:
            l_true = df_gt['label'].tolist()
            l_pred = df_pred['pred_id'].tolist()
            df_aux = df
        
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
        df_aux = df_aux.reset_index()
        for index, row in df_aux.iterrows():
            i = int(row['gt_id'])
            pi = int(row['pred_id'])
            scores[gt_mapping[i], pred_mapping[pi] ] = float(row['iou'])
        
        assert 0 <= np.min(scores) <= np.max(scores) <= 1
        scores = scores[1:,1:]  # ignoring background
        n_true, n_pred = scores.shape
        n_matched = min(n_true, n_pred)

        map_rev_true = gt_mapping.keys()
        map_rev_pred = pred_mapping.keys()

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
                category           = cat, 
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
            return namedtuple('Matching', stats_dict.keys())(*stats_dict.values())

        if np.isscalar(thresh):
            final_values.append(_single(thresh))
        else:
            value = tuple(map(_single,thresh))
            for v in value:
                final_values.append(v)

    out_results = pd.DataFrame.from_dict([v for v in final_values])   
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
    
    t = PrettyTable(df.columns.tolist())
    df = df.reset_index()
    for index, row in df.iterrows():
        t.add_row(row.tolist()[1:])
    
    txt = "Matching metrics"
    txt = txt.center(t.get_string().find(os.linesep))
    print(txt)
    print(t)