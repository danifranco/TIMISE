#!/usr/bin/env python
# coding: utf-8

"""
This script allows you to obtain gt instance and prediction instance matches for the 3D mAP model evaluation. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True (default).
"""

import time
import os, sys
import argparse
import numpy as np
import h5py
from prettytable import PrettyTable

from .vol3d_eval import VOL3Deval
from .vol3d_util import seg_iou3d_sorted, readh5_handle, readh5, unique_chunk


def load_data(args, slices):
    # load data arguments
    pred_seg = readh5_handle(args.predict_seg)
    gt_seg = readh5_handle(args.gt_seg)
    if slices[1] == -1:
        slices[1] = gt_seg.shape[0]
    pred_bbox, gt_bbox = None, None
    if args.predict_bbox != '':
        pred_bbox = np.loadtxt(args.predict_bbox).astype(int)
    if args.gt_bbox != '':
        gt_bbox = np.loadtxt(args.gt_bbox).astype(int)

    # check shape match
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt-sz_pred)).max()>0:
        raise ValueError('Warning: size mismatch. gt: {}, pred: '.format(sz_gt,sz_pred))

    if args.predict_score != '':
        print('\t\t Load prediction score')
        # Nx2: pred_id, pred_sc
        if '.h5' in args.predict_score:
            pred_score = readh5(args.predict_score)
        elif '.txt' in args.predict_score:
            pred_score = np.loadtxt(args.predict_score)
        else:
            raise ValueError('Unknown file format for the prediction score')

        if not np.any(np.array(pred_score.shape)==2):
            raise ValueError('The prediction score should be a Nx2 array')
        if pred_score.shape[1] != 2:
            pred_score = pred_score.T
    else: # default
        print('\t\t Assign prediction score')
        # assign same weight
        """
        ui = unique_chunk(pred_seg, slices, chunk_size = args.chunk_size, do_count = False)
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        """
        # alternative: sort by size
        ui,uc = unique_chunk(pred_seg, slices, chunk_size = args.chunk_size)
        uc = uc[ui>0]
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        pred_score[:,1] = uc

    th_group, areaRng = np.zeros(0), np.zeros(0)
    group_gt, group_pred = None, None
    if args.group_gt != '': # exist gt group file
        group_gt = np.loadtxt(args.group_gt).astype(int)
        group_pred = np.loadtxt(args.group_pred).astype(int)
    else:
        thres = np.fromstring(args.threshold, sep = ",")
        areaRng = np.zeros((len(thres)+2,2),int)
        areaRng[0,1] = 1e10
        areaRng[-1,1] = 1e10
        areaRng[2:,0] = thres
        areaRng[1:-1,1] = thres

    return gt_seg, pred_seg, pred_score, group_gt, group_pred, areaRng, slices, gt_bbox, pred_bbox


def mAP_computation(_args):
    """
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset.
    """
    args = _args

    ## 1. Load data
    start_time = int(round(time.time() * 1000))
    if args.verbose: print('\t1. Load data')

    def _return_slices():
        # check if args.slices is well defined and return slices array [slice1, sliceN]
        if str(args.slices) == "-1":
            slices = [0, -1]
        else: # load specific slices only
            try:
                slices = np.fromstring(args.slices, sep = ",", dtype=int)
                 #test only 2 boundaries, boundary1<boundary2, and boundaries positive
                if (slices.shape[0] != 2) or \
                    slices[0] > slices[1] or \
                    slices[0] < 0 or slices[1] < 0:
                    raise ValueError("\nspecify a valid slice range, ex: -sl '50, 350'\n")
            except:
                print("\nplease specify a valid slice range, ex: -sl '50, 350'\n")
        return slices

    slices = _return_slices()

    gt_seg, pred_seg, pred_score, group_gt, group_pred, areaRng, slices, gt_bbox, pred_bbox = load_data(args, slices)
    ## 2. create complete mapping of ids for gt and pred:
    if args.verbose: print('\t2. Compute IoU')
    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, slices, group_gt, areaRng, args.chunk_size, args.threshold_crumb, pred_bbox, gt_bbox)
    stop_time = int(round(time.time() * 1000))
    if args.verbose: print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )

    ## 3. Evaluation script for 3D instance segmentation
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name=args.output_name, verbose=args.verbose)
    if args.do_txt > 0:
        v3dEval.save_match_p()
        v3dEval.save_match_fn()
    if args.do_eval > 0:
        if args.verbose: print('start evaluation')
        #Evaluation
        v3dEval.set_group(group_gt, group_pred)
        v3dEval.params.areaRng = areaRng
        v3dEval.accumulate()
        v3dEval.summarize()

def print_mAP_stats(stats_file):
    """Print mAP statistics."""

    if not os.path.exists(stats_file):
        raise ValueError('File {} not found. Did you call TIMISE.evaluate()?'.format(stats_file))

    search = open(stats_file)
    values = np.zeros((4))
    for line in search:
        line = line.strip()
        if line:
            line = line.split()
            if "IoU=0.50:0.95" in line:
                values[0] = float(line[-1])
            elif "IoU=0.50" in line and 'all' in line:
                values[1] = float(line[-1])
            elif "IoU=0.75" in line and 'all' in line:
                values[2] = float(line[-1])
            elif "IoU=0.90" in line and 'all' in line:
                values[3] = float(line[-1])

    columns = ['IoU=0.50:0.95', 'IoU=0.50', 'IoU=0.75', 'IoU=0.90']
    t = PrettyTable(columns)
    t.add_row(values.tolist())

    txt = "Average Precision (AP)"
    txt = txt.center(t.get_string().find(os.linesep))
    print(txt)
    print(t)

