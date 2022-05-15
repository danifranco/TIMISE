import os
import pandas as pd
import numpy as np
import networkx as nx
from timagetk.components.labelled_image import LabelledImage
from skimage.io import imread
from ctrl.image_overlap import fast_image_overlap3d
from prettytable import PrettyTable


def calculate_associations(pred_file, gt_file, out_dir, verbose=True):
    """Calculate associations between instances. Based on the work presented in `Assessment of deep learning
       algorithms for 3D instance segmentation of confocal image datasets <https://www.biorxiv.org/content/10.1101/2021.06.09.447748v1.full>`_.
       Code `here <https://mosaic.gitlabpages.inria.fr/publications/seg_compare/evaluation.html>`_.

       Parameters
       ----------
       pred_file : str
           Path to the file containing the predicted instances.

       pred_file : str
           Path to the file containing the ground truth instances.

       out_dir : str
           Path to the directory where auxiliary matching statistics will be placed.

       verbose : bool, optional
           Wheter to be more verbose.
    """

    # Calculating the matching between instances
    pred_matching_file = os.path.join(out_dir, "target_mother_matching_file.csv")
    gt_matching_file = os.path.join(out_dir, "target_daughter_matching_file.csv")
    if not os.path.exists(pred_matching_file) or not os.path.exists(gt_matching_file):
        if verbose:
            print("Associations between images {} and {}".format(pred_file,gt_file))
            print("Image loading . . .")
        pred_img = LabelledImage(imread(pred_file), no_label_id=0)
        gt_img = LabelledImage(imread(gt_file), no_label_id=0)

        if verbose: print("Calculation of matching between instances . . .")
        df_pred = fast_image_overlap3d(pred_img, gt_img, method='target_mother', ds=1, verbose=verbose)
        df_pred.columns = ['pred_id', 'gt_id', 'iou']
        df_gt = fast_image_overlap3d(pred_img, gt_img, method='target_daughter', ds=1, verbose=verbose)
        df_gt.columns = ['pred_id', 'gt_id', 'iou']

        del pred_img,gt_img
        # Save matching files
        df_pred.to_csv(pred_matching_file)
        df_gt.to_csv(gt_matching_file)
    else:
        df_pred = pd.read_csv(pred_matching_file, index_col=False)
        df_gt = pd.read_csv(gt_matching_file, index_col=False)

    # Delete matching with 0 IoU
    df_pred = df_pred.loc[~(df_pred.iou == 0)].copy()
    df_gt = df_gt.loc[~(df_gt.iou == 0)].copy()
    all_gt_labels_processed = np.unique(np.array(df_gt['gt_id'].tolist()))

    # Associate each prediction/gt instances in which it is the most included
    df_pred = df_pred.loc[df_pred.groupby('pred_id')['iou'].idxmax()]
    df_gt = df_gt.loc[df_gt.groupby('gt_id')['iou'].idxmax()]

    # Convert in dict
    pred_in_gt = df_pred[['pred_id', 'gt_id']].set_index('pred_id').to_dict()['gt_id']
    gt_in_pred = df_gt[['gt_id', 'pred_id']].set_index('gt_id').to_dict()['pred_id']
    pred_labels = list(set(df_pred.pred_id.values) | set(df_gt.pred_id.values))
    gt_labels = list(set(df_pred.gt_id.values) | set(df_gt.gt_id.values))
    label_tp_list = [(m, 'l') for m in pred_labels] + [(d, 'r') for d in gt_labels]
    lg2nid = dict(zip(label_tp_list, range(len(label_tp_list))))

    # Create the graph
    G = nx.Graph()
    G.add_nodes_from([(nid, {'label': lab, 'group': g}) for (lab, g), nid in lg2nid.items()])
    pred_to_gt_list = [(lg2nid[(i, 'l')], lg2nid[(j, 'r')]) for i, j in pred_in_gt.items()]
    G.add_edges_from(pred_to_gt_list)
    gt_to_pred_list = [(lg2nid[(i, 'r')], lg2nid[(j, 'l')]) for i, j in gt_in_pred.items()]
    G.add_edges_from(gt_to_pred_list)

    # Overlap analysis: get the predicted instances <--> ground truth instances from the connected subgraph in G
    connected_graph = [list(G.subgraph(c)) for c in nx.connected_components(G)]

    # Gather all the connected subgraph and reindex according to the image labels
    nid2lg = {v: k for k, v in lg2nid.items()}
    out_results = []
    for c in connected_graph:
        if len(c) > 1:  # at least two labels
            preds, gt = [], []
            for nid in c:
                if nid2lg[nid][1] == 'l':
                    preds.append(nid2lg[nid][0])
                else:
                    gt.append(nid2lg[nid][0])
            out_results.append({'predicted': preds, 'gt': gt})

    if verbose: print("Calculating rest of missing instances not present in associations . . .")
    img = imread(gt_file)
    labels = np.unique(img)[1:] # remove background
    total_instances = len(labels)
    new_gt_labels_processed = []
    for instance in labels:
        if instance not in all_gt_labels_processed:
            out_results.append({'predicted': [], 'gt': [instance]})
            new_gt_labels_processed.append(instance)
    all_gt_labels_processed = all_gt_labels_processed.tolist() + new_gt_labels_processed
    if verbose: print("Total gt labels: {} (processed {})".format(total_instances, len(all_gt_labels_processed)))

    out_results = pd.DataFrame(out_results)
    out_results['association_type'] = out_results.apply(lambda row: lab_association(row), axis=1)

    # Creating association statistics to not do it everytime the user wants to print them
    cell_statistics = {'one-to-one': 0, 'missing': 0, 'over-segmentation': 0,
                      'under-segmentation': 0, 'many-to-many': 0}
    out_results = out_results.reset_index()
    for index, row in out_results.iterrows():
        for gt_instances in row['gt']:
            cell_statistics[row['association_type']] += 1
    df_out = pd.DataFrame.from_dict({k:[v] for k,v in cell_statistics.items()})

    # Saving dataframes
    df_out.to_csv(os.path.join(out_dir, "associations_stats.csv"))
    out_results.to_csv(os.path.join(out_dir, "associations.csv"))


def lab_association(row):
    """Determines the association type."""

    if len(row['predicted']) == 0:
        return 'missing'
    elif len(row['predicted']) == 1:
        if len(row['gt']) == 1:
            return 'one-to-one'
        else:
            return 'under-segmentation'
    else:
        if len(row['gt']) == 1:
            return 'over-segmentation'
        else:
            return 'many-to-many'


def print_association_stats(stats_csv):
    """Print association statistics."""

    if not os.path.exists(stats_csv):
        raise ValueError('File {} not found. Did you call TIMISE.evaluate()?'.format(stats_csv))

    df_out = pd.read_csv(stats_csv, index_col=False)
    cell_statistics = {'one-to-one': df_out['one-to-one'][0],
                       'missing': df_out['missing'][0],
                       'over-segmentation': df_out['over-segmentation'][0],
                       'under-segmentation': df_out['under-segmentation'][0],
                       'many-to-many': df_out['many-to-many'][0]}
    total_instances = cell_statistics['one-to-one']+cell_statistics['missing'] \
                      +cell_statistics['over-segmentation']+cell_statistics['under-segmentation'] \
                      +cell_statistics['many-to-many']

    t = PrettyTable([' ',]+list(cell_statistics.keys())+['Total'])
    t.add_row(['Count',]+list(cell_statistics.values())+[total_instances])

    # Percentages
    cell_statistics = {state: np.around((val/total_instances)*100, 2) for state, val in cell_statistics.items()}

    t.add_row(['%',]+list(cell_statistics.values())+[' '])
    print("                                         Associations                                         ")
    print(t)
