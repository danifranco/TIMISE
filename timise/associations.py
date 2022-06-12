import os
import statistics
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from timagetk.components.labelled_image import LabelledImage
from skimage.io import imread
from ctrl.image_overlap import fast_image_overlap3d
from prettytable import PrettyTable
import plotly.express as px


def calculate_associations(pred_file, gt_file, gt_stats_file, final_file, verbose=True):
    """Calculate associations between instances. Based on the work presented in `Assessment of deep learning
       algorithms for 3D instance segmentation of confocal image datasets 
       <https://www.biorxiv.org/content/10.1101/2021.06.09.447748v1.full>`_.
       Code `here <https://mosaic.gitlabpages.inria.fr/publications/seg_compare/evaluation.html>`_.

       Parameters
       ----------
       pred_file : str
           Path to the file containing the predicted instances.

       gt_file : str
           Path to the file containing the ground truth instances.

       gt_stats_file : str
           Path to the prediction statistics.

       final_file : str
           Path where the final error file will be stored.

       verbose : bool, optional
           Wheter to be more verbose.
    """

    # Calculating the matching between instances
    out_dir = os.path.dirname(final_file)
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
        if verbose: print("Association matching seems to be computed. Loading from file: {}".format(pred_matching_file))
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
    # and modifying the predictions stats to insert information about the association
    gt_stats = pd.read_csv(gt_stats_file, index_col=False)
    gt_stats['counter'] = 0
    gt_stats['association_counter'] = 0
    gt_stats['association_type'] = 'over-segmentation'
    _labels = np.array(gt_stats['label'].tolist())
    _counter = np.array(gt_stats['counter'].tolist())
    _association_counter = np.array(gt_stats['association_counter'].tolist(), dtype=np.float32)
    _association_type = np.array(gt_stats['association_type'].tolist())
    if 'category' in gt_stats.columns:
        cell_statistics = []
        categories = pd.unique(gt_stats['category']).tolist()
        for i in range(len(categories)):
            cell_statistics.append({'one-to-one': 0, 'missing': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'many-to-many': 0})
    else:
        cell_statistics = {'one-to-one': 0, 'missing': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'many-to-many': 0}
    out_results = out_results.reset_index()
    for index, row in out_results.iterrows():
        gt_instances = row['gt']
        for gt_ins in gt_instances:
            if type(cell_statistics) is list:
                result = gt_stats[gt_stats['label']==gt_ins]
                tag = result['category'].iloc[0]
                cell_statistics[categories.index(tag)][row['association_type']] += 1
            else:
                cell_statistics[row['association_type']] += 1
        if row['association_type'] == 'over-segmentation':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_counter[itemindex] = len(row['predicted'])
            _association_type[itemindex] = 'over'
        elif row['association_type'] == 'under-segmentation':
            for ins in gt_instances:
                itemindex = np.where(_labels==ins)
                _counter[itemindex] += 1
                _association_counter[itemindex] = -len(gt_instances)
                _association_type[itemindex] = 'under'
        elif row['association_type'] == 'many-to-many':
            pred_count = len(row['predicted'])
            if pred_count >= len(gt_instances):
                val = pred_count/len(gt_instances)
                t = 'over'
            else:
                val = -pred_count/len(gt_instances)
                t = 'under'
            for ins in gt_instances:
                itemindex = np.where(_labels==ins)
                _counter[itemindex] += 1
                _association_counter[itemindex] = val
                _association_type[itemindex] = t
        elif row['association_type'] == 'one-to-one':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'one-to-one'
        elif row['association_type'] == 'missing':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'missing'
        else:
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'other'

    gt_stats['counter'] = _counter
    gt_stats['association_counter'] = _association_counter
    gt_stats['association_type'] = _association_type
    if type(cell_statistics) is list:
        df_out = pd.DataFrame.from_dict([ {k:[v] for k,v in a.items()} for a in cell_statistics] )
        df_out = df_out.set_axis(categories)
    else:
        df_out = pd.DataFrame.from_dict([{k:[v] for k,v in cell_statistics.items()}])
        df_out = df_out.set_axis(['all'])

    # Saving dataframes
    gt_stats.to_csv(final_file)
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


def print_association_stats(stats_csv, show_categories=False):
    """Print association statistics.

       Parameters
       ----------
       stats_csv : str
           Path where the statistics of the associations are stored.
        
       show_categories : bool, optional
           Whether to print one row per category or just all the instances together. 
    """

    if not os.path.exists(stats_csv):
        raise ValueError('File {} not found. Did you call TIMISE.evaluate()?'.format(stats_csv))

    df_out = pd.read_csv(stats_csv, index_col=0)
    total_instances_all = 0
    df_len = df_out.shape[0]

    # If more than one line found means that the user selected to split into categories 
    if df_len > 1:
        max_str_size = 0
        total_assoc = [0, 0, 0, 0, 0]

    for i in range(df_len):
        total_instances = 0
        cell_statistics = {}
        if isinstance(df_out['one-to-one'][i], str):
            cell_statistics['one-to-one'] = int(df_out['one-to-one'][i][1:][:-1])
        else:
            cell_statistics['one-to-one'] = df_out['one-to-one'][i]
        if isinstance(df_out['missing'][i], str):
            cell_statistics['missing'] = int(df_out['missing'][i][1:][:-1])
        else:
            cell_statistics['missing'] = df_out['missing'][i]
        if isinstance(df_out['over-segmentation'][i], str):
            cell_statistics['over-segmentation'] = int(df_out['over-segmentation'][i][1:][:-1])
        else:
            cell_statistics['over-segmentation'] = df_out['over-segmentation'][i]
        if isinstance(df_out['under-segmentation'][i], str):
            cell_statistics['under-segmentation'] = int(df_out['under-segmentation'][i][1:][:-1])
        else:
            cell_statistics['under-segmentation'] = df_out['under-segmentation'][i]
        if isinstance(df_out['many-to-many'][i], str):
            cell_statistics['many-to-many'] = int(df_out['many-to-many'][i][1:][:-1])
        else:
            cell_statistics['many-to-many'] = df_out['many-to-many'][i]
        total_instances += cell_statistics['one-to-one']+cell_statistics['missing'] \
                           +cell_statistics['over-segmentation']+cell_statistics['under-segmentation'] \
                           +cell_statistics['many-to-many']
        total_instances_all += total_instances
        if i == 0:
            extra_column = ['category',] if df_len > 1 else []
            t = PrettyTable(extra_column+[' ',]+list(cell_statistics.keys())+['Total'])
        extra_column = [df_out.iloc[i].name,] if df_len > 1 else []
        t.add_row(extra_column+['Count',]+list(cell_statistics.values())+[total_instances])

        if df_len > 1 :
            total_assoc[0] += cell_statistics['one-to-one']
            total_assoc[1] += cell_statistics['missing']
            total_assoc[2] += cell_statistics['over-segmentation']
            total_assoc[3] += cell_statistics['under-segmentation']
            total_assoc[4] += cell_statistics['many-to-many']

            if max_str_size < len(str(df_out.iloc[i].name)):
                max_str_size = len(str(df_out.iloc[i].name))+3

        # Percentages
        cell_statistics = {state: np.around((val/total_instances)*100, 2) for state, val in cell_statistics.items()}
        extra_column = [' ',] if df_len > 1 else []
        t.add_row(extra_column+['%',]+list(cell_statistics.values())+[' '])

    if df_len > 1 and show_categories:
        t.add_row(['',]*(3+len(cell_statistics.values()) ))
        t.add_row(['TOTAL','Count',]+total_assoc+[total_instances_all,])
        t.add_row([' ','%',]+[np.around((val/total_instances_all)*100, 2) for val in total_assoc]+[total_instances_all,])
        more_space = ' '*int(max_str_size/2)
    else:
        t = PrettyTable([' ',]+list(cell_statistics.keys())+['Total'])
        t.add_row(['Count',]+total_assoc+[total_instances_all,])
        t.add_row(['%',]+[np.around((val/total_instances_all)*100, 2) for val in total_assoc]+[total_instances_all,])
        more_space = ''

    print(more_space+"                                         Associations                                         "+more_space)
    print(t)


def association_plot_2d(final_file, save_path, show=True, bins=30, draw_std=True, log_x=False, log_y=False, shape=[1100,500]):
    """Plot 2D errors.
    
       Parameters
       ----------
       final_file : str
           Path to the final statistics file.

       save_path : str
           Directory to store the plot into. 

       show : bool, optional
           Wheter to show or not the plot after saving. 

       bins : int, optional
           Defines the number of equal-width bins to create when creating the plot. 
    
       draw_std : bool, optional
           Whether to draw or not standar deviation of y axis.

       log_x : bool, optional
           True to apply log in 'x' axis. 

       log_y : bool, optional
           True to apply log in 'y' axis. 

       shape : 2d array of ints, optional
           Defines the shape of the plot.
    """

    df = pd.read_csv(final_file, index_col=False)
    
    X = np.array(df['cable_length'].tolist(), dtype=float).tolist()
    Z = np.array(df['association_counter'].tolist(), dtype=float).tolist()

    ret_x, bin_edges, binnumber = stats.binned_statistic(X, X, 'mean', bins=bins)
    size = stats.binned_statistic(X, X, 'count', bins=bins).statistic
    df['binnumber'] = binnumber
    ret_y = []
    std = [] if draw_std else np.zeros((len(binnumber)),dtype=int)
    for i in range(1,bins+1):
        r = df.loc[df['binnumber'] == i, 'association_counter']   
        if r.shape[0] == 0:
            ret_y.append(0) 
        else:
            ret_y.append(statistics.mean(r))
        
        # collect std
        if draw_std:
            r = df.loc[df['binnumber'] == i, 'association_counter']
            if r.shape[0] < 2:
                std.append(0) 
            else:
                std.append(statistics.stdev(r))        

    # Create dataframe for the plot
    data_tuples = list( zip( np.nan_to_num(ret_x), np.nan_to_num(ret_y), np.log(size+1)*50, std ) )
    df2 = pd.DataFrame(data_tuples, columns=['cable_length','association_counter', 'bin_counter', 'stdev_assoc'])
    error_y = 'stdev_assoc' if draw_std else None

    # Plot 
    username = os.path.basename(save_path)
    fig = px.scatter(df2, x="cable_length", y="association_counter", color="association_counter",size="bin_counter",
                        error_y=error_y, log_x=log_x, log_y=log_y, title=username+' - Error analysis',
                        color_continuous_scale=px.colors.sequential.Bluered)
    fig.layout.showlegend = False
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(font=dict(size=25))

    fig.write_image(os.path.join(save_path,username+"_error.svg"), width=shape[0], height=shape[1])
    if show:
      fig.show()  


def association_plot_3d(assoc_file, save_path, show=True, draw_plane=True, log_x=True, log_y=True, color="association_type",
                        symbol="category", shape=[800,800]):
    """Plot 3D errors.
    
       Parameters
       ----------
       assoc_file : str
           Path where to the association file. 

       save_path : str
           Directory to store the plot into. 

       show : bool, optional
           Wheter to show or not the plot after saving. 

       draw_plane : bool, optional
           Wheter to draw or not the plane in z=0 to see better the 'over' and 'under' segmentations. 

       log_x : bool, optional
           True to apply log in 'x' axis. 

       log_y : bool, optional
           True to apply log in 'y' axis. 
    
       color : str, optional
           Property to based the color selection. 

       symbol : str, optional
           Property to based the symbol selection. 

       shape : 2d array of ints, optional
           Defines the shape of the plot.
    """

    axis_propety = ['volume','cable_length','association_counter']

    df = pd.read_csv(assoc_file, index_col=False)
    fig = px.scatter_3d(df, x=axis_propety[0], y=axis_propety[1], z=axis_propety[2], color=color,
                        symbol=symbol, log_x=log_x, log_y=log_y)

    if draw_plane:
        height = 0
        x= np.linspace(df['volume'].min(), df['volume'].max(), 100)
        y= np.linspace(df['cable_length'].min(), df['cable_length'].max(), 100)
        z= height*np.ones((100,100))
        mycolorscale = [[0, '#f6ff00'], [1, '#f6ff00']]
        fig.add_surface(x=x, y=y, z=z, colorscale=mycolorscale, opacity=0.3, showscale=False)

    username = os.path.basename(save_path)
    fig.update_layout(title=username+' - Error analysis', scene = dict(xaxis_title='Volume', yaxis_title='Cable length',
                      zaxis_title='Associations'), autosize=False, width=shape[0], height=shape[1],
                      margin=dict(l=65, r=50, b=65, t=90), font=dict(size=25))

    fig.write_image(os.path.join(save_path,username+"_error_3D.svg"))
    if show:
      fig.show()


def association_multiple_predictions(prediction_dirs, assoc_stats_file, show=True, order=[], shape=[1100,500]):
    """Create a plot that gather multiple prediction information.
    
       Parameters
       ----------
       prediction_dirs : str
           Directory where all the predictions folders are placed. 

       assoc_stats_file : str
           Name of the association stats file. 

       show : bool, optional
           Wheter to show or not the plot after saving. 

       order : list of str, optional
           Order each prediction based on a given list. The names need to match the names used for
           each prediction folder. E.g ['prediction1', 'prediction2'].

       shape : 2d array of ints, optional
           Defines the shape of the plot.
    """

    dataframes = []
    for folder in prediction_dirs:
        df_method = pd.read_csv(os.path.join(folder,assoc_stats_file), index_col=0)
        df_method['method'] = os.path.basename(folder)
        dataframes.append(df_method)

        # Initialize in the first loop 
        if 'ncategories' not in locals():
            ncategories = df_method.shape[0]
            categories_names = [df_method.iloc[i].name for i in range(ncategories)]

    df = pd.concat([val for val in dataframes])
    df = df.sort_values(by=['method'], ascending=False)

    # Change strings to integers
    df["one-to-one"] = df["one-to-one"].str[1:-1].astype(int)
    df["missing"] = df["missing"].str[1:-1].astype(int)
    df["over-segmentation"] = df["over-segmentation"].str[1:-1].astype(int)
    df["under-segmentation"] = df["under-segmentation"].str[1:-1].astype(int)
    df["many-to-many"] = df["many-to-many"].str[1:-1].astype(int)

    if len(order)>1:
        df['position'] = 0
        for i, name in enumerate(order):
            df.loc[df['method'] == name, 'position'] = i
        df = df.sort_values(by=['position'], ascending=True)

    # Order colors
    colors = px.colors.qualitative.Plotly
    tmp = colors[0]
    colors[0] = colors[2]
    colors[2] = tmp

    # Create a plot for each type of category 
    for i in range(ncategories):
        fig = px.bar(df.loc[categories_names[i]], x="method", y=["one-to-one", "missing", "over-segmentation",
                     "under-segmentation", "many-to-many"], title="Association performance comparison",
                     color_discrete_sequence=colors, labels={'method':'Methods', 'value':'Number of instances'})
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=0.835, title_text='',
                                      font=dict(size=13)), font=dict(size=22))
        fig.update_xaxes(tickangle=45)
        fig.write_image(os.path.join(os.path.dirname(folder),"all_methods_"+categories_names[i]+"_errors.svg"),
                        width=shape[0], height=shape[1])
        if show:
            fig.show()

