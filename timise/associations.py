import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable

from .utils import str_list_to_ints_list

def calculate_associations_from_map(assoc_file, gt_stats_file, assoc_stats_file, final_file, log_prefix_str=''):
    """Calculate associations between instances. Based on the idea presented in `Assessment of deep learning
       algorithms for 3D instance segmentation of confocal image datasets
       <https://www.biorxiv.org/content/10.1101/2021.06.09.447748v1.full>`_.

       Parameters
       ----------
       assoc_file : str
           Path to the file containing the predicted instances.

       gt_stats_file : str
           Path to the prediction statistics.

       assoc_stats_file : str
           Path to the file containing the predicted instances.

       final_file : str
           Path where the final error file will be stored.

       log_prefix_str : str, optional
           Prefix to be prepended to all prints. 
    """
    assoc_df = pd.read_csv(assoc_file, index_col=False)
    # Categorize associations and save again 
    if not 'association_type' in assoc_df:
        assoc_df['predicted'] = str_list_to_ints_list(assoc_df, 'predicted', void_to_number=False)
        assoc_df['gt'] = str_list_to_ints_list(assoc_df, 'gt', void_to_number=False)
        assoc_df['association_type'] = assoc_df.apply(lambda row: lab_association(row), axis=1)
        assoc_df = assoc_df.sort_values(by=['association_type'])

        assoc_df.to_csv(assoc_file, index=False)

    # Creating association statistics to not do it everytime the user wants to print them
    # and modifying the predictions stats to insert information about the association
    final_df = pd.read_csv(gt_stats_file, index_col=False)
    final_df['counter'] = 0
    final_df['association_type'] = 'over-segmentation'
    
    _labels = np.array(final_df['label'].tolist())
    _counter = np.array(final_df['counter'].tolist())
    pred_number = np.zeros(len(final_df['counter']), dtype=np.uint16)
    gt_number = np.zeros(len(final_df['counter']), dtype=np.uint16)
    _association_type = np.array(final_df['association_type'].tolist())
    if 'category' in final_df.columns:
        cell_statistics = []
        categories = pd.unique(final_df['category']).tolist()
        for i in range(len(categories)):
            cell_statistics.append({'one-to-one': 0, 'missing': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'many-to-many': 0, 'background': 0})
    else:
        cell_statistics = {'one-to-one': 0, 'missing': 0, 'over-segmentation': 0, 'under-segmentation': 0, 'many-to-many': 0, 'background': 0}

    assoc_df = assoc_df.reset_index()
    for index, row in assoc_df.iterrows():
        gt_instances = row['gt']
        if not type(gt_instances) is list:
            gt_instances = row['gt'].replace('[',' ').replace(']',' ').replace(',','').split()
            gt_instances = [int(x) for x in gt_instances]
        pred_instances = row['predicted']
        if not type(pred_instances) is list:
            pred_instances = row['predicted'].replace('[',' ').replace(']',' ').replace(',','').split()
            pred_instances = [int(x) for x in pred_instances]
            
        if gt_instances == []:
            if type(cell_statistics) is list:
                # Add FPs in the first category 
                cell_statistics[0]['background'] += 1
            else:
                cell_statistics['background'] += 1
        else:
            for gt_ins in gt_instances:
                if type(cell_statistics) is list:
                    result = final_df[final_df['label']==gt_ins]
                    tag = result['category'].iloc[0]
                    cell_statistics[categories.index(tag)][row['association_type']] += 1
                else:
                    cell_statistics[row['association_type']] += 1

        if row['association_type'] == 'over-segmentation':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            pred_number[itemindex] = len(pred_instances)
            gt_number[itemindex] = 1
            _association_type[itemindex] = 'over'
        elif row['association_type'] == 'under-segmentation':
            for ins in gt_instances:
                itemindex = np.where(_labels==ins)
                _counter[itemindex] += 1
                pred_number[itemindex] = 1
                gt_number[itemindex] = len(gt_instances)
                _association_type[itemindex] = 'under'
        elif row['association_type'] == 'many-to-many':
            pred_count = len(pred_instances)
            t = 'many (over)' if pred_count >= len(gt_instances) else 'many (under)'
            for ins in gt_instances:
                itemindex = np.where(_labels==ins)
                _counter[itemindex] += 1
                pred_number[itemindex] = len(pred_instances)
                gt_number[itemindex] = len(gt_instances)
                _association_type[itemindex] = t
        elif row['association_type'] == 'one-to-one':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'one-to-one'
        elif row['association_type'] == 'missing':
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'missing'
        elif row['association_type'] == 'background':
            itemindex = np.where(_labels==gt_instances)
            _association_type[itemindex] = 'background'
        else:
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'other'

    final_df['counter'] = _counter
    final_df['pred_number'] = pred_number
    final_df['gt_number'] = gt_number
    final_df['association_type'] = _association_type

    if type(cell_statistics) is list:
        assoc_summary_df = pd.DataFrame.from_dict([ {k:[v] for k,v in a.items()} for a in cell_statistics] )
        assoc_summary_df = assoc_summary_df.set_axis(categories)
    else:
        assoc_summary_df = pd.DataFrame.from_dict([{k:[v] for k,v in cell_statistics.items()}])
        assoc_summary_df = assoc_summary_df.set_axis(['all'])

    print(log_prefix_str+"\tSaving associations metrics results in {}".format(final_file))
    print(log_prefix_str+"\tSaving associations summary in {}".format(assoc_stats_file))
    # Save association final results and a summary to print it easily 
    final_df.to_csv(final_file, index=False)
    assoc_summary_df.to_csv(assoc_stats_file)


def lab_association(row):
    """Determines the association type."""   
    if len(row['predicted']) == 0:
        return 'missing'
    elif len(row['predicted']) == 1:
        if len(row['gt']) == 1:
            return 'one-to-one'
        elif len(row['gt']) == 0:
            return 'background'
        else:
            return 'under-segmentation'
    else:
        if len(row['gt']) == 1:
            return 'over-segmentation'
        else:
            return 'many-to-many'


def print_association_stats(stats_csv, map_aux_dir, show_categories=False):
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

    pred_id = np.load(os.path.join(map_aux_dir, "pred_ui.npy"))
    num_pred_instances = len(pred_id)

    df_out = pd.read_csv(stats_csv, index_col=0)
    total_instances_all = 0
    df_len = df_out.shape[0]

    total_assoc = [0, 0, 0, 0, 0]
    pred_total_background = [0]
    for i in range(df_len):
        total_instances = 0
        cell_statistics = {}
        pred_cell_statistics = {}
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
        if isinstance(df_out['background'][i], str):
            pred_cell_statistics['background'] = int(df_out['background'][i][1:][:-1])
        else:
            pred_cell_statistics['background'] = df_out['background'][i]
        if i != 0: pred_cell_statistics['background'] = '-'

        total_instances += cell_statistics['one-to-one']+cell_statistics['missing'] \
                           +cell_statistics['over-segmentation']+cell_statistics['under-segmentation'] \
                           +cell_statistics['many-to-many']
        total_instances_all += total_instances
        if i == 0:
            extra_column = ['category',] if df_len > 1 else []
            t = PrettyTable(extra_column+['',]+list(cell_statistics.keys())+['Total']+[' ',]+['background'])
        extra_column = [df_out.iloc[i].name,] if df_len > 1 else []
        t.add_row(extra_column+['Count',]+list(cell_statistics.values())+[total_instances]+['    ',]+[pred_cell_statistics['background']])

        total_assoc[0] += cell_statistics['one-to-one']
        total_assoc[1] += cell_statistics['missing']
        total_assoc[2] += cell_statistics['over-segmentation']
        total_assoc[3] += cell_statistics['under-segmentation']
        total_assoc[4] += cell_statistics['many-to-many']
        if i == 0: pred_total_background[0] += pred_cell_statistics['background']

        # Percentages
        cell_statistics = {state: np.around((val/total_instances)*100, 2) for state, val in cell_statistics.items()}
        extra_column = [' ',] if df_len > 1 else []
        t.add_row(extra_column+['%',]+list(cell_statistics.values())+[' ',' ', ' '])

    if df_len > 1 and show_categories:
        t.add_row(['',]*(3+len(cell_statistics.values()) )+[' ',' '])
        t.add_row(['TOTAL','Count',]+total_assoc+[total_instances_all,]+[' ',]+pred_total_background)
        per_back_instances = np.around((pred_total_background[0]/num_pred_instances)*100, 2)
        t.add_row([' ','%',]+[np.around((val/total_instances_all)*100, 2) for val in total_assoc]+['100',' ',per_back_instances])
    else:
        t = PrettyTable([' ',]+list(cell_statistics.keys())+['Total'])
        t.add_row(['Count',]+total_assoc+[total_instances_all,])
        t.add_row(['%',]+[np.around((val/total_instances_all)*100, 2) for val in total_assoc]+[total_instances_all,])

    # Header positioning
    txt = "Ground truth associations"
    txt2 = "Prediction"
    txt3 = "false positives"

    # Center txt 
    line = t.get_string().split(os.linesep)[0].split('+')[:-3]
    c = 0
    for v in line:
        c += len(v)   
    c += len(line)
    txt = txt.center(c)

    # Center txt2 and txt3
    line = t.get_string().split(os.linesep)[0].split('+')[-3:]
    c = 0
    for i in range(1,len(line)):
        c += len(v)   
    c += len(line)
    txt2 = txt2.center(c)
    txt3 = txt3.center(c)
    txt = txt + ' '*(len(line[0])-2)

    # Print headers
    print(' '*len(txt)+ txt2)
    print(txt+txt3)
    print(t)

