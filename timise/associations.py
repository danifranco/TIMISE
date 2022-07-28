import os
import statistics
import pandas as pd
import numpy as np
from scipy import stats
from prettytable import PrettyTable
import plotly.express as px

from .utils import str_list_to_ints_list

def calculate_associations_from_map(assoc_file, gt_stats_file, assoc_stats_file, final_file, verbose=True):
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

       verbose : bool, optional
           Wheter to be more verbose.
    """
    assoc_df = pd.read_csv(assoc_file, index_col=False)
    out_dir = os.path.dirname(final_file)
    
    # Categorize associations and save again 
    if not 'association_type' in assoc_df:
        assoc_df['predicted'] = str_list_to_ints_list(assoc_df, 'predicted', void_to_number=False)
        assoc_df['gt'] = str_list_to_ints_list(assoc_df, 'gt', void_to_number=False)
        assoc_df['association_type'] = assoc_df.apply(lambda row: lab_association(row), axis=1)
        assoc_df = assoc_df.sort_values(by=['association_type'])

        assoc_df.to_csv(os.path.join(out_dir, assoc_file), index=False)

    # Creating association statistics to not do it everytime the user wants to print them
    # and modifying the predictions stats to insert information about the association
    final_df = pd.read_csv(gt_stats_file, index_col=False)
    final_df['counter'] = 0
    final_df['association_counter'] = 0
    final_df['association_type'] = 'over-segmentation'
    
    _labels = np.array(final_df['label'].tolist())
    _counter = np.array(final_df['counter'].tolist())
    _association_counter = np.array(final_df['association_counter'].tolist(), dtype=np.float32)
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
            _association_counter[itemindex] = len(pred_instances)
            _association_type[itemindex] = 'over'
        elif row['association_type'] == 'under-segmentation':
            for ins in gt_instances:
                itemindex = np.where(_labels==ins)
                _counter[itemindex] += 1
                _association_counter[itemindex] = -len(gt_instances)
                _association_type[itemindex] = 'under'
        elif row['association_type'] == 'many-to-many':
            pred_count = len(pred_instances)
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
        elif row['association_type'] == 'background':
            itemindex = np.where(_labels==gt_instances)
            _association_type[itemindex] = 'background'
        else:
            itemindex = np.where(_labels==gt_instances)
            _counter[itemindex] += 1
            _association_type[itemindex] = 'other'

    final_df['counter'] = _counter
    final_df['association_counter'] = _association_counter
    final_df['association_type'] = _association_type
  
    if type(cell_statistics) is list:
        assoc_summary_df = pd.DataFrame.from_dict([ {k:[v] for k,v in a.items()} for a in cell_statistics] )
        assoc_summary_df = assoc_summary_df.set_axis(categories)
    else:
        assoc_summary_df = pd.DataFrame.from_dict([{k:[v] for k,v in cell_statistics.items()}])
        assoc_summary_df = assoc_summary_df.set_axis(['all'])

    # Save association final results and a summary to print it easily 
    final_df.to_csv(final_file, index=False)
    assoc_summary_df.to_csv(os.path.join(out_dir, assoc_stats_file))


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

    max_str_size = 0
    total_assoc = [0, 0, 0, 0, 0, 0]
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
        if isinstance(df_out['background'][i], str):
            cell_statistics['background'] = int(df_out['background'][i][1:][:-1])
        else:
            cell_statistics['background'] = df_out['background'][i]
        total_instances += cell_statistics['one-to-one']+cell_statistics['missing'] \
                           +cell_statistics['over-segmentation']+cell_statistics['under-segmentation'] \
                           +cell_statistics['many-to-many']+cell_statistics['background']
        total_instances_all += total_instances
        if i == 0:
            extra_column = ['category',] if df_len > 1 else []
            t = PrettyTable(extra_column+[' ',]+list(cell_statistics.keys())+['Total'])
        extra_column = [df_out.iloc[i].name,] if df_len > 1 else []
        t.add_row(extra_column+['Count',]+list(cell_statistics.values())+[total_instances])

        total_assoc[0] += cell_statistics['one-to-one']
        total_assoc[1] += cell_statistics['missing']
        total_assoc[2] += cell_statistics['over-segmentation']
        total_assoc[3] += cell_statistics['under-segmentation']
        total_assoc[4] += cell_statistics['many-to-many']
        total_assoc[5] += cell_statistics['background']

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
    else:
        t = PrettyTable([' ',]+list(cell_statistics.keys())+['Total'])
        t.add_row(['Count',]+total_assoc+[total_instances_all,])
        t.add_row(['%',]+[np.around((val/total_instances_all)*100, 2) for val in total_assoc]+[total_instances_all,])

    txt = "Associations"
    txt = txt.center(t.get_string().find(os.linesep))
    print(txt)
    print(t)


def association_plot_2d(final_file, save_path, show=True, bins=30, draw_std=True, xaxis_range=None,
                        yaxis_range=None, log_x=False, log_y=False, font_size=25, shape=[1100,500]):
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

       xaxis_range : array of 2 floats, optional
           Range of x axis.

       yaxis_range : array of 2 floats, optional
           Range of x axis.

       log_x : bool, optional
           True to apply log in 'x' axis.

       log_y : bool, optional
           True to apply log in 'y' axis.

       font_size : int, optional
           Size of the font. 

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
                        color_continuous_scale=px.colors.sequential.Bluered, width=shape[0], height=shape[1])
    fig.layout.showlegend = False
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(font=dict(size=font_size), xaxis_range=xaxis_range, yaxis_range=yaxis_range)

    fig.write_image(os.path.join(save_path,username+"_error.svg"), width=shape[0], height=shape[1])
    if show:
      fig.show()


def association_plot_3d(assoc_file, save_path, show=True, draw_plane=True, xaxis_range=None,
                        yaxis_range=None, log_x=True, log_y=True, color="association_type",
                        symbol="category", font_size=25, shape=[800,800]):
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

       xaxis_range : array of 2 floats, optional
           Range of x axis.

       yaxis_range : array of 2 floats, optional
           Range of x axis.

       log_x : bool, optional
           True to apply log in 'x' axis.

       log_y : bool, optional
           True to apply log in 'y' axis.

       color : str, optional
           Property to based the color selection.

       symbol : str, optional
           Property to based the symbol selection.

       font_size : int, optional
           Size of the font. 

       shape : 2d array of ints, optional
           Defines the shape of the plot.
    """
    axis_propety = ['volume','cable_length','association_counter']

    df = pd.read_csv(assoc_file, index_col=False)
    fig = px.scatter_3d(df, x=axis_propety[0], y=axis_propety[1], z=axis_propety[2], color=color,
                        symbol=symbol, log_x=log_x, log_y=log_y, width=shape[0], height=shape[1])

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
                      margin=dict(l=65, r=50, b=65, t=90), font=dict(size=font_size),
                      xaxis_range=xaxis_range, yaxis_range=yaxis_range)

    fig.write_image(os.path.join(save_path,username+"_error_3D.svg"))
    if show:
      fig.show()


def association_multiple_predictions(prediction_dirs, assoc_stats_file, show=True, show_categories=False,
                                     order=[], shape=[1100,500]):
    """Create a plot that gather multiple prediction information.

       Parameters
       ----------
       prediction_dirs : str
           Directory where all the predictions folders are placed.

       assoc_stats_file : str
           Name of the association stats file.

       show : bool, optional
           Wheter to show or not the plot after saving.

       show_categories :  bool, optional
           Whether to create a plot per category or just one.

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
    df["background"] = df["background"].str[1:-1].astype(int)

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

    if show_categories:
        # Create a plot for each type of category
        for i in range(ncategories):
            fig = px.bar(df.loc[categories_names[i]], x="method", y=["one-to-one", "missing", "over-segmentation",
                         "under-segmentation", "many-to-many", "background"], title="Association performance ("+categories_names[i]+")",
                         color_discrete_sequence=colors, labels={'method':'Methods', 'value':'Number of instances'},
                         width=shape[0], height=shape[1])
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=0.835, title_text='',
                                        font=dict(size=13)), font=dict(size=22))
            fig.update_xaxes(tickangle=45)
            fig.write_image(os.path.join(os.path.dirname(folder),"all_methods_"+categories_names[i]+"_errors.svg"),
                            width=shape[0], height=shape[1])
            if show:
                fig.show()
    else:
        df = df.groupby('method', sort=False).sum()
        df.reset_index(inplace=True)
        fig = px.bar(df, x="method", y=["one-to-one", "missing", "over-segmentation",
                     "under-segmentation", "many-to-many", "background"], title="Association performance comparison",
                     color_discrete_sequence=colors, labels={'method':'Methods', 'value':'Number of instances'},
                     width=shape[0], height=shape[1])
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=0.835, title_text='',
                                    font=dict(size=13)), font=dict(size=22))
        fig.update_xaxes(tickangle=45)
        fig.write_image(os.path.join(os.path.dirname(folder),"all_methods_errors.svg"),
                        width=shape[0], height=shape[1])
        if show:
            fig.show()

