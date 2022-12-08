import os
import statistics
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from collections import OrderedDict
from scipy import stats
import plotly.express as px


class Plotter():
    """ Sets up a seaborn environment and holds the functions for plotting our figures. """

    def __init__(self, quality=1, xaxis_range=-1, yaxis_range=-1, match_th=0.75):
        # Set mpl DPI in case we want to output to the screen / notebook
        mpl.rcParams['figure.dpi'] = 150
        
        # Seaborn color palette
        sns.set_palette('muted', 10)
        current_palette = sns.color_palette()
        
        # Seaborn style
        sns.set(style="whitegrid")

        self.colors_main = OrderedDict({
            'one-to-one': current_palette[2],
            'missing': current_palette[3],
            'over': current_palette[8],
            'under': current_palette[6],
            'many': current_palette[9],
            'background': current_palette[4],
        })
        self.vplot_pallete = OrderedDict({
            'FP': current_palette[0],
            'FN': current_palette[1]})
        self.quality = quality
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.match_th = match_th 

    def error_pie_multiple_predictions(self, prediction_dirs, assoc_file, matching_file, show=True):
        """Calls error_pie for each prediction. """
        for folder in prediction_dirs:
            plot_dir = os.path.join(folder, "plots")
            _assoc_file = os.path.join(folder, assoc_file)
            _matching_file = os.path.join(folder, matching_file)

            self.combined_plot(_assoc_file, _matching_file, plot_dir, show=show, hide_correct=True)

    def error_pie(self, assoc_file, matching_file, save_path, show=True, hide_correct=True):
        """
        Create a combination of a pie and two bar plots. Adapted from TIDE: 

        https://github.com/dbolya/tide     

        Parameters
        ----------
        assoc_file : str
            Path to the associations file.

        matching_file : str
            Path to the matching file.
        
        save_path : str
            Folder to save generated plots.

        show : bool, optional
            Wheter to show or not the plot after saving.


        hide_correct : bool, optional
            Whether to hide or not the correct matches and TP. 
        """
        model_name = os.path.basename(os.path.dirname(save_path))

        print("Creating combined plot for {} . . .".format(model_name))

        df = pd.read_csv(assoc_file, index_col=0)
        mat_df = pd.read_csv(matching_file, index_col=0)
        
        high_dpi = int(500*self.quality)
        low_dpi  = int(300*self.quality)

        error_types = list(df.columns)
        categories = df.index.values.tolist()
        if 'all' not in categories:
            categories += ['all',]

        colors = self.colors_main.copy()

        error_types.remove('background')
        del colors['background']
        if hide_correct:
            error_types.remove('one-to-one')
            del colors['one-to-one']

        for cat in categories:
            if cat != 'all':
                df2 = df.loc[cat]
                mat_df2 = mat_df.loc[cat]
                fname = cat+'_'
            else:
                df2 = df.copy()
                mat_df2 = mat_df.loc['total']
                fname = ''
            mat_df2 = mat_df2[mat_df2['thresh'] == self.match_th] 

            error_sum = 0
            error_number_per_type = {}
            for error in error_types:
                error_number_per_type[error] = error_sum
                if isinstance(df2[error], str):
                    error_sum += int(df2[error].replace('[',' ').replace(']',' ').replace(',',''))
                else:
                    for i in range(len(df2[error])):
                        error_sum += int(df2[error][i].replace('[',' ').replace(']',' ').replace(',',''))
                error_number_per_type[error] = error_sum-error_number_per_type[error]
            error_sizes = [error_number_per_type[e] / error_sum for e in error_types]
            error_dfs = pd.DataFrame((k2, v2) for k2, v2 in error_number_per_type.items())
            error_dfs.columns = ['Error Type', 'Count']
            
            os.makedirs(os.path.join(save_path, cat), exist_ok=True)

            # Pie plot
            fig, ax = plt.subplots(1, 1, figsize=(11, 11), dpi=high_dpi)
            patches, outer_text, inner_text = ax.pie(error_sizes, colors=colors.values(), labels=error_types,
                                                    autopct='%1.1f%%', startangle=90)
            for text in outer_text + inner_text:
                text.set_text('')
            for i in range(len(colors)):
                if error_sizes[i] > 0.05:
                    inner_text[i].set_text(list(colors.keys())[i])
                inner_text[i].set_fontsize(48)
                inner_text[i].set_fontweight('bold')
            ax.axis('equal')
            
            plt.title(model_name, fontdict={'fontsize': 60, 'fontweight': 'bold'})
            pie_path = os.path.join(save_path, cat, '{}_{}pie.png'.format(model_name, fname))
            plt.savefig(pie_path, bbox_inches='tight', dpi=low_dpi)            
            plt.close()

            # horizontal bar plot for main error types
            fig, ax = plt.subplots(1, 1, figsize = (6, 5), dpi=high_dpi)
            sns.barplot(data=error_dfs, x='Count', y='Error Type', ax=ax,
                        palette=colors.values())
            if self.xaxis_range is not None:
                ax.set_xlim(self.xaxis_range[0], self.xaxis_range[1])
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.setp(ax.get_xticklabels(), fontsize=28)
            plt.setp(ax.get_yticklabels(), fontsize=36)
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            
            ax.grid(False)
            sns.despine(left=True, bottom=True, right=True)
            hbar_path = os.path.join(save_path, cat, '{}_{}hbar.png'.format(model_name, fname))
            plt.savefig(hbar_path, bbox_inches='tight', dpi=low_dpi)
            plt.close()

            # vertical bar plot for special error types
            data = {'FP': [mat_df2['fp'][0]], 'FN': [mat_df2['fn'][0]]}
            xlabels = ['FP', 'FN']
            if not hide_correct:
                xlabels = ['TP',] + xlabels
                data['TP'] = [mat_df2['tp'][0]]
            df_aux = pd.DataFrame.from_dict(data, orient='index')
            df_aux.reset_index(inplace=True)
            df_aux.columns=["Metric", "Count"]
            fig, ax = plt.subplots(1, 1, figsize = (2, 5), dpi=high_dpi)
            sns.barplot(data=df_aux, x="Metric", y="Count", ax=ax, palette=self.vplot_pallete.values())
            if self.yaxis_range is not None:
                ax.set_ylim(self.yaxis_range[0], self.yaxis_range[1])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(xlabels)
            plt.setp(ax.get_xticklabels(), fontsize=36)
            plt.setp(ax.get_yticklabels(), fontsize=28)
            ax.grid(False)
            sns.despine(left=True, bottom=True, right=True)
            vbar_path = os.path.join(save_path, cat, '{}_{}vbar.png'.format(model_name, fname))
            plt.savefig(vbar_path, bbox_inches='tight', dpi=low_dpi)
            plt.close()

            # get each subplot image
            pie_im  = cv2.imread(pie_path)
            hbar_im = cv2.imread(hbar_path)
            vbar_im = cv2.imread(vbar_path)

            # pad the hbar image vertically
            if vbar_im.shape[0] - hbar_im.shape[0] > 0:
                hbar_im = np.concatenate([np.zeros((vbar_im.shape[0] - hbar_im.shape[0], hbar_im.shape[1], 3)) + 255, hbar_im],
                                        axis=0)
            else:
                vbar_im = np.concatenate([np.zeros((hbar_im.shape[0] - vbar_im.shape[0], vbar_im.shape[1], 3)) + 255, vbar_im],
                                        axis=0)
            summary_im = np.concatenate([hbar_im, vbar_im], axis=1)
            
            # pad summary_im
            if summary_im.shape[1]<pie_im.shape[1]:
                lpad, rpad = int(np.ceil((pie_im.shape[1] - summary_im.shape[1])/2)), \
                        int(np.floor((pie_im.shape[1] - summary_im.shape[1])/2))
                summary_im = np.concatenate([np.zeros((summary_im.shape[0], lpad, 3)) + 255,
                                        summary_im,
                                        np.zeros((summary_im.shape[0], rpad, 3)) + 255], axis=1)
                
            # pad pie_im
            else:
                lpad, rpad = int(np.ceil((summary_im.shape[1] - pie_im.shape[1])/2)), \
                        int(np.floor((summary_im.shape[1] - pie_im.shape[1])/2))
                pie_im = np.concatenate([np.zeros((pie_im.shape[0], lpad, 3)) + 255,
                                        pie_im, 
                                        np.zeros((pie_im.shape[0], rpad, 3)) + 255], axis=1)
            
            summary_im = np.concatenate([pie_im, summary_im], axis=0)
            
            cv2.imwrite(os.path.join(save_path, cat, '{}_{}summary.png'.format(model_name, fname)), summary_im)
            
        if show:
            fig = plt.figure()
            ax = plt.axes([0,0,1,1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
            plt.show()
            
        
    def association_plot_2d(self, final_file, save_path, show=True, bins=30, draw_std=True, log_x=False, 
        log_y=False, font_size=25, shape=[1100,500]):
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

        font_size : int, optional
            Size of the font. 

        shape : 2d array of ints, optional
            Defines the shape of the plot.
        """
        print("Creating association plot . . .")

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
        fig.update_layout(font=dict(size=font_size), xaxis_range=self.xaxis_range, yaxis_range=self.yaxis_range)

        fig.write_image(os.path.join(save_path,username+"_error.svg"), width=shape[0], height=shape[1])
        if show:
            fig.show()


    def association_plot_3d(self, assoc_file, save_path, show=True, draw_plane=True, log_x=True, log_y=True, 
        color="association_type", symbol="category", font_size=25, shape=[800,800]):
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

        font_size : int, optional
            Size of the font. 

        shape : 2d array of ints, optional
            Defines the shape of the plot.
        """
        print("Creating association plot . . .")

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
                        xaxis_range=self.xaxis_range, yaxis_range=self.yaxis_range)

        fig.write_image(os.path.join(save_path,username+"_error_3D.svg"))
        if show:
            fig.show()


    def association_multiple_predictions(self, prediction_dirs, assoc_stats_file, show=True, show_categories=False,
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
        print("Creating multiple prediction association plots . . .")
        
        dataframes = []
        for folder in prediction_dirs:
            df_method = pd.read_csv(os.path.join(folder,assoc_stats_file), index_col=0)
            df_method['method'] = os.path.basename(folder)
            dataframes.append(df_method)

            # Initialize in the first loop
            if 'ncategories' not in locals():
                ncategories = df_method.shape[0]
                category_names = [df_method.iloc[i].name for i in range(ncategories)]

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
                fig = px.bar(df.loc[category_names[i]], x="method", y=["one-to-one", "missing", "over-segmentation",
                            "under-segmentation", "many-to-many", "background"], title="Association performance ("+category_names[i]+")",
                            color_discrete_sequence=colors, labels={'method':'Methods', 'value':'Number of instances'},
                            width=shape[0], height=shape[1])
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=0.835, title_text='',
                                            font=dict(size=13)), font=dict(size=22))
                fig.update_xaxes(tickangle=45)
                fig.write_image(os.path.join(os.path.dirname(folder),"all_methods_"+category_names[i]+"_errors.svg"),
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


    def error_bar_multiple_predictions(self, prediction_dirs, assoc_stats_file, matching_stats_file,
        show=True, order=[], shape=[1100,500]):
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
        print("Creating multiple prediction association plots . . .")

        dataframes = []
        for folder in prediction_dirs:
            df_method = pd.read_csv(os.path.join(folder,assoc_stats_file), index_col=0)
            df_method['method'] = os.path.basename(folder)
            dataframes.append(df_method)

            # Initialize in the first loop
            if 'ncategories' not in locals():
                ncategories = df_method.shape[0]
                category_names = [df_method.iloc[i].name for i in range(ncategories)]

        df = pd.concat([val for val in dataframes])
        df = df.sort_values(by=['method'], ascending=False)

        # Change strings to integers
        df["background"] = df["background"].str[1:-1].astype(int)
        df["missing"] = df["missing"].str[1:-1].astype(int)
        df["over-segmentation"] = df["over-segmentation"].str[1:-1].astype(int)
        df["under-segmentation"] = df["under-segmentation"].str[1:-1].astype(int)
        df["many-to-many"] = df["many-to-many"].str[1:-1].astype(int)

        # Convert index to column
        df.reset_index(inplace=True)
        df['index'] = df['index'].str.capitalize()

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

        # Association error bar plot
        fig = px.histogram(df, x="index", y=["missing", "over-segmentation", "under-segmentation", "many-to-many"], 
            title="Association performance", barmode='group', color="method", histfunc='sum',
            color_discrete_sequence=colors, width=shape[0], height=shape[1])
        fig.update_layout(xaxis_title="Cable length", yaxis_title="Association Error",
            legend_title="Methods", font=dict(size=18))
        fig.update_yaxes(range=self.yaxis_range)
        fig.write_image(os.path.join(os.path.dirname(folder),"association_errors_bars.svg"),
                        width=shape[0], height=shape[1])
        
        # Load all matching dataframes into one
        all_dfs = []

        for folder in prediction_dirs:
            mat_df = pd.read_csv(os.path.join(folder, matching_stats_file))
            mat_df['method'] = os.path.basename(folder)
            all_dfs.append(mat_df)
        mat_df = pd.concat(all_dfs)
        del all_dfs

        mat_df = mat_df[mat_df["category"]!="total"]
        mat_df = mat_df[mat_df['thresh'] == self.match_th] 

        # False negatives bar plot
        fig2 = px.histogram(mat_df, x="category", y="fn", title="False Negatives", barmode='group', color="method",
            color_discrete_sequence=colors, width=shape[0], height=shape[1])
        fig2.update_layout(xaxis_title="Cable length", yaxis_title="False Negatives",
            legend_title="Methods", font=dict(size=18))
        fig2.update_yaxes(range=self.yaxis_range)
        fig2.write_image(os.path.join(os.path.dirname(folder),"fn_errors_bars.svg"),
                        width=shape[0], height=shape[1])
        if show:
            fig.show()
            fig2.show()
